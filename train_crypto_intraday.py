#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
加密分时细节重训: 小时线特征 + E1 beta 中性标签

数据: crypto_daily (日线) + crypto_hourly (Binance 1h, fetch_crypto_intraday.py)
口径与 train_crypto_ab.py 完全一致 (训练 2018-2023 / 验证 2024+, E1 标签,
超参 lr0.03/leaves31/min_data1000/ff0.6)，只加 intraday 特征:

  rv24/rv72/rv168  已实现波动率 (滚动 24/72/168 小时 log 收益平方和开方)
  mom4/mom8/mom24  尾盘动量 (最后 4/8/24 小时收益, 锚定 23:00 bar)
  on_ret/day_ret   隔夜 (00-08 UTC) 与白天 (08-23) 收益拆分
  vol_conc         最后 4 小时成交量占比
  rangepos         日内收盘在 24h 高低区间的位置
  vwap_dev         收盘相对当日 VWAP 偏离

对照组:
  E1              tech40 (复现生产口径)
  E1Rv            tech40 + rv24/rv168 + mom4/mom8 (小特征集, 4 列)
  E1Id            tech40 + 全部 intraday (11 列)

评价: 验证集 RankIC/ICIR (R_ex 口径), 分年 IC, Top10 residual 年化。
胜出者写入 production/prod_crypto_h{h}_tech.txt。

用法:
    python3 train_crypto_intraday.py --variants E1,E1Rv,E1Id --horizons 5,10
    nohup python3 train_crypto_intraday.py > crypto_intraday_train.log 2>&1 &
"""
import argparse
import gc
import json
import os
import sqlite3
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lgbm_train_backtest import _compute_feats  # noqa: E402
from train_markets import load_bars, rank_ic  # noqa: E402

DB = os.path.join(HERE, "stock_cache.db")
OUT = os.path.join(HERE, "markets_output")
PROD = os.path.join(HERE, "production")
FEAT_FILE = os.path.join(HERE, "production", "features_tech.json")

COST_RT = 0.002
DAYS = 365
BTC = "BTCUSDT"
BETA_WIN, BETA_MIN = 90, 60
TRAIN_END, VALID_START = "2023-12-31", "2024-01-01"

PARAMS = {
    "objective": "binary", "metric": ["auc", "binary_logloss"],
    "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 1000,
    "feature_fraction": 0.6, "bagging_fraction": 0.7, "bagging_freq": 1,
    "lambda_l2": 20, "max_bin": 127, "verbosity": -1,
}
LGB_KW = dict(n_estimators=3000, learning_rate=PARAMS["learning_rate"],
              callbacks=[lgb.early_stopping(150, verbose=False),
                         lgb.log_evaluation(0)])

ID_RV = ["rv24", "rv168", "mom4", "mom8"]
ID_ALL = ["rv24", "rv72", "rv168", "mom4", "mom8", "mom24",
          "on_ret", "day_ret", "vol_conc", "rangepos", "vwap_dev"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_hourly():
    con = sqlite3.connect(DB)
    df = pd.read_sql_query(
        "SELECT code, ts, open, high, low, close, vol FROM crypto_hourly "
        "ORDER BY code, ts", con)
    con.close()
    df["code"] = df["code"].astype(str)
    for c in ["open", "high", "low", "close", "vol"]:
        df[c] = df[c].astype("float64")
    df["hour"] = df["ts"].str[11:13].astype(int)
    df["date"] = df["ts"].str[:10]
    log(f"crypto_hourly: {len(df):,} 行, {df['code'].nunique()} 个代码")
    return df


def build_intraday(hdf):
    """日内特征: 每 (code, date) 一行, 锚定当日 23:00 bar"""
    log("计算 intraday 特征 ...")
    hd = hdf.sort_values(["code", "ts"]).reset_index(drop=True)
    hd["lr"] = np.log((hd["close"] + 1e-12) / (hd["open"] + 1e-12))
    hd = hd.sort_values(["code", "ts"]).reset_index(drop=True)
    c = hd["close"]
    lr = hd["lr"]
    code = hd["code"]
    date = hd["date"]

    feats = {}
    # --- 已实现波动率 & 尾盘动量 (rolling, 按 code) ---
    sq = lr ** 2
    for win, name in ((24, "rv24"), (72, "rv72"), (168, "rv168")):
        m = max(1, int(win * 0.7))
        s = sq.groupby(code, sort=False).transform(
            lambda x: x.rolling(win, min_periods=m).sum())
        feats[name] = np.sqrt(s).astype(np.float32)
    for lag, name in ((4, "mom4"), (8, "mom8"), (24, "mom24")):
        ret = c / c.groupby(code, sort=False).shift(lag) - 1.0
        feats[name] = ret.astype(np.float32)

    # --- 当日聚合 ---
    gkey = [code, date]
    hi24 = hd.groupby(gkey)["high"].transform("max")
    lo24 = hd.groupby(gkey)["low"].transform("min")
    vol_day = hd.groupby(gkey)["vol"].transform("sum")
    last4 = hd.groupby(gkey)["vol"].transform(lambda s: s.iloc[-4:].sum())
    feats["rangepos"] = ((c - lo24) / (hi24 - lo24 + 1e-12)).astype(np.float32)
    feats["vol_conc"] = (last4 / (vol_day + 1e-12)).astype(np.float32)

    # VWAP 当日
    cv = (hd["close"] * hd["vol"]).groupby(gkey).transform("sum")
    vwap = cv / (vol_day + 1e-12)
    feats["vwap_dev"] = ((c - vwap) / (vwap + 1e-12)).astype(np.float32)

    F = pd.DataFrame(feats)
    F["code"] = code.values
    F["date"] = date.values
    F["hour"] = hd["hour"].values

    # 隔夜 (00-08) 与 白天 (08-23): 按 hour 取收盘
    closem = (hd.pivot_table(index=["code", "date"], columns="hour",
                             values="close", aggfunc="last")
              .reset_index().sort_values(["code", "date"]))
    if 8 in closem.columns and 23 in closem.columns:
        c23_prev = closem.groupby("code")[23].shift(1)
        on = closem[8] / (c23_prev + 1e-12) - 1.0
        day = closem[23] / (closem[8] + 1e-12) - 1.0
        oc = pd.DataFrame({"code": closem["code"].values,
                           "date": closem["date"].values,
                           "on_ret": on.astype(np.float32).values,
                           "day_ret": day.astype(np.float32).values})
    else:
        oc = pd.DataFrame({"code": closem["code"].values,
                           "date": closem["date"].values,
                           "on_ret": np.nan, "day_ret": np.nan})
    day_end = F[F["hour"] == 23].drop(columns=["hour"]).copy()
    day_end = day_end.merge(oc, on=["code", "date"], how="left")
    log(f"  intraday 日表 {len(day_end):,} 行")
    return day_end


def build_beta(df):
    gb = df.groupby("code", sort=False)
    ret1 = gb["close"].pct_change()
    tmp = pd.DataFrame({"code": df["code"].astype(str).values,
                        "date": df["date"].values, "r": ret1.values})
    piv = tmp.pivot_table(index="date", columns="code", values="r")
    cols = piv.columns.difference([BTC])
    x = piv[cols]
    btc = piv[BTC]
    rm_xy = (x.mul(btc, axis=0)).rolling(BETA_WIN, BETA_MIN).mean()
    rm_x = x.rolling(BETA_WIN, BETA_MIN).mean()
    rm_y = btc.rolling(BETA_WIN, BETA_MIN).mean()
    var_y = btc.rolling(BETA_WIN, BETA_MIN).var()
    beta_piv = (rm_xy - rm_x.mul(rm_y, axis=0)).div(var_y, axis=0)
    beta_piv[BTC] = 1.0
    beta_long = beta_piv.stack().rename("beta")
    idx = pd.MultiIndex.from_arrays(
        [df["date"].values, df["code"].astype(str).values])
    out = beta_long.reindex(idx).values
    return pd.Series(out, index=df.index, dtype=np.float32)


def build_labels(df, h, beta):
    gb = df.groupby("code", sort=False)
    open1 = gb["open"].shift(-1)
    close_h = gb["close"].shift(-h)
    R = (close_h / open1 - 1.0).astype(np.float32)
    btc_mask = (df["code"].astype(str) == BTC)
    btc_map = dict(zip(df["date"][btc_mask].values, R[btc_mask].values))
    Rb = df["date"].map(btc_map).astype(np.float32)
    R_ex = R - beta * Rb
    ok = R.notna() & Rb.notna() & beta.notna()
    n_day = R.groupby(df["date"]).transform("count")
    return R, R_ex, ok, n_day


def topk_resid(pred, meta, h):
    d = pd.DataFrame({"date": meta["date"].values, "p": pred,
                      "Rex": np.clip(meta["R_ex"].values, -1.0, 1.0)})
    rows = {}
    for k in (10, 50):
        tk = d.groupby("date").apply(
            lambda g: g.nlargest(k, "p")["Rex"].mean() if len(g) >= k else np.nan,
            include_groups=False)
        rows[k] = {
            "topk_R": float(tk.mean()),
            "topk_ann_net": (float(tk.mean()) - COST_RT) / h * DAYS,
            "cover": float(tk.notna().mean()),
        }
    return rows


def run(args):
    feats = json.load(open(FEAT_FILE))["features"]
    df = load_bars("crypto_daily")
    df["code"] = df["code"].astype(str)
    beta = build_beta(df)
    log("计算 tech40 特征 ...")
    F = _compute_feats(df.assign(code=df["code"].astype("category")))
    X = pd.DataFrame({k: F[k] for k in feats}, dtype=np.float32)
    hdf = load_hourly()
    idf = build_intraday(hdf)
    idf = idf[["code", "date"] + ID_ALL].copy()
    keys = pd.DataFrame({"code": df["code"].values, "date": df["date"].values})
    idm = keys.merge(idf, on=["code", "date"], how="left")
    assert len(idm) == len(df)
    Xr = pd.concat([X, idm[ID_RV].reset_index(drop=True)], axis=1)
    Xall = pd.concat([X, idm[ID_ALL].reset_index(drop=True)], axis=1)
    del F, hdf, idf, idm
    gc.collect()

    results = []
    threads = args.threads or os.cpu_count()
    for h in args.horizons:
        R, R_ex, ok, n_day = build_labels(df, h, beta)
        base_ok = np.asarray(ok & (n_day >= 10) & (R.abs() <= 5.0))
        Xok = X[base_ok].reset_index(drop=True)
        Xrok = Xr[base_ok].reset_index(drop=True)
        Xallok = Xall[base_ok].reset_index(drop=True)
        meta = pd.DataFrame({"date": df["date"][base_ok].values,
                             "R": R[base_ok].values,
                             "R_ex": R_ex[base_ok].values})
        med = R_ex[base_ok].groupby(df["date"][base_ok]).transform("median")
        y_ex = (R_ex[base_ok] > med).astype(np.int8).reset_index(drop=True)
        is_tr = (meta["date"] <= TRAIN_END).values
        is_va = (meta["date"] >= VALID_START).values
        mva = meta[is_va].reset_index(drop=True)
        yva = y_ex[is_va].values

        configs = [
            ("E1", Xok),
            ("E1Rv", Xrok),
            ("E1Id", Xallok),
        ]
        for name, Xfull in configs:
            if name not in args.variants:
                continue
            Xtr, ytr = Xfull[is_tr], y_ex[is_tr]
            Xva = Xfull[is_va]
            log(f"训练 crypto intraday h={h} variant={name} "
                f"(train={len(ytr)}, valid={len(yva)}, feats={Xtr.shape[1]}) ...")
            t0 = time.time()
            model = lgb.LGBMClassifier(**PARAMS, n_jobs=threads, **{
                k: v for k, v in LGB_KW.items() if k != "learning_rate"})
            model.fit(Xtr, ytr, eval_set=[(Xva, yva)])
            best = model.best_iteration_ or model.n_estimators
            pred = model.predict_proba(Xva)[:, 1]

            aucs = roc_auc_score(yva, pred)
            ic_R, icir_R, _ = rank_ic(pred, mva, clip=1.0)
            mva_ex = pd.DataFrame({"date": mva["date"], "R": mva["R_ex"]})
            ic_Rex, icir_Rex, _ = rank_ic(pred, mva_ex, clip=1.0)
            year_ics = {}
            for yr in ("2024", "2025", "2026"):
                sel = (mva["date"].str[:4] == yr)
                if sel.sum() > 100:
                    ric, _, _ = rank_ic(
                        pred[sel.values],
                        pd.DataFrame({"date": mva["date"][sel],
                                    "R": mva["R_ex"][sel]}), clip=1.0)
                    year_ics[yr] = round(ric, 4)
            tg = topk_resid(pred, mva, h)
            row = {"variant": name, "h": h, "best_iter": best,
                   "auc_vs_own": round(aucs, 4),
                   "ic_R": round(ic_R, 4), "ic_Rex": round(ic_Rex, 4),
                   "icir_Rex": round(icir_Rex, 3),
                   "ic_Rex_24": year_ics.get("2024"),
                   "ic_Rex_25": year_ics.get("2025"),
                   "ic_Rex_26": year_ics.get("2026"),
                   "top10_Rex_mean": round(tg[10]["topk_R"], 5),
                   "top10_ann_net": round(tg[10]["topk_ann_net"], 4),
                   "train_rows": int(len(ytr)), "valid_rows": int(len(yva)),
                   "secs": round(time.time() - t0, 1)}
            results.append(row)
            log(f"  {name} h={h}: AUC={aucs:.4f} IC_Rex={ic_Rex:.4f} "
                f"ICIR={icir_Rex:.2f} year={year_ics} "
                f"Top10_ann={tg[10]['topk_ann_net']:+.1%}")
            model.booster_.save_model(os.path.join(
                OUT, f"crypto_id_{name}_h{h}.txt"))
            del model, pred
            gc.collect()

    res = pd.DataFrame(results)
    path = os.path.join(OUT, "crypto_intraday.csv")
    res.to_csv(path, index=False)
    print(res.to_string(index=False))
    log(f"结果已写入 {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="E1,E1Rv,E1Id")
    ap.add_argument("--horizons", default="5,10")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    args.horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    args.variants = {v.strip() for v in args.variants.split(",") if v.strip()}
    run(args)


if __name__ == "__main__":
    main()
