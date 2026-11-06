#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
加密市场训练方法消融 (README §19.3 的后续改进实验)

对比对象 (全部以 beta 中性 residual 收益为主要评价口径, 避免牛熊 beta 假 alpha):
  base  : 现状 —— tech40 + cross_binary(相对截面中位数) + LGBM binary
  E3    : base  + 样本权重 (截面数 / 时间衰减半衰期2年)
  E1    : beta 中性标签 R_ex = R_h - beta_i * R_btc_h,
          beta 为 90 日滚动回归系数, y = (R_ex > 当日中位数)
  E13   : E1 + E3
  E134  : E1 + E3 + E4 (加密专属特征: 相对 BTC 动量 / 波动归一动量 /
          beta / 成交额份额 / 市场 regime, 在 tech40 上追加)

评价口径 (验证集 2024-01 ~ 2026-11):
  - RankIC / ICIR (对 R_ex 为主, 同时对 raw R 供对照)
  - 分年 IC (R_ex)
  - TopK residual (R_ex) 年化 (扣除双边 0.2%)
  - AUC (对各自训练标签, 只做 sanity check)

说明:
  - 特征计算基准 (_compute_feats) 与 train_markets.py 一致;
  - h 只跑 5/10: h1 已证明被成本吃掉 (README §19.3), 不浪费算力;
  - 模型存 markets_output/crypto_ab_<variant>_h{h}.txt, 结果汇总 crypto_ab.csv。

用法:
    python3 train_crypto_ab.py --variants base,E3,E1,E13,E134
    nohup python3 train_crypto_ab.py > crypto_ab.log 2>&1 &
"""
import argparse
import gc
import json
import os
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

OUT = os.path.join(HERE, "markets_output")
FEAT_FILE = os.path.join(HERE, "production", "features_tech.json")

COST_RT = 0.002          # 加密双边 0.2%
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


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _pivot_idx(df):
    """(date, code) MultiIndex, 与 pivot_table 后的 index 对齐"""
    return pd.MultiIndex.from_arrays(
        [df["date"].values, df["code"].astype(str).values])


def build_beta(df):
    """每币 90 日滚动 beta (对 BTC), 返回与 df 同序的 Series"""
    log("计算 beta (90日滚动) ...")
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
    out = beta_long.reindex(_pivot_idx(df)).values
    return pd.Series(out, index=df.index, dtype=np.float32)


def build_e4(df, F, beta):
    """加密专属特征 (在 tech40 基础上追加)"""
    log("计算 E4 加密专属特征 ...")
    dates = df["date"]
    out = {}

    # --- 相对 BTC 的动量 (coin - BTC, 同列名用 F 里已有 ret) ---
    ret_df = pd.DataFrame({k: F[k] for k in ("ret1", "ret5", "ret20", "ret60", "vol20")})
    ret_df["code"] = df["code"].astype(str).values
    ret_df["date"] = dates.values
    for win in (1, 5, 20, 60):
        piv = ret_df.pivot_table(index="date", columns="code", values=f"ret{win}")
        s = piv.sub(piv[BTC], axis=0).stack()
        out[f"btc_diff_ret{win}"] = s.reindex(_pivot_idx(df)).values.astype(np.float32)
    piv20 = ret_df.pivot_table(index="date", columns="code", values="vol20")
    out["btc_vol20"] = piv20[BTC].reindex(_pivot_idx(df).get_level_values(0)).values.astype(np.float32)

    # --- 波动归一动量 / beta / 成交额份额 / 市场 regime ---
    out["vmom20"] = (F["ret20"] / (F["vol20"] + 1e-9)).astype(np.float32)
    out["vmom60"] = (F["ret60"] / (F["vol60"] + 1e-9)).astype(np.float32)
    out["beta"] = beta.astype(np.float32)
    vol = df["vol"].astype(np.float64)
    share = vol / vol.groupby(dates).transform("sum")
    out["vol_share"] = np.log((share + 1e-9).astype(np.float64)).astype(np.float32)
    out["r_vol_share"] = share.groupby(dates).rank(pct=True).astype(np.float32)
    mkt20 = (df["close"] / df.groupby("code", sort=False)["close"]
             .transform(lambda s: s.rolling(20, min_periods=20).mean()) - 1.0)
    out["mkt_breadth20"] = (mkt20 > 0).astype(np.float32).groupby(dates).transform("mean")
    return out


def build_labels(df, h, beta):
    """返回 R(原始), R_ex(beta 中性), ok, n_day"""
    log(f"构建标签 h={h} ...")
    gb = df.groupby("code", sort=False)
    open1 = gb["open"].shift(-1)
    close_h = gb["close"].shift(-h)
    R = (close_h / open1 - 1.0).astype(np.float32)
    btc_mask = (df["code"].astype(str) == BTC)
    btc_R = R[btc_mask]
    btc_map = dict(zip(df["date"][btc_mask].values, btc_R.values))
    Rb = df["date"].map(btc_map).astype(np.float32)
    R_ex = R - beta * Rb
    ok = R.notna() & Rb.notna()
    n_day = R.groupby(df["date"]).transform("count")
    return R, R_ex, ok, n_day


def topk_resid(pred, meta, h):
    """TopK residual (R_ex) 年化, 扣双边成本"""
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
    df = load_bars("crypto_daily")      # code 为独立 category, 供 _compute_feats
    beta = build_beta(df)
    log("计算 tech40 特征 ...")
    F = _compute_feats(df)
    X = pd.DataFrame({k: F[k] for k in feats}, dtype=np.float32)
    e4 = build_e4(df, F, beta)
    Xe4 = pd.concat([X, pd.DataFrame(e4, index=df.index)], axis=1)
    del F, e4
    gc.collect()

    n_day_all = df.groupby("date")["close"].transform("count")
    years = (pd.Timestamp(TRAIN_END) - pd.to_datetime(df["date"])).dt.days / 365.0
    w3 = ((n_day_all.clip(upper=50) / 50.0) * (0.5 ** np.clip(years, 0, None) / 2.0)
          ).astype(np.float32)

    results = []
    threads = args.threads or os.cpu_count()
    for h in args.horizons:
        R, R_ex, ok, n_day = build_labels(df, h, beta)
        base_ok = np.asarray(ok & (n_day >= 10) & (R.abs() <= 5.0) & beta.notna())
        Xok = X[base_ok].reset_index(drop=True)
        Xe4ok = Xe4[base_ok].reset_index(drop=True)
        meta = pd.DataFrame({"date": df["date"][base_ok].values,
                             "R": R[base_ok].values,
                             "R_ex": R_ex[base_ok].values})
        med = R_ex[base_ok].groupby(df["date"][base_ok]).transform("median")
        y_ex = (R_ex[base_ok] > med).astype(np.int8).reset_index(drop=True)
        medR = R[base_ok].groupby(df["date"][base_ok]).transform("median")
        y_base = (R[base_ok] > medR).astype(np.int8).reset_index(drop=True)

        is_tr = (meta["date"] <= TRAIN_END).values
        is_va = (meta["date"] >= VALID_START).values
        mva = meta[is_va].reset_index(drop=True)
        yva_base = y_base[is_va].values
        yva_ex = y_ex[is_va].values

        configs = [
            ("base", Xok, y_base, None, yva_base),
            ("E3", Xok, y_base, w3[base_ok].values, yva_base),
            ("E1", Xok, y_ex, None, yva_ex),
            ("E13", Xok, y_ex, w3[base_ok].values, yva_ex),
            ("E134", Xe4ok, y_ex, w3[base_ok].values, yva_ex),
        ]

        for name, Xfull, yfull, wfull, yva in configs:
            if name not in args.variants:
                continue
            Xtr, ytr = Xfull[is_tr], yfull[is_tr]
            Xva = Xfull[is_va]
            wtr = (wfull[is_tr] if wfull is not None else None)
            log(f"训练 crypto h={h} variant={name} "
                f"(train={len(ytr)}, valid={len(yva)}, feats={Xtr.shape[1]}) ...")
            t0 = time.time()
            model = lgb.LGBMClassifier(**PARAMS, n_jobs=threads, **{
                k: v for k, v in LGB_KW.items() if k != "learning_rate"})
            model.fit(Xtr, ytr, eval_set=[(Xva, yva)], sample_weight=wtr)
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
            log(f"  {name} h={h}: AUC(own)={aucs:.4f} IC_Rex={ic_Rex:.4f} "
                f"ICIR={icir_Rex:.2f} year={year_ics} "
                f"Top10_ann={tg[10]['topk_ann_net']:+.1%}")
            model.booster_.save_model(os.path.join(
                OUT, f"crypto_ab_{name}_h{h}.txt"))
            del model, pred
            gc.collect()

    res = pd.DataFrame(results)
    path = os.path.join(OUT, "crypto_ab.csv")
    res.to_csv(path, index=False)
    print(res.to_string(index=False))
    log(f"结果已写入 {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="base,E3,E1,E13,E134",
                    help="逗号分隔的变种名")
    ap.add_argument("--horizons", default="5,10")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    args.horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    args.variants = {v.strip() for v in args.variants.split(",") if v.strip()}
    run(args)


if __name__ == "__main__":
    main()
