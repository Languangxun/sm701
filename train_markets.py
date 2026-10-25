#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
港股/美股/期货 三市场 LightGBM 横截面模型 (tech 40 特征 + cross_binary 标签)

口径与 A 股生产定稿 (README §18) 一致:
  特征   tech 40 列 (production/features_tech.json), 全部 t 日及以前
  标签   cross_binary: R_h = close[t+h]/open[t+1] - 1, 截面 > 当日中位数
  切分   训练 2018-01-01 ~ 2023-12-31 / 验证 2024-01-01 起 (按年硬切)
  模型   LGBM binary, lr=0.03 / leaves=31 / min_data=1000 / ff=0.6 / bag=0.7
  执行   T+1 开盘买入持有 h 天, 成本单程万 15

与 A 股的差异 (数据源为新浪不复权日线):
  - 美股/港股分红拆股会产生收益跳变: |R_h| > 0.5 (期货 0.25) 的样本剔除,
    RankIC 评估用 R 截断 (股票 ±0.30 / 期货 ±0.10);
  - cross_binary 取截面中位数, 对离群值本身稳健;
  - 期货为主力连续合约, 移仓换月存在跳空, 同上处理;
  - 无涨跌停/停牌过滤 (港美无涨跌停, 期货 T+0)。

用法:
    python3 train_markets.py --markets hk,us,fut --horizons 1,5,10
    python3 train_markets.py --markets hk --limit 300     # 冒烟
    nohup python3 train_markets.py > markets.log 2>&1 &
"""
import argparse
import gc
import json
import os
import sqlite3
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from lgbm_train_backtest import _compute_feats, log

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "stock_cache.db")
OUT = os.path.join(HERE, "markets_output")
FEAT_FILE = os.path.join(HERE, "production", "features_tech.json")

LOAD_START = "2016-01-01"          # 留 2 年 warmup 给 60 日窗口指标
TRAIN_START, TRAIN_END = "2018-01-01", "2023-12-31"
VALID_START = "2024-01-01"
MIN_BARS_PER_CODE = 250            # 剔除历史过短/长期停牌代码
MIN_NAMES_PER_DAY = 10             # 截面太小的日期剔除
COST_RT = 0.003                    # 万15单程 ×2 (进+出)
N_TRADE_DAYS = 243

MARKETS = {
    "hk":  {"table": "hk_daily",  "drop_abs": 0.50, "clip_ic": 0.30},
    "us":  {"table": "us_daily",  "drop_abs": 0.50, "clip_ic": 0.30},
    "fut": {"table": "fut_daily", "drop_abs": 0.25, "clip_ic": 0.10},
}

PARAMS = {
    "objective": "binary", "metric": ["auc", "binary_logloss"],
    "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 1000,
    "feature_fraction": 0.6, "bagging_fraction": 0.7, "bagging_freq": 1,
    "lambda_l2": 20, "max_bin": 127, "verbosity": -1,
}

LGB_KW = dict(n_estimators=3000, learning_rate=PARAMS["learning_rate"],
              callbacks=[lgb.early_stopping(150, verbose=False),
                         lgb.log_evaluation(0)])


def load_bars(table, limit=None, seed=42):
    con = sqlite3.connect(DB)
    q = (f"SELECT code, date, open, high, low, close, vol "
         f"FROM {table} WHERE date >= ?")
    params = [LOAD_START]
    if limit:
        codes = [r[0] for r in con.execute(
            f"SELECT DISTINCT code FROM {table} ORDER BY code")]
        rs = np.random.RandomState(seed)
        codes = sorted(rs.choice(codes, min(limit, len(codes)),
                                 replace=False).tolist())
        q += " AND code IN (%s)" % ",".join(["?"] * len(codes))
        params += codes
    q += " ORDER BY code, date"
    df = pd.read_sql_query(q, con, params=params)
    con.close()
    # 剔除过长历史过短的代码
    cnt = df.groupby("code")["date"].transform("count")
    df = df[cnt >= MIN_BARS_PER_CODE].copy()
    df["code"] = df["code"].astype("category")
    for c in ["open", "high", "low", "close", "vol"]:
        df[c] = df[c].astype("float64")
    log(f"{table}: {len(df):,} 行, {df['code'].nunique()} 个代码")
    return df


def build_xy(df, feats, h, drop_abs):
    log(f"构建特征/标签 h={h} ...")
    F = _compute_feats(df)
    X = pd.DataFrame(index=df.index, dtype=np.float32)
    for k in feats:
        X[k] = F[k]
    del F
    gc.collect()

    gb = df.groupby("code", sort=False)
    open1 = gb["open"].shift(-1)
    close_h = gb["close"].shift(-h)
    R = (close_h / open1 - 1.0).astype(np.float32)

    med = R.groupby(df["date"]).transform("median")
    n_day = R.groupby(df["date"]).transform("count")
    y = (R > med).astype(np.int8)

    ok = R.notna() & (n_day >= MIN_NAMES_PER_DAY) & (R.abs() <= drop_abs)
    X = X[ok]
    y = y[ok]
    meta = pd.DataFrame({
        "date": df["date"][ok].values,
        "R": R[ok].values,
        "code": df["code"][ok].cat.codes.values,
    })
    log(f"  样本 {len(y):,} (剔除 {len(df) - len(y):,}), "
        f"正例率 {y.mean():.4f}")
    return X.reset_index(drop=True), y.reset_index(drop=True), meta


def rank_ic(pred, meta, clip):
    R = np.clip(meta["R"].values, -clip, clip)
    d = pd.DataFrame({"date": meta["date"].values, "p": pred, "R": R})
    ic = d.groupby("date").apply(
        lambda g: g["p"].corr(g["R"], method="spearman"),
        include_groups=False)
    ic = ic.dropna()
    return float(ic.mean()), float(ic.mean() / (ic.std() + 1e-9)), ic


def topk_stats(pred, meta, h, clip, k_list=(10, 50)):
    d = pd.DataFrame({"date": meta["date"].values, "p": pred,
                      "R": np.clip(meta["R"].values, -clip, clip)})
    rows = []
    for k in k_list:
        def topk_mean(g):
            return g.nlargest(k, "p")["R"].mean() if len(g) >= k else np.nan
        tk = d.groupby("date").apply(topk_mean, include_groups=False)
        base = d.groupby("date")["R"].mean()
        rows.append({
            "topk": k,
            "topk_R": float(tk.mean()), "base_R": float(base.mean()),
            "topk_ann_net": (float(tk.mean()) - COST_RT) / h * N_TRADE_DAYS,
            "base_ann": float(base.mean()) / h * N_TRADE_DAYS,
            "cover": float(tk.notna().mean()),
        })
    return rows


def train_market(market, horizons, limit, seed, threads):
    cfg = MARKETS[market]
    feats = json.load(open(FEAT_FILE))["features"]
    df = load_bars(cfg["table"], limit, seed)
    results = []
    for h in horizons:
        X, y, meta = build_xy(df, feats, h, cfg["drop_abs"])
        is_train = (meta["date"] <= TRAIN_END).values
        is_valid = (meta["date"] >= VALID_START).values
        Xtr, ytr = X[is_train], y[is_train].values
        Xva, yva = X[is_valid], y[is_valid].values
        meta_va = meta[is_valid].reset_index(drop=True)
        log(f"{market} h={h}: 训练 {len(ytr):,} / 验证 {len(yva):,}")

        model = lgb.LGBMClassifier(**PARAMS, n_jobs=threads, **{
            k: v for k, v in LGB_KW.items() if k != "learning_rate"})
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)])
        best = model.best_iteration_ or model.n_estimators
        log(f"{market} h={h}: best_iter={best}")

        pred = model.predict_proba(Xva)[:, 1]
        from sklearn.metrics import roc_auc_score, log_loss
        auc = roc_auc_score(yva, pred)
        ll = log_loss(yva, pred, labels=[0, 1])
        icm, icir, _ = rank_ic(pred, meta_va, cfg["clip_ic"])

        d = pd.DataFrame({"date": meta_va["date"].values, "y": yva,
                          "p": pred,
                          "R": np.clip(meta_va["R"].values,
                                       -cfg["clip_ic"], cfg["clip_ic"])})
        d["year"] = d["date"].str[:4]
        yearly = d.groupby("year").apply(
            lambda g: pd.Series({
                "auc": roc_auc_score(g["y"], g["p"]),
                "ic": g.groupby("date").apply(
                    lambda t: t["p"].corr(t["R"], method="spearman"),
                    include_groups=False).mean()}),
            include_groups=False)

        tops = topk_stats(pred, meta_va, h, cfg["clip_ic"])

        os.makedirs(OUT, exist_ok=True)
        model.booster_.save_model(os.path.join(OUT, f"{market}_h{h}.txt"))
        row = {"market": market, "h": h, "best_iter": best,
               "valid_auc": round(auc, 4), "valid_logloss": round(ll, 4),
               "valid_ic": round(icm, 4), "valid_icir": round(icir, 3),
               "train_rows": len(ytr), "valid_rows": len(yva)}
        for t in tops:
            row[f"top{t['topk']}_ann_net"] = round(t["topk_ann_net"], 4)
            row[f"top{t['topk']}_R"] = round(t["topk_R"], 5)
        for yr, r in yearly.iterrows():
            row[f"auc_{yr}"] = round(r["auc"], 4)
            row[f"ic_{yr}"] = round(r["ic"], 4)
        results.append(row)
        log(f"{market} h={h}: AUC={auc:.4f} IC={icm:.4f} ICIR={icir:.2f} "
            + " ".join(f"Top{t['topk']}净年化={t['topk_ann_net']:+.1%}"
                       for t in tops))
        del X, y, meta, Xtr, Xva, d
        gc.collect()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default="hk,us,fut")
    ap.add_argument("--horizons", default="1,5,10")
    ap.add_argument("--limit", type=int, default=0, help="每市场抽样代码数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    markets = [m.strip() for m in args.markets.split(",") if m.strip()]
    horizons = [int(h) for h in args.horizons.split(",")]
    threads = args.threads or os.cpu_count()

    all_rows = []
    for m in markets:
        if m not in MARKETS:
            raise SystemExit(f"未知市场 {m}, 可选: {', '.join(MARKETS)}")
        t0 = time.time()
        all_rows += train_market(m, horizons, args.limit, args.seed, threads)
        log(f"{m} 完成, 用时 {(time.time() - t0) / 60:.1f} min")

    os.makedirs(OUT, exist_ok=True)
    res = pd.DataFrame(all_rows)
    res.to_csv(os.path.join(OUT, "markets_results.csv"), index=False)
    print(res.to_string(index=False))
    log(f"结果已写入 {OUT}/markets_results.csv, 模型 {OUT}/<market>_h<h>.txt")


if __name__ == "__main__":
    main()
