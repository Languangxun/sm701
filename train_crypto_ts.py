#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A8: 逐币时序标签 (替代/补充 cross_binary)

标签 (per-coin, 不再取截面中位数):
  A8_pos  : y = 1{R_h > 0}                        (方向)
  A8_vol  : y = 1{R_h > λ·σ_i·sqrt(h)}            (波动标准化门槛, λ=0.5)
  A8_ex   : y = 1{R_ex > λ·σ_i·sqrt(h)}           (beta 中性 + 波动门槛)

σ_i 为每币 60 日滚动日收益标准差 (t 日及以前)。
全部用修复后的早停 (2024 ES, patience 80), 2026 为 OOS。

评价: 三组标签的预测都按 R_ex 的 RankIC 评价 (与生产口径一致),
以便判断「时序信号」是否也能用于横截面选币 / 是否值得 ensemble。

用法:
    python3 train_crypto_ts.py --variants E1_auc,A8_pos,A8_vol,A8_ex --horizons 5,10
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
from train_crypto_v2 import (build_beta, build_labels,  # noqa: E402
                             fit_binary, topk_resid, year_ic)
from train_markets import load_bars, rank_ic  # noqa: E402

OUT = os.path.join(HERE, "markets_output")
FEAT_FILE = os.path.join(HERE, "production", "features_tech.json")
TRAIN_END, VALID_START, ES_END, TEST_START = (
    "2023-12-31", "2024-01-01", "2025-12-31", "2026-01-01")
LAMBDA_VOL = 0.5

PARAMS = {
    "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 1000,
    "feature_fraction": 0.6, "bagging_fraction": 0.7, "bagging_freq": 1,
    "lambda_l2": 20, "max_bin": 127, "verbosity": -1,
}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def run(args):
    feats = json.load(open(FEAT_FILE))["features"]
    df = load_bars("crypto_daily")
    df["code"] = df["code"].astype(str)
    beta = build_beta(df)
    log("计算 tech40 特征 ...")
    F = _compute_feats(df.assign(code=df["code"].astype("category")))
    X = pd.DataFrame({k: F[k] for k in feats}, dtype=np.float32)
    del F
    gc.collect()
    threads = args.threads or os.cpu_count()

    # 每币 60 日波动 (t 日及以前)
    ret1 = df.groupby("code", sort=False)["close"].pct_change()
    vol60 = ret1.groupby(df["code"], sort=False).transform(
        lambda s: s.rolling(60, min_periods=40).std())

    results = []
    for h in args.horizons:
        R, R_ex, ok, n_day = build_labels(df, h, beta)
        base_ok = np.asarray(ok & (n_day >= 10) & (R.abs() <= 5.0))
        Xok = X[base_ok].reset_index(drop=True)
        meta = pd.DataFrame({"date": df["date"][base_ok].values,
                             "R": R[base_ok].values,
                             "R_ex": R_ex[base_ok].values})
        med = R_ex[base_ok].groupby(df["date"][base_ok]).transform("median")
        z = (vol60[base_ok] * np.sqrt(h)).values
        R_h = R[base_ok].values
        R_ex_h = R_ex[base_ok].values
        labels = {
            "E1_auc": (R_ex_h > med.values).astype(np.int8),
            "A8_pos": (R_h > 0).astype(np.int8),
            "A8_vol": (R_h > LAMBDA_VOL * z).astype(np.int8),
            "A8_ex": (R_ex_h > LAMBDA_VOL * z).astype(np.int8),
        }
        is_tr = (meta["date"] <= TRAIN_END).values
        is_es = ((meta["date"] >= VALID_START) &
                 (meta["date"] <= ES_END)).values
        is_va = (meta["date"] >= VALID_START).values
        mva = meta[is_va].reset_index(drop=True)
        te_mask = mva["date"].str[:4].isin(["2026"]).values
        mte = mva[te_mask].reset_index(drop=True)
        Xtr, Xes, Xva = Xok[is_tr], Xok[is_es], Xok[is_va]

        for name, y in labels.items():
            if name not in args.variants:
                continue
            y = pd.Series(y)
            log(f"h={h} {name}: pos_rate(train)="
                f"{y[is_tr].mean():.3f} 训练 ...")
            t0 = time.time()
            model, pred = fit_binary(
                Xtr, y[is_tr].values, Xes, y[is_es].values,
                Xva, y[is_va].values, "auc", args.rounds, args.patience,
                PARAMS, threads)
            best = model.best_iteration
            ic_all, icir_all, _ = rank_ic(
                pred, pd.DataFrame({"date": mva["date"], "R": mva["R_ex"]}),
                clip=1.0)
            yics = year_ic(pred, mva)
            ic_te, _, _ = rank_ic(
                pred[te_mask],
                pd.DataFrame({"date": mte["date"], "R": mte["R_ex"]}),
                clip=1.0)
            tg_all = topk_resid(pred, mva, h)
            tg_te = topk_resid(pred, mva, h, mask=te_mask)
            row = {"variant": name, "h": h, "best_iter": best,
                   "pos_rate": round(float(y[is_tr].mean()), 3),
                   "ic_full": round(ic_all, 4),
                   "icir_full": round(icir_all, 3),
                   "ic24": yics["2024"], "ic25": yics["2025"],
                   "ic26": yics["2026"],
                   "ic26_oos": round(ic_te, 4),
                   "top10_full": round(tg_all["top10_ann_net"], 4),
                   "top10_26": round(tg_te["top10_ann_net"], 4),
                   "secs": round(time.time() - t0, 1)}
            results.append(row)
            log(f"  {name} h={h}: best={best} IC_full={ic_all:.4f} "
                f"IC26(OOS)={ic_te:.4f} IC24/25/26="
                f"{yics['2024']}/{yics['2025']}/{yics['2026']} "
                f"Top10_full={tg_all['top10_ann_net']:+.1%} "
                f"Top10_26={tg_te['top10_ann_net']:+.1%} "
                f"({row['secs']}s)")
            model.save_model(os.path.join(OUT, f"crypto_ts_{name}_h{h}.txt"))
            del model
            gc.collect()

    res = pd.DataFrame(results)
    path = os.path.join(OUT, "crypto_ts.csv")
    res.to_csv(path, index=False)
    print(res.to_string(index=False))
    log(f"结果已写入 {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="E1_auc,A8_pos,A8_vol,A8_ex")
    ap.add_argument("--horizons", default="5,10")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--patience", type=int, default=80)
    args = ap.parse_args()
    args.horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    args.variants = {v.strip() for v in args.variants.split(",") if v.strip()}
    run(args)


if __name__ == "__main__":
    main()
