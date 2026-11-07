#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用指定超参 JSON 在加密 h10 (可选 id 特征) 上重训, 输出 2024+/2026 指标.

用法:
    python3 eval_crypto_params.py --params '{"learning_rate":0.02,...}' --features id
"""
import argparse
import gc
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lgbm_train_backtest import _compute_feats  # noqa: E402
from train_crypto_v2 import (build_beta, build_labels, fit_binary,  # noqa: E402
                             topk_resid, year_ic)
from train_markets import load_bars, rank_ic  # noqa: E402

OUT = os.path.join(HERE, "markets_output")
FEAT_FILE = os.path.join(HERE, "production", "features_tech.json")
TRAIN_END, VALID_START, ES_END, TEST_START = (
    "2023-12-31", "2024-01-01", "2025-12-31", "2026-01-01")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--features", default="id", choices=["tech", "rv", "id"])
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--tag", default="hpo")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    p = json.loads(args.params)
    threads = args.threads or os.cpu_count()
    h = args.horizon

    feats = json.load(open(FEAT_FILE))["features"]
    df = load_bars("crypto_daily")
    df["code"] = df["code"].astype(str)
    beta = build_beta(df)
    F = _compute_feats(df.assign(code=df["code"].astype("category")))
    X = pd.DataFrame({k: F[k] for k in feats}, dtype=np.float32)
    del F
    if args.features in ("rv", "id"):
        from train_crypto_intraday import (ID_ALL, ID_RV, build_intraday,
                                           load_hourly)
        hdf = load_hourly()
        idf = build_intraday(hdf)[["code", "date"] + ID_ALL]
        keys = pd.DataFrame({"code": df["code"].values,
                             "date": df["date"].values})
        idm = keys.merge(idf, on=["code", "date"], how="left")
        cols = ID_RV if args.features == "rv" else ID_ALL
        X = pd.concat([X, idm[cols].reset_index(drop=True)], axis=1)
        del hdf, idf, idm
    gc.collect()

    R, R_ex, ok, n_day = build_labels(df, h, beta)
    base_ok = np.asarray(ok & (n_day >= 10) & (R.abs() <= 5.0))
    Xok = X[base_ok].reset_index(drop=True)
    meta = pd.DataFrame({"date": df["date"][base_ok].values,
                         "R": R[base_ok].values,
                         "R_ex": R_ex[base_ok].values})
    med = R_ex[base_ok].groupby(df["date"][base_ok]).transform("median")
    y = (R_ex[base_ok] > med).astype(np.int8).values
    is_tr = (meta["date"] <= TRAIN_END).values
    is_es = ((meta["date"] >= VALID_START) &
             (meta["date"] <= ES_END)).values
    is_va = (meta["date"] >= VALID_START).values
    mva = meta[is_va].reset_index(drop=True)
    te_mask = mva["date"].str[:4].isin(["2026"]).values
    mte = mva[te_mask].reset_index(drop=True)

    params = {k: v for k, v in p.items()
              if k in ("learning_rate", "num_leaves", "min_data_in_leaf",
                       "feature_fraction", "bagging_fraction", "lambda_l2",
                       "max_depth", "max_bin")}
    params.setdefault("bagging_freq", 1)
    params.setdefault("verbosity", -1)
    model, pred = fit_binary(
        Xok[is_tr], y[is_tr], Xok[is_es], y[is_es],
        Xok[is_va], y[is_va], "auc", 4000, 80, params, threads)
    ic_all, icir_all, _ = rank_ic(
        pred, pd.DataFrame({"date": mva["date"], "R": mva["R_ex"]}), clip=1.0)
    yics = year_ic(pred, mva)
    ic_te, icir_te, _ = rank_ic(
        pred[te_mask],
        pd.DataFrame({"date": mte["date"], "R": mte["R_ex"]}), clip=1.0)
    tg_te = topk_resid(pred, mva, h, mask=te_mask)
    log = (f"{args.tag} feats={args.features} h={h} best={model.best_iteration} "
           f"IC_full={ic_all:.4f} ICIR={icir_all:.2f} "
           f"IC24/25/26={yics['2024']}/{yics['2025']}/{yics['2026']} "
           f"IC26_OOS={ic_te:.4f} Top10_26={tg_te['top10_ann_net']:+.1%}")
    print(log)
    model.save_model(os.path.join(OUT, f"crypto_prod_{args.tag}_h{h}.txt"))
    with open(os.path.join(OUT, f"crypto_prod_{args.tag}_h{h}.json"), "w") as f:
        json.dump({"params": params, "features": args.features,
                   "best_iter": model.best_iteration, "ic_full": ic_all,
                   "ic26": ic_te, "years": yics,
                   "top10_26": tg_te["top10_ann_net"]}, f, indent=2)


if __name__ == "__main__":
    main()
