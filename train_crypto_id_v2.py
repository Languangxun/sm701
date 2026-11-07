#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分时特征复测 (早停修复后): tech40 vs +rv/mom vs +全部 intraday

复用 train_crypto_intraday.build_intraday 的日级 intraday 特征,
训练改为 lgb.train + 正确早停 (2024-2025 ES, 2026 OOS), 其余同 E1.

用法:
    python3 train_crypto_id_v2.py --variants E1_auc,E1Rv,E1Id --horizons 5,10
"""
import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lgbm_train_backtest import _compute_feats  # noqa: E402
from train_crypto_intraday import ID_ALL, ID_RV, build_intraday, load_hourly  # noqa: E402
from train_crypto_v2 import (PARAMS, build_beta, build_labels,  # noqa: E402
                             fit_binary, topk_resid, year_ic)
from train_markets import load_bars, rank_ic  # noqa: E402

OUT = os.path.join(HERE, "markets_output")
FEAT_FILE = os.path.join(HERE, "production", "features_tech.json")
TRAIN_END, VALID_START, ES_END, TEST_START = (
    "2023-12-31", "2024-01-01", "2025-12-31", "2026-01-01")


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
    hdf = load_hourly()
    idf = build_intraday(hdf)[["code", "date"] + ID_ALL]
    keys = pd.DataFrame({"code": df["code"].values, "date": df["date"].values})
    idm = keys.merge(idf, on=["code", "date"], how="left")
    assert len(idm) == len(df)
    Xr = pd.concat([X, idm[ID_RV].reset_index(drop=True)], axis=1)
    Xall = pd.concat([X, idm[ID_ALL].reset_index(drop=True)], axis=1)
    del F, hdf, idf, idm
    gc.collect()
    threads = args.threads or os.cpu_count()

    results = []
    for h in args.horizons:
        R, R_ex, ok, n_day = build_labels(df, h, beta)
        base_ok = np.asarray(ok & (n_day >= 10) & (R.abs() <= 5.0))
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
        Xok = {"E1_auc": X[base_ok].reset_index(drop=True),
               "E1Rv": Xr[base_ok].reset_index(drop=True),
               "E1Id": Xall[base_ok].reset_index(drop=True)}

        for name in ("E1_auc", "E1Rv", "E1Id"):
            if name not in args.variants:
                continue
            Xf = Xok[name]
            log(f"h={h} {name} ({Xf.shape[1]} feats) 训练 ...")
            t0 = time.time()
            model, pred = fit_binary(
                Xf[is_tr], y[is_tr], Xf[is_es], y[is_es],
                Xf[is_va], y[is_va], "auc", args.rounds, args.patience,
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
                   "n_feats": Xf.shape[1],
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
                f"Top10_26={tg_te['top10_ann_net']:+.1%} "
                f"({row['secs']}s)")
            model.save_model(os.path.join(OUT, f"crypto_idv2_{name}_h{h}.txt"))
            del model
            gc.collect()

    res = pd.DataFrame(results)
    path = os.path.join(OUT, "crypto_id_v2.csv")
    res.to_csv(path, index=False)
    print(res.to_string(index=False))
    log(f"结果已写入 {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="E1_auc,E1Rv,E1Id")
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
