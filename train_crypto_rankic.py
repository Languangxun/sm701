#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A1: LambdaRankIC 直接优化 (加密, E1 口径)

对照:
  E1     : binary on cross_binary(R_ex)  —— 生产基线
  A1_32  : LambdaRankIC 目标, max_pairs_per_day=32
  A1_64  : LambdaRankIC 目标, max_pairs_per_day=64

目标函数复用 rank_ic_objective.py (逐日 pairwise, 按收益 rank 差加权,
优化 Spearman RankIC)。早停用验证集子采样(每 4 个交易日)的 RankIC feval,
最终指标在完整验证集上用 rank_ic() 计算, 两者的差异只影响停止轮数。

用法:
    python3 train_crypto_rankic.py --variants E1,A1_32,A1_64 --horizons 5,10
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
from rank_ic_objective import (make_lambdarank_ic_objective,  # noqa: E402
                               make_rank_ic_eval)
from train_markets import load_bars, rank_ic  # noqa: E402

OUT = os.path.join(HERE, "markets_output")
FEAT_FILE = os.path.join(HERE, "production", "features_tech.json")

COST_RT = 0.002
DAYS = 365
BTC = "BTCUSDT"
BETA_WIN, BETA_MIN = 90, 60
TRAIN_END, VALID_START = "2023-12-31", "2024-01-01"

PARAMS = {
    "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 1000,
    "feature_fraction": 0.6, "bagging_fraction": 0.7, "bagging_freq": 1,
    "lambda_l2": 20, "max_bin": 127, "verbosity": -1, "num_threads": 8,
}
N_ROUNDS = 1500
EARLY = 80
FEVAL_SUBSAMPLE = 4      # 每 4 个交易日取 1 天做早停 feval


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


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
    return pd.Series(beta_long.reindex(idx).values, index=df.index,
                     dtype=np.float32)


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
        rows[k] = {"topk_R": float(tk.mean()),
                   "topk_ann_net": (float(tk.mean()) - COST_RT) / h * DAYS,
                   "cover": float(tk.notna().mean())}
    return rows


def date_subsample_mask(dates, every):
    ud = pd.unique(dates)
    pick = set(ud[::every])
    return np.array([d in pick for d in dates])


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

    results = []
    threads = args.threads or os.cpu_count()
    for h in args.horizons:
        R, R_ex, ok, n_day = build_labels(df, h, beta)
        base_ok = np.asarray(ok & (n_day >= 10) & (R.abs() <= 5.0))
        Xok = X[base_ok].reset_index(drop=True)
        meta = pd.DataFrame({"date": df["date"][base_ok].values,
                             "R": R[base_ok].values,
                             "R_ex": R_ex[base_ok].values})
        med = R_ex[base_ok].groupby(df["date"][base_ok]).transform("median")
        y = (R_ex[base_ok] > med).astype(np.int8).reset_index(drop=True)
        is_tr = (meta["date"] <= TRAIN_END).values
        is_va = (meta["date"] >= VALID_START).values
        mva = meta[is_va].reset_index(drop=True)
        yva = y[is_va].values
        Rtr = meta["R_ex"].values[is_tr]
        Rva_full = meta["R_ex"].values[is_va]
        dates_tr = meta["date"].values[is_tr]
        dates_va = meta["date"].values[is_va]

        # 早停 feval 子采样
        sub = date_subsample_mask(dates_va, FEVAL_SUBSAMPLE)
        Rva_sub = Rva_full[sub]
        dates_va_sub = dates_va[sub]
        yva_sub = yva[sub]

        Xtr, Xva = Xok[is_tr], Xok[is_va]
        Xva_sub = Xva[sub]

        # --- E1 baseline ---
        if "E1" in args.variants:
            log(f"训练 h={h} E1 (binary) ...")
            t0 = time.time()
            dtr = lgb.Dataset(Xtr, label=y[is_tr],
                              params={"objective": "binary"})
            dva = lgb.Dataset(Xva, label=yva, reference=dtr)
            params = dict(PARAMS, objective="binary",
                          metric=["auc"], num_threads=threads)
            model = lgb.train(params, dtr, num_boost_round=args.rounds,
                              valid_sets=[dva],
                              callbacks=[lgb.early_stopping(EARLY, verbose=False)])
            pred = model.predict(Xva, num_iteration=model.best_iteration)
            best = model.best_iteration
            auc = roc_auc_score(yva, pred)
            ic_Rex, icir, _ = rank_ic(pred, mva.assign(R=mva["R_ex"]),
                                      clip=1.0)
            tg = topk_resid(pred, mva, h)
            results.append({"variant": "E1", "h": h, "best_iter": best,
                            "auc": round(auc, 4), "ic_Rex": round(ic_Rex, 4),
                            "icir_Rex": round(icir, 3),
                            "top10_ann_net": round(tg[10]["topk_ann_net"], 4),
                            "secs": round(time.time() - t0, 1)})
            log(f"  E1 h={h}: AUC={auc:.4f} IC_Rex={ic_Rex:.4f} "
                f"Top10={tg[10]['topk_ann_net']:+.1%}")
            model.save_model(os.path.join(OUT, f"crypto_rk_E1_h{h}.txt"))

        # --- A1 LambdaRankIC ---
        for name, mpairs in (("A1_32", 32), ("A1_64", 64)):
            if name not in args.variants:
                continue
            log(f"训练 h={h} {name} (LambdaRankIC, pairs={mpairs}) ...")
            t0 = time.time()
            dtr = lgb.Dataset(Xtr, label=Rtr)
            dva_sub = lgb.Dataset(Xva_sub, label=Rva_sub, reference=dtr)
            fobj = make_lambdarank_ic_objective(
                Rtr, dates_tr, max_pairs_per_day=mpairs)
            feval = make_rank_ic_eval(Rva_sub, dates_va_sub)
            params = dict(PARAMS, objective=fobj, metric="None",
                          num_threads=threads)
            model = lgb.train(params, dtr, num_boost_round=args.rounds,
                              valid_sets=[dva_sub], feval=feval,
                              callbacks=[lgb.early_stopping(EARLY, verbose=False)])
            pred = model.predict(Xva, num_iteration=model.best_iteration)
            best = model.best_iteration
            ic_Rex, icir, _ = rank_ic(pred, mva.assign(R=mva["R_ex"]),
                                      clip=1.0)
            # AUC 仅作 sanity (标签一致才有意义)
            auc = roc_auc_score(yva, pred)
            tg = topk_resid(pred, mva, h)
            results.append({"variant": name, "h": h, "best_iter": best,
                            "auc": round(auc, 4), "ic_Rex": round(ic_Rex, 4),
                            "icir_Rex": round(icir, 3),
                            "top10_ann_net": round(tg[10]["topk_ann_net"], 4),
                            "secs": round(time.time() - t0, 1)})
            log(f"  {name} h={h}: AUC={auc:.4f} IC_Rex={ic_Rex:.4f} "
                f"ICIR={icir:.2f} Top10={tg[10]['topk_ann_net']:+.1%} "
                f"({time.time()-t0:.0f}s)")
            model.save_model(os.path.join(OUT, f"crypto_rk_{name}_h{h}.txt"))
            del dtr, model
            gc.collect()

    res = pd.DataFrame(results)
    path = os.path.join(OUT, "crypto_rankic.csv")
    res.to_csv(path, index=False)
    print(res.to_string(index=False))
    log(f"结果已写入 {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="E1,A1_32,A1_64")
    ap.add_argument("--horizons", default="5,10")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=N_ROUNDS)
    args = ap.parse_args()
    args.horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    args.variants = {v.strip() for v in args.variants.split(",") if v.strip()}
    run(args)


if __name__ == "__main__":
    main()
