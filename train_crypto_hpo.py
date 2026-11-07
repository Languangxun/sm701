#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A9: 加密超参随机搜索 (早停修复后)

背景: 旧超参沿用 A 股 (min_data=1000 / leaves=31 / lr=0.03), 且早停失效.
      修复后最优轮数仅 ~40, 明显过正则化, 需要重新搜.

协议 (防泄漏):
  - 早停集: 2024 全年 (patience 80, metric=AUC)
  - 选择集: 2025 全年 RankIC(R_ex) 为主, 同分时看 2024 IC
  - 测试集: 2026 全年 (只在最终报告使用, 不参与选择)
  - 特征/标签: tech40 + E1 beta 中性 R_ex 截面中位数

搜索: 随机采样 N 组 (默认 24), h 默认 10 (生产主周期), 输出:
  markets_output/crypto_hpo.csv (全部 trial 明细)
  markets_output/crypto_hpo_best.json (top-3 配置)

用法:
    python3 train_crypto_hpo.py --trials 24 --horizon 10 --seed 7
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
                             year_ic, topk_resid)
from train_markets import load_bars, rank_ic  # noqa: E402

OUT = os.path.join(HERE, "markets_output")
FEAT_FILE = os.path.join(HERE, "production", "features_tech.json")

TRAIN_END = "2023-12-31"
ES_START, ES_END = "2024-01-01", "2024-12-31"     # 早停
SEL_START, SEL_END = "2025-01-01", "2025-12-31"   # 选择
TEST_START = "2026-01-01"                         # 测试

SPACE = {
    "learning_rate": [0.01, 0.015, 0.02, 0.03, 0.05],
    "num_leaves": [7, 15, 31, 63],
    "min_data_in_leaf": [50, 100, 200, 500, 1000],
    "feature_fraction": [0.3, 0.4, 0.5, 0.6, 0.8],
    "bagging_fraction": [0.6, 0.7, 0.8, 0.9, 1.0],
    "lambda_l2": [1, 5, 20, 50, 100],
    "max_depth": [-1, 4, 5, 6, 8],
}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sample_params(rng):
    p = {k: v[rng.randint(len(v))] for k, v in SPACE.items()}
    p["bagging_freq"] = 1
    p["max_bin"] = 127
    p["feature_pre_filter"] = False
    p["verbosity"] = -1
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=24)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=4000)
    ap.add_argument("--patience", type=int, default=80)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--features", default="tech",
                    choices=["tech", "rv", "id"],
                    help="tech=tech40; rv=+rv/mom 4列; id=+全部 intraday 11列")
    args = ap.parse_args()
    rng = np.random.RandomState(args.seed)
    threads = args.threads or os.cpu_count()

    feats = json.load(open(FEAT_FILE))["features"]
    df = load_bars("crypto_daily")
    df["code"] = df["code"].astype(str)
    beta = build_beta(df)
    log("计算 tech40 特征 ...")
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
        assert len(idm) == len(df)
        cols = ID_RV if args.features == "rv" else ID_ALL
        X = pd.concat([X, idm[cols].reset_index(drop=True)], axis=1)
        del hdf, idf, idm
    gc.collect()
    log(f"特征数: {X.shape[1]} ({args.features})")

    h = args.horizon
    R, R_ex, ok, n_day = build_labels(df, h, beta)
    base_ok = np.asarray(ok & (n_day >= 10) & (R.abs() <= 5.0))
    Xok = X[base_ok].reset_index(drop=True)
    meta = pd.DataFrame({"date": df["date"][base_ok].values,
                         "R": R[base_ok].values,
                         "R_ex": R_ex[base_ok].values})
    med = R_ex[base_ok].groupby(df["date"][base_ok]).transform("median")
    y = (R_ex[base_ok] > med).astype(np.int8).reset_index(drop=True)

    is_tr = (meta["date"] <= TRAIN_END).values
    is_es = ((meta["date"] >= ES_START) & (meta["date"] <= ES_END)).values
    is_sel = ((meta["date"] >= SEL_START) & (meta["date"] <= SEL_END)).values
    is_te = (meta["date"] >= TEST_START).values
    Xtr, Xes, Xsel, Xte = (Xok[is_tr], Xok[is_es], Xok[is_sel], Xok[is_te])
    ytr, yes, ysel, yte = (y[is_tr], y[is_es], y[is_sel], y[is_te])
    msel = meta[is_sel].reset_index(drop=True)
    mte = meta[is_te].reset_index(drop=True)
    dtr = lgb.Dataset(Xtr, label=ytr)
    des = lgb.Dataset(Xes, label=yes, reference=dtr)

    rows = []
    for t in range(args.trials):
        p = sample_params(rng)
        t0 = time.time()
        params = dict(p, objective="binary", metric=["auc"],
                      num_threads=threads)
        try:
            model = lgb.train(params, dtr, num_boost_round=args.rounds,
                              valid_sets=[des],
                              callbacks=[lgb.early_stopping(args.patience,
                                                            verbose=False),
                                         lgb.log_evaluation(0)])
        except Exception as e:
            log(f"trial {t}: ERROR {e}")
            continue
        best = model.best_iteration
        p_sel = model.predict(Xsel, num_iteration=best)
        p_te = model.predict(Xte, num_iteration=best)
        ic_sel, icir_sel, _ = rank_ic(
            p_sel, pd.DataFrame({"date": msel["date"], "R": msel["R_ex"]}),
            clip=1.0)
        ic_te, _, _ = rank_ic(
            p_te, pd.DataFrame({"date": mte["date"], "R": mte["R_ex"]}),
            clip=1.0)
        auc_sel = roc_auc_score(ysel, p_sel)
        row = {"trial": t, "best_iter": best,
               "ic_sel": round(ic_sel, 4), "icir_sel": round(icir_sel, 3),
               "auc_sel": round(auc_sel, 4), "ic_test26": round(ic_te, 4),
               "secs": round(time.time() - t0, 1), **p}
        rows.append(row)
        log(f"trial {t}: IC25={ic_sel:.4f} ICIR={icir_sel:.2f} "
            f"AUC25={auc_sel:.4f} IC26={ic_te:.4f} best={best} "
            f"lr={p['learning_rate']} leaves={p['num_leaves']} "
            f"min_data={p['min_data_in_leaf']} ff={p['feature_fraction']} "
            f"bag={p['bagging_fraction']} l2={p['lambda_l2']} "
            f"depth={p['max_depth']} ({row['secs']}s)")
        del model
        gc.collect()

    res = pd.DataFrame(rows).sort_values("ic_sel", ascending=False)
    path = os.path.join(OUT, "crypto_hpo.csv")
    res.to_csv(path, index=False)
    top = res.head(3).to_dict("records")
    with open(os.path.join(OUT, "crypto_hpo_best.json"), "w",
              encoding="utf-8") as f:
        json.dump(top, f, ensure_ascii=False, indent=2)
    print(res.head(12).to_string(index=False))
    log(f"结果已写入 {path}")


if __name__ == "__main__":
    main()
