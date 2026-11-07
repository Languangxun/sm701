#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A3: 成本感知标签 / 直接优化净收益消融

核心思路: 标签从「跑赢截面中位数」改为「预期净收益覆盖成本」
  y = 1{R_ex > cost_rt / h}      (硬门槛)
  或 sample_weight = max(0, |R_ex| - cost_rt/h)  (软权重)

对照组 (同 E1 口径、同超参、同 splits):
  E1           : baseline (cross_binary on R_ex)
  E1_cost_hard : y = (R_ex > cost/h)
  E1_cost_soft : baseline y, sample_weight = max(0, |R_ex| - cost/h)
  E1_cost_mix  : y = (R_ex > cost/h), sample_weight = max(0, |R_ex| - cost/h)

评价: 验证集 RankIC(R_ex)、Top10 residual 年化 (扣双边 0.2%)
重点看 h1 是否转正、h5/h10 是否受损。

用法:
    python3 train_crypto_cost.py --variants E1,E1_cost_hard,E1_cost_soft,E1_cost_mix --horizons 1,5,10
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

DB = os.path.join(HERE, "stock_cache.db")
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
    del F
    gc.collect()

    results = []
    threads = args.threads or os.cpu_count()
    for h in args.horizons:
        R, R_ex, ok, n_day = build_labels(df, h, beta)
        cost_thresh = COST_RT / h
        base_ok = np.asarray(ok & (n_day >= 10) & (R.abs() <= 5.0))
        Xok = X[base_ok].reset_index(drop=True)
        meta = pd.DataFrame({"date": df["date"][base_ok].values,
                             "R": R[base_ok].values,
                             "R_ex": R_ex[base_ok].values})
        med = R_ex[base_ok].groupby(df["date"][base_ok]).transform("median")

        # === 四种标签/权重构造 ===
        # E1: baseline cross_binary on R_ex
        y_E1 = (R_ex[base_ok] > med).astype(np.int8).reset_index(drop=True)
        w_E1 = None

        # E1_cost_hard: 硬门槛 y = (R_ex > cost/h)  -- 已验证失败，保留作对照
        y_hard = (R_ex[base_ok] > cost_thresh).astype(np.int8).reset_index(drop=True)
        w_hard = None

        # E1_cost_soft: baseline y, 权重 = max(0, |R_ex| - cost/h)
        w_soft = (R_ex[base_ok].abs() - cost_thresh).clip(lower=0).astype(np.float32).reset_index(drop=True)

        # E1_cost_mix: 硬门槛 y + 软权重
        w_mix = (R_ex[base_ok].abs() - cost_thresh).clip(lower=0).astype(np.float32).reset_index(drop=True)

        # E1_cost_double: 双条件 y = (R_ex > median) & (R_ex > cost/h)
        y_double = ((R_ex[base_ok] > med) & (R_ex[base_ok] > cost_thresh)).astype(np.int8).reset_index(drop=True)
        w_double = None

        # E1_cost_weight_only: baseline y, 权重 = max(0, R_ex - cost/h) 只奖励正超额
        w_pos = (R_ex[base_ok] - cost_thresh).clip(lower=0).astype(np.float32).reset_index(drop=True)

        is_tr = (meta["date"] <= TRAIN_END).values
        is_va = (meta["date"] >= VALID_START).values
        mva = meta[is_va].reset_index(drop=True)
        yva_E1 = y_E1[is_va].values
        yva_hard = y_hard[is_va].values

        configs = [
            ("E1", Xok, y_E1, w_E1, yva_E1),
            ("E1_cost_hard", Xok, y_hard, w_hard, yva_hard),
            ("E1_cost_soft", Xok, y_E1, w_soft, yva_E1),
            ("E1_cost_mix", Xok, y_hard, w_mix, yva_hard),
            ("E1_cost_double", Xok, y_double, w_double, y_double[is_va].values),
            ("E1_cost_weight_only", Xok, y_E1, w_pos, yva_E1),
        ]

        for name, Xfull, yfull, wfull, yva in configs:
            if name not in args.variants:
                continue
            Xtr, ytr = Xfull[is_tr], yfull[is_tr]
            Xva = Xfull[is_va]
            wtr = (wfull[is_tr] if wfull is not None else None)
            pos_rate = ytr.mean()
            log(f"训练 crypto cost-aware h={h} variant={name} "
                f"(train={len(ytr)}, valid={len(yva)}, feats={Xtr.shape[1]}, pos_rate={pos_rate:.4f}) ...")
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
                   "pos_rate": round(float(pos_rate), 4),
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
            log(f"  {name} h={h}: pos={pos_rate:.3f} AUC={aucs:.4f} IC_Rex={ic_Rex:.4f} "
                f"ICIR={icir_Rex:.2f} year={year_ics} "
                f"Top10_ann={tg[10]['topk_ann_net']:+.1%}")
            model.booster_.save_model(os.path.join(
                OUT, f"crypto_cost_{name}_h{h}.txt"))
            del model, pred
            gc.collect()

    res = pd.DataFrame(results)
    path = os.path.join(OUT, "crypto_cost.csv")
    res.to_csv(path, index=False)
    print(res.to_string(index=False))
    log(f"结果已写入 {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="E1,E1_cost_hard,E1_cost_soft,E1_cost_mix")
    ap.add_argument("--horizons", default="1,5,10")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    args.horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    args.variants = {v.strip() for v in args.variants.split(",") if v.strip()}
    run(args)


if __name__ == "__main__":
    main()