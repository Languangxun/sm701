#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A10: 加密模型 Stacking / Blending (早停修复后的模型)

候选模型 (h 相同):
  E1        : crypto_v2_E1_auc_h{h}.txt         (tech40)
  A1_32     : crypto_v2_A1_32_h{h}.txt          (LambdaRankIC)
  E1Rv      : crypto_idv2_E1Rv_h{h}.txt         (tech40+4 intraday)
  E1Id      : crypto_idv2_E1Id_h{h}.txt         (tech40+11 intraday)

融合: 每日截面 rank 归一后, 用 2024-2025 的 R_ex rank 做线性回归拟合权重,
      2026 为 OOS 验证; 同时报告等权平均与单模型对照.

用法:
    python3 stack_crypto.py --horizon 10
"""
import argparse
import gc
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lgbm_train_backtest import _compute_feats  # noqa: E402
from train_crypto_v2 import build_beta, build_labels, topk_resid, year_ic  # noqa: E402
from train_markets import load_bars, rank_ic  # noqa: E402

OUT = os.path.join(HERE, "markets_output")
FEAT_FILE = os.path.join(HERE, "production", "features_tech.json")
TRAIN_END, VALID_START, TEST_START = "2023-12-31", "2024-01-01", "2026-01-01"


def log(m):
    import time
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def daily_rank(pred, dates):
    """每日截面 rank -> [0,1]"""
    s = pd.Series(pred)
    return s.groupby(pd.Series(dates)).rank(pct=True).values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--models", default="E1,E1Rv,E1Id,A1_32")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    h = args.horizon
    names = [s.strip() for s in args.models.split(",") if s.strip()]

    feats = json.load(open(FEAT_FILE))["features"]
    df = load_bars("crypto_daily")
    df["code"] = df["code"].astype(str)
    beta = build_beta(df)
    log("计算 tech40 特征 ...")
    F = _compute_feats(df.assign(code=df["code"].astype("category")))
    X = pd.DataFrame({k: F[k] for k in feats}, dtype=np.float32)
    from train_crypto_intraday import ID_ALL, ID_RV, build_intraday, load_hourly
    hdf = load_hourly()
    idf = build_intraday(hdf)[["code", "date"] + ID_ALL]
    keys = pd.DataFrame({"code": df["code"].values, "date": df["date"].values})
    idm = keys.merge(idf, on=["code", "date"], how="left")
    assert len(idm) == len(df)
    Xr = pd.concat([X, idm[ID_RV].reset_index(drop=True)], axis=1)
    Xall = pd.concat([X, idm[ID_ALL].reset_index(drop=True)], axis=1)
    del F, hdf, idf, idm
    gc.collect()

    R, R_ex, ok, n_day = build_labels(df, h, beta)
    base_ok = np.asarray(ok & (n_day >= 10) & (R.abs() <= 5.0))
    meta = pd.DataFrame({"date": df["date"][base_ok].values,
                         "R": R[base_ok].values,
                         "R_ex": R_ex[base_ok].values})
    is_va = (meta["date"] >= VALID_START).values
    mva = meta[is_va].reset_index(drop=True)
    te_mask = mva["date"].str[:4].isin(["2026"]).values
    fit_mask = ~te_mask          # 2024-2025 拟合融合权重

    feature_map = {"E1": X, "A1_32": X, "A1_64": X,
                   "E1Rv": Xr, "E1Id": Xall}
    preds, used = {}, []
    for name in names:
        path = os.path.join(OUT, f"crypto_v2_{name}_h{h}.txt")
        if not os.path.exists(path):
            path = os.path.join(OUT, f"crypto_idv2_{name}_h{h}.txt")
        if not os.path.exists(path):
            log(f"跳过 {name}: 模型不存在")
            continue
        Xf = feature_map[name][base_ok].reset_index(drop=True)[is_va]
        model = lgb.Booster(model_file=path)
        preds[name] = model.predict(Xf)
        used.append(name)
        ic, _, _ = rank_ic(preds[name], pd.DataFrame(
            {"date": mva["date"], "R": mva["R_ex"]}), clip=1.0)
        log(f"{name}: IC_full={ic:.4f}")

    # 单模型 + 组合
    def report(tag, pred):
        ic_all, icir_all, _ = rank_ic(
            pred, pd.DataFrame({"date": mva["date"], "R": mva["R_ex"]}),
            clip=1.0)
        ic_te, _, _ = rank_ic(
            pred[te_mask],
            pd.DataFrame({"date": mva["date"][te_mask],
                          "R": mva["R_ex"][te_mask]}), clip=1.0)
        tg_te = topk_resid(pred, mva, h, mask=te_mask)
        log(f"  {tag}: IC_full={ic_all:.4f} IC26(OOS)={ic_te:.4f} "
            f"Top10_26={tg_te['top10_ann_net']:+.1%}")
        return {"blend": tag, "ic_full": round(ic_all, 4),
                "ic26_oos": round(ic_te, 4),
                "top10_26": round(tg_te["top10_ann_net"], 4)}

    rows = []
    for name in used:
        rows.append(report(name, preds[name]))

    if len(used) > 1:
        # 等权 rank 平均
        M = np.column_stack([daily_rank(preds[n], mva["date"].values)
                             for n in used])
        eq = M.mean(axis=1)
        rows.append(report("equal_rank", eq))

        # 线性回归拟合权重 (target: 每日截面 rank 的 R_ex)
        y_rank = daily_rank(mva["R_ex"].values, mva["date"].values)
        from sklearn.linear_model import Ridge
        ridge = Ridge(alpha=1.0, fit_intercept=False)
        ridge.fit(M[fit_mask], y_rank[fit_mask])
        w = dict(zip(used, np.round(ridge.coef_, 4)))
        log(f"  ridge 权重: {w}")
        blend = M @ ridge.coef_
        rows.append(report("ridge", blend))

    res = pd.DataFrame(rows)
    path = os.path.join(OUT, f"crypto_stack_h{h}.csv")
    res.to_csv(path, index=False)
    print(res.to_string(index=False))
    log(f"结果已写入 {path}")


if __name__ == "__main__":
    main()
