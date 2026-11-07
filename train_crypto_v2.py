#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
加密重训 v2: 修复早停 + 方法对比

背景: LightGBM 4.7 sklearn 接口会静默忽略构造函数里的 callbacks,
      旧脚本(train_markets.py / train_crypto_ab.py / ...)实际未早停,
      全部跑满 3000 轮, 严重过拟合. 本脚本统一改用 lgb.train.

对比 (E1 标签: beta 中性 R_ex 的截面中位数二分类):
  E1_auc      : early stopping on validation AUC          (推荐候选)
  E1_logloss  : early stopping on validation logloss      (旧行为近似, 无早停)
  A1_32/64    : LambdaRankIC 目标 + early stopping on IC

评估:
  - ES 期默认 2024-01 ~ 2025-12, 2026 为未参与早停的 out-of-time 测试;
  - 报告全验证集 (2024+, 有 ES 泄漏, 供对照) 与 2026-only 两套 IC;
  - Top10 residual 年化 (全验证集 / 2026-only), 扣双边 0.2%.

用法:
    python3 train_crypto_v2.py --variants E1_auc,E1_logloss,A1_32,A1_64 --horizons 5,10
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
TRAIN_END = "2023-12-31"
VALID_START = "2024-01-01"
ES_END = "2025-12-31"
TEST_START = "2026-01-01"

PARAMS = {
    "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 1000,
    "feature_fraction": 0.6, "bagging_fraction": 0.7, "bagging_freq": 1,
    "lambda_l2": 20, "max_bin": 127, "verbosity": -1,
}


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


def year_ic(pred, meta):
    out = {}
    for yr in ("2024", "2025", "2026"):
        sel = (meta["date"].str[:4] == yr).values
        if sel.sum() > 100:
            ic, _, _ = rank_ic(
                pred[sel],
                pd.DataFrame({"date": meta["date"][sel],
                              "R": meta["R_ex"][sel]}), clip=1.0)
            out[yr] = round(ic, 4)
        else:
            out[yr] = None
    return out


def topk_resid(pred, meta, h, mask=None):
    m = np.ones(len(meta), bool) if mask is None else mask
    d = pd.DataFrame({"date": meta["date"].values[m],
                      "p": pred[m],
                      "Rex": np.clip(meta["R_ex"].values[m], -1.0, 1.0)})
    tk = d.groupby("date").apply(
        lambda g: g.nlargest(10, "p")["Rex"].mean() if len(g) >= 10 else np.nan,
        include_groups=False)
    return {"top10_R": float(tk.mean()),
            "top10_ann_net": (float(tk.mean()) - COST_RT) / h * DAYS,
            "cover": float(tk.notna().mean())}


def fit_binary(Xtr, ytr, Xes, yes, Xva, yva, metric, rounds, patience,
               params, threads, wtr=None, wes=None):
    dtr = lgb.Dataset(Xtr, label=ytr, weight=wtr)
    des = lgb.Dataset(Xes, label=yes, reference=dtr, weight=wes)
    p = dict(params, objective="binary", metric=[metric], num_threads=threads)
    model = lgb.train(p, dtr, num_boost_round=rounds, valid_sets=[des],
                      callbacks=[lgb.early_stopping(patience, verbose=False),
                                 lgb.log_evaluation(0)])
    pred = model.predict(Xva, num_iteration=model.best_iteration)
    return model, pred


def fit_rankic(Xtr, Rtr, dates_tr, Xes, Res, dates_es, Xva, rounds, patience,
               params, threads, mpairs):
    dtr = lgb.Dataset(Xtr, label=Rtr)
    des = lgb.Dataset(Xes, label=Res, reference=dtr)
    fobj = make_lambdarank_ic_objective(Rtr, dates_tr,
                                        max_pairs_per_day=mpairs)
    feval = make_rank_ic_eval(Res, dates_es)
    p = dict(params, objective=fobj, metric="None", num_threads=threads)
    model = lgb.train(p, dtr, num_boost_round=rounds, valid_sets=[des],
                      feval=feval,
                      callbacks=[lgb.early_stopping(patience, verbose=False),
                                 lgb.log_evaluation(0)])
    pred = model.predict(Xva, num_iteration=model.best_iteration)
    return model, pred


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

    results = []
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
        is_es = ((meta["date"] >= VALID_START) &
                 (meta["date"] <= ES_END)).values
        is_te = (meta["date"] >= TEST_START).values
        is_va = (meta["date"] >= VALID_START).values
        mva = meta[is_va].reset_index(drop=True)
        pva = np.full(len(mva), np.nan, dtype=np.float64)
        test_off = np.cumsum(is_va) - 1

        Xtr = Xok[is_tr]
        Xes = Xok[is_es]
        Xva = Xok[is_va]
        ytr, yes, yva = y[is_tr], y[is_es], y[is_va]

        configs = []
        for name in ("E1_auc", "E1_logloss", "E1_w3"):
            if name in args.variants:
                metric = "binary_logloss" if name == "E1_logloss" else "auc"
                configs.append((name, "binary", metric))
        for name, mp in (("A1_32", 32), ("A1_64", 64)):
            if name in args.variants:
                configs.append((name, "rankic", mp))

        # E3 权重: 截面数截断 (max 50) × 时间衰减 (半衰期 2 年)
        n_day_ok = n_day.values[base_ok]
        years = (pd.Timestamp(TRAIN_END) - pd.to_datetime(
            meta["date"])).dt.days.values / 365.0
        w3 = ((np.clip(n_day_ok, None, 50) / 50.0) *
              (0.5 ** (np.clip(years, 0, None) / 2.0))).astype(np.float32)

        for name, kind, extra in configs:
            log(f"h={h} {name} 训练 ...")
            t0 = time.time()
            if kind == "binary":
                if name == "E1_w3":
                    model, pred = fit_binary(
                        Xtr, ytr, Xes, yes, Xva, yva, extra,
                        args.rounds, args.patience, PARAMS, threads,
                        wtr=w3[is_tr], wes=w3[is_es])
                else:
                    model, pred = fit_binary(
                        Xtr, ytr, Xes, yes, Xva, yva, extra,
                        args.rounds, args.patience, PARAMS, threads)
            else:
                model, pred = fit_rankic(
                    Xtr, meta["R_ex"].values[is_tr], meta["date"].values[is_tr],
                    Xes, meta["R_ex"].values[is_es], meta["date"].values[is_es],
                    Xva, args.rounds, args.patience, PARAMS, threads, extra)
            best = model.best_iteration
            auc = roc_auc_score(yva, pred)
            ic_all, icir_all, _ = rank_ic(
                pred, pd.DataFrame({"date": mva["date"], "R": mva["R_ex"]}),
                clip=1.0)
            yics = year_ic(pred, mva)
            te_mask = mva["date"].str[:4].isin(["2026"]).values
            te = mva[te_mask].reset_index(drop=True)
            ic_te, icir_te, _ = rank_ic(
                pred[te_mask],
                pd.DataFrame({"date": te["date"], "R": te["R_ex"]}), clip=1.0)
            tg_all = topk_resid(pred, mva, h)
            tg_te = topk_resid(pred, mva, h, mask=te_mask)
            row = {"variant": name, "h": h, "best_iter": best,
                   "auc_full": round(auc, 4),
                   "ic_full": round(ic_all, 4),
                   "icir_full": round(icir_all, 3),
                   "ic24": yics["2024"], "ic25": yics["2025"],
                   "ic26": yics["2026"],
                   "ic26_oos": round(ic_te, 4),
                   "top10_full": round(tg_all["top10_ann_net"], 4),
                   "top10_26": round(tg_te["top10_ann_net"], 4),
                   "secs": round(time.time() - t0, 1)}
            results.append(row)
            log(f"  {name} h={h}: best={best} AUC={auc:.4f} "
                f"IC_full={ic_all:.4f} IC26(OOS)={ic_te:.4f} "
                f"IC24/25/26={yics['2024']}/{yics['2025']}/{yics['2026']} "
                f"Top10_full={tg_all['top10_ann_net']:+.1%} "
                f"Top10_26={tg_te['top10_ann_net']:+.1%} ({row['secs']}s)")
            model.save_model(os.path.join(OUT, f"crypto_v2_{name}_h{h}.txt"))
            del model
            gc.collect()

    res = pd.DataFrame(results)
    path = os.path.join(OUT, "crypto_v2.csv")
    res.to_csv(path, index=False)
    print(res.to_string(index=False))
    log(f"结果已写入 {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="E1_auc,E1_logloss,A1_32,A1_64")
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
