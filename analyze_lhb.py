#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""龙虎榜深挖: 只用龙虎榜特征训练, 分年度/分月拆解 Top20 收益, 判断"179%"真伪

输出: ablation_output/lhb_deepdive.md 与 lhb_monthly.csv
"""
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from lgbm_train_backtest import (DATA_START, TRAIN_END, build_features,
                                 load_bars, log)
from features_extra import build_lhb_features

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "ablation_output")


def yearly_stats(bt):
    rows = []
    for yr, g in bt.groupby("year"):
        dates = g["date"].unique()
        top20 = g[g["rk"] <= 20].groupby("date")["next_ret"].mean()
        top50 = g[g["rk"] <= 50].groupby("date")["next_ret"].mean()
        bench = g.groupby("date")["next_ret"].mean()
        days = len(top20)
        ann = (1.0 + top20).prod() ** (252.0 / days) - 1.0
        ann50 = (1.0 + top50).prod() ** (252.0 / days) - 1.0
        annb = (1.0 + bench).prod() ** (252.0 / days) - 1.0
        rows.append({"year": yr, "days": days, "n": len(g),
                     "Top20_ann": ann, "Top50_ann": ann50,
                     "bench_ann": annb,
                     "Top20_win": float((top20 > 0).mean()),
                     "Top20_best_day": float(top20.max()),
                     "Top20_worst_day": float(top20.min())})
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_bars(None, 42)
    X, next_ret, y, tradable = build_features(df)
    year = df["date"].str.slice(0, 4).astype(int)
    dates = df["date"].values
    code_cat = df["code"].cat.categories.to_numpy()
    codes = code_cat[df["code"].cat.codes.to_numpy()]
    base = ((year >= int(DATA_START[:4])) & next_ret.notna() & tradable
            & X["ret60"].notna()).values
    m_tr = (base & (year <= int(TRAIN_END[:4])).values)
    m_va = (base & ~(year <= int(TRAIN_END[:4])).values)
    tr_idx = np.flatnonzero(m_tr)
    va_idx = np.flatnonzero(m_va)
    del X
    y_all = y.to_numpy(dtype=np.float32)
    nr = next_ret.to_numpy(dtype=np.float32)

    mat, names = build_lhb_features(dates, codes)
    log(f"龙虎榜特征 {len(names)}: {names}, 训练 {len(tr_idx):,} / 验证 {len(va_idx):,}")
    age = (pd.Timestamp(TRAIN_END) - pd.to_datetime(dates[tr_idx])).days.values
    w_tr = (0.5 ** (age / 756.0)).astype(np.float32)
    params = {"objective": "binary", "metric": ["auc"],
              "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 1000,
              "feature_fraction": 0.8, "bagging_fraction": 0.8,
              "bagging_freq": 1, "lambda_l2": 20.0, "max_bin": 127,
              "num_threads": os.cpu_count(), "verbosity": -1, "seed": 42}
    dtr = lgb.Dataset(mat[tr_idx], label=y_all[tr_idx], weight=w_tr,
                      feature_name=names)
    dva = lgb.Dataset(mat[va_idx], label=y_all[va_idx], reference=dtr)
    model = lgb.train(params, dtr, num_boost_round=1000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
    p_va = model.predict(mat[va_idx], num_iteration=model.best_iteration)
    auc = roc_auc_score(y_all[va_idx], p_va)
    log(f"lhb模型 valid_auc {auc:.4f}, best_iter {model.best_iteration}")
    imp = pd.DataFrame({"feature": names,
                        "gain": model.feature_importance("gain")}
                       ).sort_values("gain", ascending=False)
    print(imp.to_string(index=False))

    bt = pd.DataFrame({"date": dates[va_idx], "code": codes[va_idx],
                       "prob": p_va, "next_ret": nr[va_idx],
                       "y": y_all[va_idx], "lhb_cnt28": mat[va_idx][:, 0]})
    bt["year"] = bt["date"].str.slice(0, 4)
    bt["rk"] = bt.groupby("date")["prob"].rank(ascending=False,
                                               method="first")
    def daily_ic(g):
        if len(g) < 50 or g["prob"].std() == 0:
            return np.nan
        return g["prob"].rank().corr(g["next_ret"].rank())

    ic = bt.groupby("date").apply(daily_ic).dropna()
    ic_yr = bt.groupby("year").apply(
        lambda g: g.groupby("date").apply(daily_ic).dropna().mean())
    print(f"\nRankIC 均值 {ic.mean():.4f}, ICIR {ic.mean() / (ic.std() + 1e-12):.3f}")
    print("分年度IC:", {k: round(v, 4) for k, v in ic_yr.items()})

    # 对照: "近28天上过龙虎榜"股票池等权(不看模型), 验证收益是否只是股票池beta
    act = bt[bt["lhb_cnt28"] > 0]
    act_daily = act.groupby("date")["next_ret"].mean()
    print("\n==== 龙虎榜股票池等权对照 ====")
    univ_rows = []
    for yr, g in act.groupby("year"):
        d = g.groupby("date")["next_ret"].mean()
        univ_rows.append({
            "year": yr, "days": len(d),
            "avg_stocks": float(g.groupby("date").size().mean()),
            "universe_ann": float((1 + d).prod() ** (252 / len(d)) - 1),
            "best_day": float(d.max()), "worst_day": float(d.min())})
    univ = pd.DataFrame(univ_rows)
    print(univ.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    ys = yearly_stats(bt)
    print("\n==== 分年度 ====")
    print(ys.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    top20d = bt[bt["rk"] <= 20].groupby("date")["next_ret"].mean()
    daily = top20d.reset_index()
    daily["month"] = daily["date"].str.slice(0, 7)
    mon = daily.groupby("month").apply(
        lambda g: (1 + g["next_ret"]).prod() - 1).reset_index(
        name="top20_ret")
    mon.to_csv(os.path.join(OUT_DIR, "lhb_monthly.csv"), index=False)
    top_days = daily.sort_values("next_ret", ascending=False).head(10)
    contrib = top_days["next_ret"].sum() / max(top20d.sum(), 1e-9)

    lines = [f"# 龙虎榜深度分析 {time.strftime('%Y-%m-%d %H:%M')}", "",
             f"- 特征 {len(names)} 个, valid_auc {auc:.4f}, "
             f"best_iter {model.best_iteration}",
             f"- RankIC 均值 **{ic.mean():.4f}**, ICIR "
             f"{ic.mean() / (ic.std() + 1e-12):.3f}, "
             f"分年度: " + ", ".join(f"{k}:{v:.4f}"
                                     for k, v in ic_yr.items()),
             f"- 单日Top20收益最高10天合计占总收益比例: **{contrib:.1%}**",
             "", "## 特征重要度", "",
             imp.to_markdown(index=False, floatfmt=".1f"),
             "", "## 分年度", "",
             ys.to_markdown(index=False, floatfmt=".4f"),
             "", "## 对照: 龙虎榜股票池等权(不用模型)", "",
             univ.to_markdown(index=False, floatfmt=".4f"),
             "", "## 分月Top20收益 (2024+)", "",
             mon.to_markdown(index=False, floatfmt=".4f"),
             "", "## 单日收益最高10天", "",
             top_days.to_markdown(index=False, floatfmt=".4f")]
    path = os.path.join(OUT_DIR, "lhb_deepdive.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log(f"报告: {path}, 用时 {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
