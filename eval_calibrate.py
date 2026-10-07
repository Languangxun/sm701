#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Platt/Isotonic 校准实验: 2024年拟合映射, 2025-2026评估
(模型训练集2018-2023, 校准集与测试集均未参与训练, 无污染)

输入: lgbm_output/valid_predictions.csv.gz,
      ensemble_output/valid_predictions.csv.gz
输出: reports/校准对比_YYYYMMDD.md
"""
import os
import time

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, log_loss, roc_auc_score)

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(HERE, "reports")

FILES = {
    "LGBM": (os.path.join(HERE, "lgbm_output", "valid_predictions.csv.gz"),
             "prob"),
    "TCN": (os.path.join(HERE, "ensemble_output", "valid_predictions.csv.gz"),
            "prob_tcn"),
    "融合": (os.path.join(HERE, "ensemble_output", "valid_predictions.csv.gz"),
             "prob_fuse"),
}


def ece_of(y, p, n_bins=10):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    e, n = 0.0, len(y)
    for b in range(n_bins):
        m = idx == b
        if m.sum():
            e += m.sum() / n * abs(p[m].mean() - y[m].mean())
    return float(e)


def thr_table(y, p):
    return {t: (float(y[p >= t].mean()) if (p >= t).any() else 0.0,
                int((p >= t).sum()))
            for t in (0.50, 0.55, 0.60)}


def main():
    out = [f"# Platt/Isotonic 校准对比 {time.strftime('%Y-%m-%d %H:%M')}",
           "", "- 拟合: 2024年预测 / 评估: 2025-2026年预测",
           "- Platt=LogisticRegression, Isotonic=保序回归(clip边界)", ""]
    print(f"{'模型':<8}{'版本':<10}{'AUC':>8}{'LogLoss':>9}{'Brier':>8}"
          f"{'ECE':>8}{'P.50':>7}{'P.55':>7}{'P.60':>7}")
    for name, (path, col) in FILES.items():
        if not os.path.exists(path):
            print(f"跳过 {name}")
            continue
        df = pd.read_csv(path, usecols=["date", col, "y"])
        df["year"] = df["date"].str.slice(0, 4).astype(int)
        cal = df[df["year"] == 2024]
        tst = df[df["year"] >= 2025]
        yc = cal["y"].to_numpy(np.float32)
        pc = cal[col].to_numpy(np.float64)
        yt = tst["y"].to_numpy(np.float32)
        pt = tst[col].to_numpy(np.float64)
        del df, cal, tst
        versions = {"原始": pt}
        try:
            lr = LogisticRegression().fit(pc.reshape(-1, 1), yc)
            versions["Platt"] = lr.predict_proba(pt.reshape(-1, 1))[:, 1]
        except Exception as e:
            print(f"{name} Platt失败: {e}")
        try:
            ir = IsotonicRegression(out_of_bounds="clip").fit(pc, yc)
            lo, hi = pc.min(), pc.max()
            versions["Isotonic"] = ir.predict(np.clip(pt, lo, hi))
        except Exception as e:
            print(f"{name} Isotonic失败: {e}")
        out += [f"## {name} (校准n={len(yc):,}, 测试n={len(yt):,})", "",
                "| 版本 | AUC | LogLoss | Brier | ECE | P≥0.5胜率 | "
                "P≥0.55胜率 | P≥0.6胜率 |",
                "|---|---|---|---|---|---|---|---|"]
        for vname, p in versions.items():
            auc = roc_auc_score(yt, p)
            ll = log_loss(yt, p)
            br = brier_score_loss(yt, p)
            ece = ece_of(yt, p)
            th = thr_table(yt, p)
            print(f"{name:<8}{vname:<10}{auc:>8.4f}{ll:>9.4f}{br:>8.4f}"
                  f"{ece:>8.4f}{th[0.50][0]:>7.4f}{th[0.55][0]:>7.4f}"
                  f"{th[0.60][0]:>7.4f}")
            out.append(f"| {vname} | {auc:.4f} | {ll:.4f} | {br:.4f} | "
                       f"{ece:.4f} | {th[0.50][0]:.4f}(n={th[0.50][1]:,}) | "
                       f"{th[0.55][0]:.4f}(n={th[0.55][1]:,}) | "
                       f"{th[0.60][0]:.4f}(n={th[0.60][1]:,}) |")
        out.append("")
    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, f"校准对比_{time.strftime('%Y%m%d')}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"\n报告: {path}")


if __name__ == "__main__":
    main()
