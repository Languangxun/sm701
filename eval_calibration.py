#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""概率校准评估: 校准曲线/ECE/Brier/PR-AUC/阈值精度 (只读, 用已存验证预测)

输入: lgbm_output/valid_predictions.csv.gz,
      ensemble_output/valid_predictions.csv.gz (date,code,prob*,next_ret,y)
输出: reports/校准评估_YYYYMMDD.md
"""
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (auc, average_precision_score, brier_score_loss,
                             log_loss, precision_recall_curve, roc_auc_score)

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(HERE, "reports")

FILES = {
    "LGBM": (os.path.join(HERE, "lgbm_output", "valid_predictions.csv.gz"),
             ["prob"]),
    "ensemble-LGBM": (os.path.join(HERE, "ensemble_output",
                                   "valid_predictions.csv.gz"),
                      ["prob_lgbm"]),
    "ensemble-TCN": (os.path.join(HERE, "ensemble_output",
                                  "valid_predictions.csv.gz"),
                     ["prob_tcn"]),
    "ensemble-融合": (os.path.join(HERE, "ensemble_output",
                                   "valid_predictions.csv.gz"),
                      ["prob_fuse"]),
}


def reliability(y, p, n_bins=10):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append({"bin": f"{edges[b]:.2f}-{edges[b+1]:.2f}",
                     "n": int(m.sum()),
                     "pred_mean": float(p[m].mean()),
                     "actual": float(y[m].mean())})
    return pd.DataFrame(rows)


def eval_one(name, path, col):
    df = pd.read_csv(path, usecols=["prob", "next_ret", "y"]
                     if col == "prob" else [col, "y"])
    y = df["y"].to_numpy(dtype=np.float32)
    p = df[col].to_numpy(dtype=np.float64)
    df = None
    auc_v = roc_auc_score(y, p)
    ll = log_loss(y, p)
    br = brier_score_loss(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    pr_auc = auc(rec, prec)
    ap = average_precision_score(y, p)
    rel = reliability(y, p)
    ece = float((rel["n"] / rel["n"].sum()
                 * (rel["pred_mean"] - rel["actual"]).abs()).sum())
    thr = []
    for t in (0.50, 0.55, 0.60, 0.65):
        m = p >= t
        thr.append({"thr": t, "n": int(m.sum()),
                    "precision": float(y[m].mean()) if m.sum() else 0.0,
                    "recall": float(m.sum() / (y == 1).sum())})
    return {"name": name, "n": len(y), "auc": auc_v, "logloss": ll,
            "brier": br, "pr_auc": pr_auc, "ap": ap, "ece": ece,
            "rel": rel, "thr": pd.DataFrame(thr)}


def main():
    lines = [f"# 概率校准评估 {time.strftime('%Y-%m-%d %H:%M')}", "",
             "- 对象: 验证集逐样本预测概率(与回测同一批样本)",
             "- ECE: 10等宽分档 | Brier越小越好 | PR-AUC/AP看不平衡下的排序",
             ""]
    print(f"{'模型':<16}{'n':>10}{'AUC':>8}{'LogLoss':>9}{'Brier':>8}"
          f"{'PR-AUC':>8}{'ECE':>8}")
    for name, (path, cols) in FILES.items():
        if not os.path.exists(path):
            print(f"跳过 {name} (文件不存在)")
            continue
        for c in cols:
            r = eval_one(name if len(cols) == 1 else f"{name}", path, c)
            print(f"{r['name']:<16}{r['n']:>10,}{r['auc']:>8.4f}"
                  f"{r['logloss']:>9.4f}{r['brier']:>8.4f}"
                  f"{r['pr_auc']:>8.4f}{r['ece']:>8.4f}")
            lines += [f"## {r['name']} (n={r['n']:,})", "",
                      f"- AUC {r['auc']:.4f} / LogLoss {r['logloss']:.4f} / "
                      f"Brier {r['brier']:.4f} / PR-AUC {r['pr_auc']:.4f} / "
                      f"AP {r['ap']:.4f} / ECE {r['ece']:.4f}", "",
                      "### 校准曲线(预测均值 vs 实际上涨率)",
                      r["rel"].to_markdown(index=False, floatfmt=".4f"), "",
                      "### 阈值精度(该阈值以上样本的实际上涨率)",
                      r["thr"].to_markdown(index=False, floatfmt=".4f"), ""]
    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR,
                        f"校准评估_{time.strftime('%Y%m%d')}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n报告: {path}")


if __name__ == "__main__":
    main()
