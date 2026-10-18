#!/usr/bin/env python3
"""对比 cross_binary 与基线 walkforward 结果."""
import os
import sys
import pandas as pd


def load(p):
    df = pd.read_csv(p)
    # 取 holdout 行
    hold = df[df["window"] == "holdout"].copy()
    return hold


def main():
    base = "ablation_output/walkforward_results.csv"
    cross = "ablation_output_cross_binary/walkforward_results.csv"

    if not os.path.exists(base):
        print(f"基线文件不存在: {base}")
        sys.exit(1)
    if not os.path.exists(cross):
        print(f"cross_binary 结果尚未生成: {cross}")
        sys.exit(1)

    b = load(base)
    c = load(cross)

    print("=" * 70)
    print("Holdout 对比: cross_binary vs binary 基线")
    print("=" * 70)
    print(f"{'h':>3} | {'AUC_base':>10} {'AUC_cross':>10} {'dAUC':>8} | "
          f"{'IC_base':>10} {'IC_cross':>10} {'dIC':>8} | "
          f"{'ICIR_base':>10} {'ICIR_cross':>10} {'dICIR':>8}")
    print("-" * 100)
    for _, row in b.iterrows():
        h = int(row["horizon"])
        c_row = c[c["horizon"] == h]
        if len(c_row) == 0:
            continue
        cr = c_row.iloc[0]
        auc_b = row["valid_auc"]
        auc_c = cr["valid_auc"]
        ic_b = row["ic"]
        ic_c = cr["ic"]
        icir_b = row["ic_ir"]
        icir_c = cr["ic_ir"]
        print(f"{h:>3} | {auc_b:>10.4f} {auc_c:>10.4f} {auc_c-auc_b:>+8.4f} | "
              f"{ic_b:>10.4f} {ic_c:>10.4f} {ic_c-ic_b:>+8.4f} | "
              f"{icir_b:>10.4f} {icir_c:>10.4f} {icir_c-icir_b:>+8.4f}")

    # 额外打印 Top20 年化净收益
    print("\nTop20 年化净收益对比")
    print(f"{'h':>3} | {'Top20_base':>12} {'Top20_cross':>12} {'dTop20':>10}")
    print("-" * 50)
    for _, row in b.iterrows():
        h = int(row["horizon"])
        c_row = c[c["horizon"] == h]
        if len(c_row) == 0 or "Top20_ann_net" not in row:
            continue
        cr = c_row.iloc[0]
        t_b = row["Top20_ann_net"]
        t_c = cr["Top20_ann_net"]
        print(f"{h:>3} | {t_b:>12.4f} {t_c:>12.4f} {t_c-t_b:>+10.4f}")


if __name__ == "__main__":
    main()
