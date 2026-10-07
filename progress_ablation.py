#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""消融训练进度: 读取 ablation.log 输出完成百分比/已用/预计剩余时间"""
import datetime
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "ablation.log")


def main():
    lines = open(LOG, errors="ignore").read().splitlines()
    done = [l for l in lines if "完成:" in l]
    starts = [l for l in lines if "消融配置" in l]
    total = 13
    for l in starts:
        m = re.search(r"消融配置 (\d+) 个", l)
        if m:
            total = int(m.group(1))
    TOTAL = total
    t0 = None
    if starts:
        m = re.search(r"\[(\d\d:\d\d:\d\d)\]", starts[-1])
        t0 = datetime.datetime.combine(
            datetime.date.today(), datetime.time.fromisoformat(m.group(1)))
        now0 = datetime.datetime.now()
        if t0 > now0:
            t0 -= datetime.timedelta(days=1)
    now = datetime.datetime.now()
    el = (now - t0).total_seconds() if t0 else 0
    n = len(done)
    pct = n / TOTAL * 100
    eta = el / n * (TOTAL - n) if n else 0
    bar = "#" * int(pct / 5) + "-" * (20 - int(pct / 5))
    print(f"[{bar}] {n}/{TOTAL} = {pct:.1f}%")
    print(f"已用 {el / 60:.0f} min, 预计剩余 {eta / 60:.0f} min, "
          f"约 {now + datetime.timedelta(seconds=eta):%H:%M} 完成")
    if n:
        print(f"平均 {el / 60 / n:.1f} min/组 (特征越多越慢, 仅供参考)")
        print("最近完成:", done[-1].strip())


if __name__ == "__main__":
    main()
