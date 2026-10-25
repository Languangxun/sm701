#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成生产定稿三周期模型的样本外预测文件。

读取 `production/features_tech.json`（40 特征名/列序）与对应市场模型
`production/prod_h{1,5,10}_tech.txt`（A股）/ `prod_{mkt}_h{1,5,10}_tech.txt`
（hk/us/fut），用与训练完全一致的 `lgbm_train_backtest._compute_feats`
计算特征，输出 `--out-start`（默认 2024-01-01）之后的全部 (code, date)
三周期概率及其等权平均：

    date,code,prob_h1,prob_h5,prob_h10,prob

`--start`（默认 2022-06-01）仅为特征预热起点，需早于 `--out-start`
至少 60 个交易日以填充 ret60/ma60。

供 stock_predict 的 `test` 档（sm701 直出）历史段流式读取；等权平均即
test 档分数口径（与 `_sm701_live_predict` 实时复算一致）。

用法:
    python3 production/make_predictions.py                 # A股, 2024-01-01 起
    python3 production/make_predictions.py --out-start 2023-01-01
    python3 production/make_predictions.py --market hk     # 港股
    python3 production/make_predictions.py --market us     # 美股
    python3 production/make_predictions.py --market fut    # 期货
"""
import argparse
import gc
import json
import os
import sqlite3
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from lgbm_train_backtest import _compute_feats  # noqa: E402

DB = os.path.join(ROOT, "stock_cache.db")
STOCK_COND = ("(s.code GLOB 'sh60[0135]*' OR s.code GLOB 'sh68[89]*' "
              "OR s.code GLOB 'sz00[0123]*' OR s.code GLOB 'sz30[01]*' "
              "OR s.code GLOB 'bj[489]*')")

MARKETS = {
    "cn":  {"table": "daily_bars", "cond": f"AND {STOCK_COND}", "join": True},
    "hk":  {"table": "hk_daily",  "cond": "", "join": False},
    "us":  {"table": "us_daily",  "cond": "", "join": False},
    "fut": {"table": "fut_daily", "cond": "", "join": False},
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_bars(market, start):
    m = MARKETS[market]
    con = sqlite3.connect(DB)
    q = (f"SELECT b.code,b.date,b.open,b.high,b.low,b.close,b.vol "
         f"FROM {m['table']} b "
         f"{'JOIN stocks s ON s.code=b.code ' if m['join'] else ''}"
         f"WHERE b.date>=? {m['cond']} ORDER BY b.code,b.date")
    log(f"读取行情数据 (>= {start}) ...")
    df = pd.read_sql_query(q, con, params=[start])
    con.close()
    df["code"] = df["code"].astype("category")
    for c in ["open", "high", "low", "close", "vol"]:
        df[c] = df[c].astype("float64")
    log(f"行情数据: {len(df):,} 行, {df['code'].nunique()} 只")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="cn", choices=list(MARKETS),
                    help="市场: cn=A股 / hk / us / fut (各自生产模型)")
    ap.add_argument("--start", default="2022-06-01",
                    help="特征计算起点(预热); 输出从 --out-start 起")
    ap.add_argument("--out-start", default="2024-01-01",
                    help="预测输出起点 (样本外)")
    ap.add_argument("--out", default=None,
                    help="输出文件 (默认 production/valid_predictions"
                         "[_市场].csv.gz)")
    args = ap.parse_args()

    if args.out is None:
        suffix = "" if args.market == "cn" else "_" + args.market
        args.out = os.path.join(
            HERE, f"valid_predictions{suffix}.csv.gz")

    man = json.load(open(os.path.join(HERE, "features_tech.json"),
                         encoding="utf-8"))
    feats = man["features"]
    mpre = "" if args.market == "cn" else args.market + "_"
    boosters = {}
    for h in (1, 5, 10):
        p = os.path.join(HERE, f"prod_{mpre}h{h}_tech.txt")
        b = lgb.Booster(model_file=p)
        assert b.feature_name() == feats, f"h{h} 特征列序与清单不一致"
        boosters[h] = b
        log(f"载入 h{h} 模型: {p} ({b.num_trees()} 树)")

    df = load_bars(args.market, args.start)
    log("计算特征 ...")
    F = _compute_feats(df)
    X = pd.DataFrame({k: np.asarray(F[k], dtype=np.float32) for k in feats},
                     dtype=np.float32)
    del F
    gc.collect()
    log(f"特征矩阵 {X.shape}, 开始预测 ...")

    codes = df["code"].astype(str).to_numpy()
    dates = df["date"].to_numpy()
    del df
    gc.collect()

    probs = {}
    for h, b in boosters.items():
        probs[h] = b.predict(X).astype(np.float32)
        log(f"h{h} 预测完成")
    del X
    gc.collect()
    prob = (probs[1] + probs[5] + probs[10]) / 3.0
    log("三周期等权平均完成")

    out = pd.DataFrame({
        "date": dates, "code": codes,
        "prob_h1": probs[1], "prob_h5": probs[5], "prob_h10": probs[10],
        "prob": prob,
    })
    out = out[out["date"] >= args.out_start]
    log(f"写出 {len(out):,} 行 -> {args.out}")
    out.to_csv(args.out, index=False, compression="gzip")
    log("完成")


if __name__ == "__main__":
    main()
