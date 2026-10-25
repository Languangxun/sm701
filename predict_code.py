#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""自动识别代码所属市场并选择对应策略模型预测 (A股/港股/美股/期货)。

识别规则 (按顺序):
  1. A股:  6 位数字 (可带 sh/sz/bj 前缀), 在 stocks 表验证, e.g. sh600519/000001
  2. 港股: 5 位数字 (自动补齐前导 0), 在 global_codes(hk) 验证, e.g. 00700/700
  3. 美股: 字母代码, 在 global_codes(us) 验证 (标普500成分), e.g. AAPL
  4. 期货: 字母+数字主力连续合约, 在 global_codes(fut) 验证, e.g. RB0

策略选择 (按市场分开定稿, 全部同一语义: 截面跑赢当日中位数的概率, T+1 开盘执行):
  A股  -> production/prod_h{1,5,10}_tech.txt       (生产定稿, 可用)
  港股 -> production/prod_hk_h{1,5,10}_tech.txt    (技能最强, ICIR 2.26, 可用)
  美股 -> production/prod_us_h{1,5,10}_tech.txt    (标普500 技术 IC≈0, 先按此使用)
  期货 -> production/prod_fut_h{1,5,10}_tech.txt   (弱信号, 辅助参考)

特征与训练完全一致 (tech 40 列, lgbm_train_backtest._compute_feats):
取该市场最近约 270 个交易日全市场行情计算特征 (60 日窗口 + 截面排名),
取目标代码最新一行预测。EMA/KDJ 类递推指标仅用 270 日预热, 与全历史
有可忽略的微小差异。

用法:
    python3 predict_code.py sh600519 00700 AAPL RB0
    python3 predict_code.py 600519 700 aapl      # 自动归一化
    python3 predict_code.py 00700 --date 2026-10-01  # 指定信号日(默认最新)
"""
import argparse
import gc
import json
import os
import re
import sqlite3
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lgbm_train_backtest import _compute_feats  # noqa: E402

DB = os.path.join(HERE, "stock_cache.db")
FEAT_FILE = os.path.join(HERE, "production", "features_tech.json")
A_STOCK_COND = ("(s.code GLOB 'sh60[0135]*' OR s.code GLOB 'sh68[89]*' "
                "OR s.code GLOB 'sz00[0123]*' OR s.code GLOB 'sz30[01]*' "
                "OR s.code GLOB 'bj[489]*')")

STRATEGY = {
    "cn":  ("A股", ["production/prod_h1_tech.txt", "production/prod_h5_tech.txt",
                    "production/prod_h10_tech.txt"], "生产定稿三周期 tech，可用"),
    "hk":  ("港股", ["production/prod_hk_h1_tech.txt", "production/prod_hk_h5_tech.txt",
                     "production/prod_hk_h10_tech.txt"],
            "生产定稿三周期 tech，技能最强（ICIR 2.26），可用"),
    "us":  ("美股", ["production/prod_us_h1_tech.txt", "production/prod_us_h5_tech.txt",
                     "production/prod_us_h10_tech.txt"],
            "生产定稿三周期 tech；标普500 技术 IC≈0，先按此使用，仅供参考"),
    "fut": ("期货", ["production/prod_fut_h1_tech.txt", "production/prod_fut_h5_tech.txt",
                     "production/prod_fut_h10_tech.txt"],
            "生产定稿三周期 tech，弱信号（IC 0.03@h10），辅助参考"),
}
MARKET_TABLE = {"cn": "daily_bars", "hk": "hk_daily",
                "us": "us_daily", "fut": "fut_daily"}
WARMUP_DAYS = 400  # 自然日, ≈270 交易日


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def norm_a_code(code):
    c = code.lower()
    if re.fullmatch(r"(sh|sz|bj)\d{6}", c):
        return c
    if re.fullmatch(r"\d{6}", c):
        if c.startswith(("60", "68", "90")):
            return "sh" + c
        if c.startswith(("00", "20", "30")):
            return "sz" + c
        return "bj" + c
    return None


def resolve_code(raw, con):
    """返回 (market, norm_code) 或 (None, 原因)。"""
    raw = raw.strip()
    # 1. A股
    a = norm_a_code(raw)
    if a:
        row = con.execute(
            f"SELECT 1 FROM stocks s WHERE s.code=? AND {A_STOCK_COND}",
            (a,)).fetchone()
        if row:
            return "cn", a
        return None, f"A股 {a} 不在 stocks 表"
    # 2. 港股: 数字代码自动补齐前导 0 (700/0700/00700)
    if re.fullmatch(r"\d{1,5}", raw):
        hk = raw.zfill(5)
        if con.execute("SELECT 1 FROM global_codes WHERE dataset='hk' "
                       "AND code=?", (hk,)).fetchone():
            return "hk", hk
        return None, f"{raw} 未匹配港股池 (5 位港股代码)"
    # 3/4. 美股 (字母) 与期货 (字母+数字)
    up = raw.upper()
    ds = None
    if re.fullmatch(r"[A-Z.\-]{1,6}", up) and \
            con.execute("SELECT 1 FROM global_codes WHERE dataset='us' "
                        "AND code=?", (up,)).fetchone():
        return "us", up
    if re.fullmatch(r"[A-Z]{1,3}\d{1,2}", up):
        if con.execute("SELECT 1 FROM global_codes WHERE dataset='fut' "
                       "AND code=?", (up,)).fetchone():
            return "fut", up
        ds = "fut"
    if ds:
        return None, f"{up} 不在期货主力合约池"
    return None, f"{raw} 无法识别 (美股需在标普500池内)"


def load_market_bars(market, start):
    table = MARKET_TABLE[market]
    con = sqlite3.connect(DB)
    if market == "cn":
        q = (f"SELECT b.code,b.date,b.open,b.high,b.low,b.close,b.vol "
             f"FROM daily_bars b JOIN stocks s ON s.code=b.code "
             f"WHERE b.date>=? AND {A_STOCK_COND} ORDER BY b.code,b.date")
    else:
        q = (f"SELECT code,date,open,high,low,close,vol FROM {table} "
             f"WHERE date>=? ORDER BY code,date")
    df = pd.read_sql_query(q, con, params=[start])
    con.close()
    df["code"] = df["code"].astype("category")
    for c in ["open", "high", "low", "close", "vol"]:
        df[c] = df[c].astype("float64")
    return df


def predict_market(market, codes, on_date=None):
    name, models, note = STRATEGY[market]
    con = sqlite3.connect(DB)
    table = MARKET_TABLE[market]
    row = con.execute(f"SELECT MAX(date) FROM {table}").fetchone()
    con.close()
    if not row or not row[0]:
        return None
    latest = row[0]
    if on_date:
        if on_date > latest:
            return None
        latest = on_date
    start = (pd.Timestamp(latest) - pd.Timedelta(days=WARMUP_DAYS)
             ).strftime("%Y-%m-%d")

    df = load_market_bars(market, start)
    have = set(df["code"].cat.categories) if market == "cn" else \
        set(df["code"].astype(str).unique())
    missing = [c for c in codes if c not in have]
    if len(missing) == len(codes):
        return {"market": market, "error": "代码在该市场行情表中无近期数据 "
                f"(缺失: {missing})"}
    log(f"{name}: {len(df):,} 行 ({df['code'].nunique()} 个代码, "
        f"{df['date'].min()} ~ {df['date'].max()}) 计算特征 ...")

    F = _compute_feats(df)
    X = pd.DataFrame({k: np.asarray(F[k], dtype=np.float32)
                      for k in FEATS}, dtype=np.float32)
    del F
    gc.collect()

    codes_arr = df["code"].astype(str).to_numpy()
    dates_arr = df["date"].to_numpy()
    mask_latest = np.empty(len(df), dtype=bool)
    # 每代码最后一行 (若指定 on_date 则取 <= on_date 的最后一行)
    if on_date is None:
        idx = pd.Series(np.arange(len(df))).groupby(codes_arr).max().values
        mask_latest[idx] = True
    else:
        sel = dates_arr <= on_date
        idx = pd.Series(np.arange(len(df))[sel]).groupby(
            codes_arr[sel]).max().values
        mask_latest[idx] = True

    probs = {}
    for h, path in zip((1, 5, 10), models):
        b = lgb.Booster(model_file=os.path.join(HERE, path))
        probs[h] = b.predict(X).astype(np.float32)
    del X
    gc.collect()
    prob_mean = (probs[1] + probs[5] + probs[10]) / 3.0

    out = {}
    ci = np.flatnonzero(mask_latest)
    for c in codes:
        sel = ci[codes_arr[ci] == c]
        if len(sel) == 0:
            out[c] = None
            continue
        i = sel[0]
        # 当日截面分位 (按综合概率)
        same_day = dates_arr == dates_arr[i]
        pct = float((prob_mean[same_day] <= prob_mean[i]).mean())
        out[c] = {"date": dates_arr[i], "prob_h1": float(probs[1][i]),
                  "prob_h5": float(probs[5][i]), "prob_h10": float(probs[10][i]),
                  "prob": float(prob_mean[i]), "pct": pct,
                  "close": float(df["close"].iloc[i])}
    return {"market": market, "latest": latest, "results": out}


def fmt_p(p):
    return "  ".join(f"h{h}={p[f'prob_h{h}']:.3f}" for h in (1, 5, 10))


def main():
    ap = argparse.ArgumentParser(
        description="自动识别市场并选择策略模型预测")
    ap.add_argument("codes", nargs="*", help="代码, e.g. sh600519 00700 AAPL RB0")
    ap.add_argument("--date", default=None,
                    help="信号日 YYYY-MM-DD (默认每市场最新交易日)")
    args = ap.parse_args()
    if not args.codes:
        ap.print_usage()
        return

    con = sqlite3.connect(DB)
    groups = {}
    errors = []
    for raw in args.codes:
        mkt, res = resolve_code(raw, con)
        if mkt:
            groups.setdefault(mkt, []).append(res)
        else:
            errors.append(f"  {raw}: {res}")
    con.close()

    feats = json.load(open(FEAT_FILE, encoding="utf-8"))["features"]
    globals()["FEATS"] = feats

    for mkt, codes in groups.items():
        r = predict_market(mkt, codes, args.date)
        name, models, note = STRATEGY[mkt]
        print(f"\n=== {name} ===")
        print(f"  策略: {note}")
        print(f"  模型: {', '.join(models)}")
        if r is None:
            print("  无行情数据")
            continue
        if "error" in r:
            print(f"  {r['error']}")
            continue
        print(f"  信号日: {r['latest']}  (T+1 开盘执行; "
              f"概率=截面跑赢当日中位数)")
        for c in codes:
            d = r["results"].get(c)
            if d is None:
                print(f"  {c:<10} 无该日数据")
                continue
            print(f"  {c:<10} 收盘 {d['close']:.2f}  {fmt_p(d)}  "
                  f"综合 {d['prob']:.3f} (截面分位 {d['pct']:.0%})")

    if errors:
        print("\n未识别:")
        print("\n".join(errors))


if __name__ == "__main__":
    main()
