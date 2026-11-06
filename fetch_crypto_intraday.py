#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
加密货币小时线抓取 (Binance data-api.binance.vision, 1h K线)

补全数据层: 现 crypto_daily 只有日线 OHLCV; 本脚本抓 1h 分时细节,
供 train_crypto_intraday.py 构造日内特征 (已实现波动率/尾盘动量/隔夜收益等)。

表 crypto_hourly(code, ts, open, high, low, close, vol),
  ts 为 UTC 小时线开盘时间 "YYYY-MM-DD HH:00" (与日线 date 按 UTC 日对齐)。

用法:
    python3 fetch_crypto_intraday.py --limit 3                 # 冒烟
    nohup python3 fetch_crypto_intraday.py --workers 4 > crypto_intraday.log 2>&1 &
    python3 fetch_crypto_intraday.py --redo BTCUSDT            # 重抓指定币
"""
import argparse
import os
import random
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "stock_cache.db")
BASE = "https://data-api.binance.vision/api/v3/klines"
TICKER_FILE = os.path.join(HERE, "crypto_tickers.txt")
START_DATE = "2017-01-01"     # BTC 现货最早 2017-08; 之后上市的币自动从上市起


def init_db(con):
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("""CREATE TABLE IF NOT EXISTS crypto_hourly(
        code TEXT NOT NULL, ts TEXT NOT NULL,
        open REAL, high REAL, low REAL, close REAL, vol REAL,
        PRIMARY KEY(code, ts))""")
    con.execute("""CREATE TABLE IF NOT EXISTS crypto_hourly_progress(
        code TEXT PRIMARY KEY, last_ts TEXT, rows INTEGER,
        status TEXT, error TEXT, finished REAL)""")


def fetch_symbol(code, start_ms):
    """分页拉取 1h K 线, 返回 DataFrame[ts,open,high,low,close,vol]"""
    rows = []
    cur = start_ms
    while True:
        params = {"symbol": code, "interval": "1h",
                  "startTime": cur, "limit": 1000}
        for attempt in range(5):
            try:
                r = requests.get(BASE, params=params, timeout=30)
                if r.status_code == 429:
                    time.sleep(5 + attempt * 5)
                    continue
                if r.status_code == 451:
                    raise SystemExit("Binance 区域限制 (451)")
                r.raise_for_status()
                chunk = r.json()
                break
            except SystemExit:
                raise
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(1.5 + attempt * 2 + random.random())
        if not chunk:
            break
        rows.extend(chunk)
        if len(chunk) < 1000:
            break
        cur = int(chunk[-1][0]) + 3_600_000
        time.sleep(0.05)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "taker_base",
        "taker_quote", "ignore"])
    now_ms = int(time.time() * 1000)
    df = df[pd.to_numeric(df["close_time"], errors="coerce") < now_ms]
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms",
                              utc=True).dt.strftime("%Y-%m-%d %H:00")
    for c in ["open", "high", "low", "close", "quote_vol"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.rename(columns={"quote_vol": "vol"})
    df = df[["ts", "open", "high", "low", "close", "vol"]]
    df = df[(df["close"] > 0) & (df["open"] > 0)]
    return df.dropna(subset=["close"]).drop_duplicates("ts")


def write(con, code, df):
    if df.empty:
        return 0
    df = df.copy()
    df.insert(0, "code", code)
    df.to_sql("crypto_hourly", con, if_exists="append", index=False,
              method="multi", chunksize=5000)
    return len(df)


def process(code, start, redo, dry):
    con = sqlite3.connect(DB, timeout=120)
    try:
        init_db(con)
        row = con.execute("SELECT status, last_ts FROM crypto_hourly_progress "
                          "WHERE code=?", (code,)).fetchone()
        cur_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
        if row and row[0] == "done" and not redo:
            return code, "skip", 0, ""
        if row and row[1] and not redo:
            cur_ms = max(cur_ms, int(pd.Timestamp(row[1], tz="UTC")
                                     .timestamp() * 1000) + 3_600_000)
        df = fetch_symbol(code, cur_ms)
        n = write(con, code, df) if not dry else len(df)
        last = df["ts"].iloc[-1] if len(df) else (row[1] if row else None)
        # 若中途断过 (分页未到最新), 状态记为 partial 以便续跑
        latest = df["ts"].max() if len(df) else None
        now = pd.Timestamp.now(tz="UTC")
        done = (latest is not None and
                (now - pd.Timestamp(latest, tz="UTC")).total_seconds() < 3 * 3600)
        status = "done" if done else "partial"
        con.execute("INSERT INTO crypto_hourly_progress VALUES(?,?,?,?,?,?) "
                    "ON CONFLICT(code) DO UPDATE SET last_ts=excluded.last_ts,"
                    "rows=rows+excluded.rows, status=excluded.status,"
                    "error=excluded.error, finished=excluded.finished",
                    (code, last, len(df), status, "", time.time()))
        con.commit()
        return code, status, n, ""
    except Exception as e:
        con.execute("INSERT INTO crypto_hourly_progress VALUES(?,?,?,?,?,?) "
                    "ON CONFLICT(code) DO UPDATE SET status='error',"
                    "error=excluded.error, finished=excluded.finished",
                    (code, None, 0, "error", str(e)[:200], time.time()))
        con.commit()
        return code, "error", 0, str(e)[:200]
    finally:
        con.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--redo", nargs="*", default=[], help="指定重抓的币种")
    ap.add_argument("--symbols", default="", help="子集, 逗号分隔")
    ap.add_argument("--dry", action="store_true", help="只拉取不入库")
    args = ap.parse_args()

    syms = [s.strip() for s in open(TICKER_FILE, encoding="utf-8")
            if s.strip() and not s.startswith("#")]
    if args.symbols:
        syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if args.limit:
        syms = syms[:args.limit]
    redo = set(args.redo)

    start = START_DATE
    print(f"待抓取 (1h) {len(syms)} 个币种, start={start}, redo={sorted(redo)}",
          flush=True)
    ok = skip = err = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process, c, start, redo, args.dry): c for c in syms}
        for f in as_completed(futs):
            code, status, n, e = f.result()
            print(f"[{time.strftime('%H:%M:%S')}] {code}: {status} +{n} 行 {e}",
                  flush=True)
            ok += (status == "done")
            skip += (status == "skip")
            err += (status == "error")
    print(f"完成: done={ok} skip={skip} error={err}", flush=True)


if __name__ == "__main__":
    main()
