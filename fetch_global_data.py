#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
港股/美股/期货日线爬取 (akshare 新浪源 -> stock_cache.db)

表 (stock_cache.db):
  hk_daily  港股日线: code(5位)/date/open/high/low/close/vol
  us_daily  美股日线: code(标普500代码)/date/open/high/low/close/vol
  fut_daily 期货主力连续日线: code(如RB0)/date/open/high/low/close/vol/hold/settle
  global_codes   股票池快照 (dataset/code/name)
  global_progress 抓取进度 (dataset, key)

股票池:
  hk  ak.stock_hk_spot()               全部港股 (约2800只)
  us  us_tickers.txt                   标普500成分股 (fetch_global_data 同目录)
  fut ak.futures_display_main_sina()   全部主力连续合约 (约82个)

特性: 断点续跑(global_progress), 失败重试, 多线程抓取+单线程入库, 幂等覆盖

用法:
    python3 fetch_global_data.py --limit 5                 # 冒烟
    python3 fetch_global_data.py                           # 全量(默认3线程)
    python3 fetch_global_data.py --datasets hk,fut         # 指定数据集
    python3 fetch_global_data.py --redo                    # 忽略进度重抓
    nohup python3 fetch_global_data.py > global.log 2>&1 &

注意: 新浪源日线均为不复权价, 美股/港股的分红拆股会产生跳变;
      训练侧 (train_markets.py) 对极端收益做截断处理, cross_binary
      标签取截面中位数, 对离群值稳健。
"""
import argparse
import os
import random
import sqlite3
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import akshare as ak
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "stock_cache.db")
TICKER_FILE = os.path.join(HERE, "us_tickers.txt")
START_DEFAULT = "20100101"

BAR_COLS = {"日期": "date", "开盘价": "open", "最高价": "high",
            "最低价": "low", "收盘价": "close", "成交量": "vol",
            "持仓量": "hold", "动态结算价": "settle"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fmt_date(s) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.strftime("%Y-%m-%d")


# ---------------------------------------------------------------- 股票池
def universe_hk():
    df = ak.stock_hk_spot()
    df = df[df["代码"].astype(str).str.fullmatch(r"\d{5}")]
    return list(zip(df["代码"].astype(str), df["中文名称"].fillna("")))


def universe_us():
    if not os.path.exists(TICKER_FILE):
        raise SystemExit(f"缺少股票池文件 {TICKER_FILE} "
                         "(标普500成分, 每行一个代码)")
    syms = [s.strip() for s in open(TICKER_FILE) if s.strip()]
    return list(zip(syms, [""] * len(syms)))


def universe_fut():
    df = ak.futures_display_main_sina()
    return list(zip(df["symbol"].astype(str), df["name"].astype(str)))


# ---------------------------------------------------------------- 抓取
def fetch_hk(code, start):
    df = ak.stock_hk_daily(symbol=code)
    df = df.rename(columns={"volume": "vol"})
    df["date"] = fmt_date(df["date"])
    df = df[df["date"] >= start]
    df = df[["date", "open", "high", "low", "close", "vol"]]
    df.insert(0, "code", code)
    return df.dropna(subset=["close"])


def fetch_us(code, start):
    df = ak.stock_us_daily(symbol=code)
    df["date"] = fmt_date(df["date"])
    df = df[df["date"] >= start]
    df = df[["date", "open", "high", "low", "close", "volume"]]
    df = df.rename(columns={"volume": "vol"})
    df.insert(0, "code", code)
    return df.dropna(subset=["close"])


def fetch_fut(code, start):
    end = time.strftime("%Y%m%d")
    df = ak.futures_main_sina(symbol=code, start_date=start, end_date=end)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns=BAR_COLS)
    df["date"] = fmt_date(df["date"])
    df.insert(0, "code", code)
    return df.dropna(subset=["close"])


# ---------------------------------------------------------------- 入库
def sql_type(s) -> str:
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_integer_dtype(s):
        return "INTEGER"
    if pd.api.types.is_float_dtype(s):
        return "REAL"
    return "TEXT"


TABLE_OF = {"hk": "hk_daily", "us": "us_daily", "fut": "fut_daily"}


class Writer:
    def __init__(self, db):
        self.con = sqlite3.connect(db, timeout=120)
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("PRAGMA synchronous=NORMAL")
        self.con.execute("""CREATE TABLE IF NOT EXISTS global_progress(
            dataset TEXT NOT NULL, key TEXT NOT NULL, status TEXT NOT NULL,
            rows INTEGER, error TEXT, finished REAL,
            PRIMARY KEY(dataset, key))""")
        for ds, table in TABLE_OF.items():
            self.con.execute(f"""CREATE TABLE IF NOT EXISTS "{table}"(
                code TEXT NOT NULL, date TEXT NOT NULL, open REAL, high REAL,
                low REAL, close REAL, vol REAL, hold REAL, settle REAL,
                PRIMARY KEY(code, date))""")
        self.con.execute("""CREATE TABLE IF NOT EXISTS global_codes(
            dataset TEXT NOT NULL, code TEXT NOT NULL, name TEXT,
            PRIMARY KEY(dataset, code))""")
        self.con.commit()

    def write_bars(self, dataset, df):
        table = TABLE_OF[dataset]
        cols = ["code", "date", "open", "high", "low", "close", "vol"]
        for extra in ("hold", "settle"):
            if extra in df.columns:
                cols.append(extra)
        df = df[cols].copy()
        for c in df.columns:
            if c not in ("code", "date"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["date", "open", "close"])
        if df.empty:
            return 0
        data = df.astype(object).where(pd.notna(df), None)
        collist = ", ".join(f'"{c}"' for c in cols)
        ph = ", ".join("?" * len(cols))
        with self.con:
            self.con.execute(f'DELETE FROM "{table}" WHERE code=?',
                             (df["code"].iloc[0],))
            self.con.executemany(
                f'INSERT INTO "{table}" ({collist}) VALUES ({ph})',
                list(data.itertuples(index=False, name=None)))
        return len(df)

    def write_codes(self, dataset, pairs):
        with self.con:
            self.con.execute(
                "DELETE FROM global_codes WHERE dataset=?", (dataset,))
            self.con.executemany(
                "INSERT OR REPLACE INTO global_codes(dataset,code,name) "
                "VALUES(?,?,?)", [(dataset, c, n) for c, n in pairs])

    def mark(self, dataset, key, status, rows, error):
        with self.con:
            self.con.execute(
                "INSERT INTO global_progress(dataset,key,status,rows,error,"
                "finished) VALUES(?,?,?,?,?,?) "
                "ON CONFLICT(dataset,key) DO UPDATE SET status=excluded.status,"
                " rows=excluded.rows, error=excluded.error,"
                " finished=excluded.finished",
                (dataset, key, status, rows, error, time.time()))

    def stats(self):
        for table in TABLE_OF.values():
            try:
                c = self.con.execute(
                    f'SELECT COUNT(*), COUNT(DISTINCT code) FROM "{table}"'
                ).fetchone()
                print(f"  {table}: {c[0]:,} 行 / {c[1]:,} 个代码")
            except sqlite3.OperationalError:
                pass


def worker(args_tuple):
    dataset, key, start, retry, sleep = args_tuple
    df, err = None, None
    for attempt in range(max(1, retry)):
        try:
            df = {"hk": fetch_hk, "us": fetch_us, "fut": fetch_fut}[dataset](
                key, start)
            if df is None:
                df = pd.DataFrame()
            if not df.empty:
                return dataset, key, df, None
            err = "empty"
        except Exception as e:
            df, err = None, f"{type(e).__name__}: {e}"
        time.sleep(min(60.0, 2.0 ** attempt + random.random()))
    if sleep > 0:
        time.sleep(sleep)
    if err == "empty":
        return dataset, key, pd.DataFrame(), None
    return dataset, key, None, err or "unknown error"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB)
    ap.add_argument("--datasets", default="hk,us,fut")
    ap.add_argument("--start", default=START_DEFAULT,
                    help="起始日期 YYYYMMDD (默认 20100101)")
    ap.add_argument("--limit", type=int, default=0, help="每市场抽样(冒烟)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--retry", type=int, default=3)
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    for d in datasets:
        if d not in TABLE_OF:
            raise SystemExit(f"未知数据集 {d}, 可选: {', '.join(TABLE_OF)}")

    start = f"{args.start[:4]}-{args.start[4:6]}-{args.start[6:]}"
    writer = Writer(args.db)
    con = writer.con

    tasks = []
    for ds in datasets:
        pairs = {"hk": universe_hk, "us": universe_us,
                 "fut": universe_fut}[ds]()
        if args.limit and len(pairs) > args.limit:
            rs = np.random.RandomState(args.seed)
            idx = sorted(rs.choice(len(pairs), args.limit, replace=False))
            pairs = [pairs[i] for i in idx]
        writer.write_codes(ds, pairs)
        log(f"{ds} 股票池 {len(pairs)} 个代码")
        tasks += [(ds, c, start, args.retry, args.sleep) for c, _ in pairs]

    done = set()
    if not args.redo:
        done = set(con.execute(
            "SELECT dataset, key FROM global_progress WHERE status='ok'"))
    tasks = [t for t in tasks if (t[0], t[1]) not in done]
    log(f"待抓取 {len(tasks):,} 个任务 (已完成 {len(done):,}), 起始 {start}")
    if not tasks:
        writer.stats()
        return

    t0 = time.time()
    total = len(tasks)
    n_ok = n_fail = n_empty = 0
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(worker, t) for t in tasks]
            for fut in as_completed(futs):
                dataset, key, df, err = fut.result()
                if err:
                    writer.mark(dataset, key, "fail", None, err)
                    n_fail += 1
                elif df is None or df.empty:
                    writer.mark(dataset, key, "ok", 0, "empty")
                    n_empty += 1
                else:
                    rows = writer.write_bars(dataset, df)
                    writer.mark(dataset, key, "ok", rows, None)
                    n_ok += 1
                n = n_ok + n_fail + n_empty
                if n % args.log_every == 0 or n == total:
                    el = time.time() - t0
                    speed = n / el if el > 0 else 0
                    eta = (total - n) / speed / 60 if speed > 0 else 0
                    log(f"进度 {n:,}/{total:,} (ok {n_ok:,}, empty {n_empty:,}"
                        f", fail {n_fail:,}), {speed:.2f} 任务/s, "
                        f"已用 {el / 60:.1f} min, 预计剩余 {eta:.0f} min")
    except KeyboardInterrupt:
        log("收到中断, 可重跑续传")

    if n_fail:
        log("失败样例:")
        for r in con.execute("SELECT dataset,key,error FROM global_progress "
                             "WHERE status='fail' LIMIT 5"):
            print(f"  {r[0]} {r[1]}: {r[2][:120]}")
    writer.stats()


if __name__ == "__main__":
    main()
