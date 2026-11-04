#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
港股/美股/期货/加密货币日线爬取 (akshare 新浪源 + Binance -> stock_cache.db)

表 (stock_cache.db):
  hk_daily     港股日线: code(5位)/date/open/high/low/close/vol
  us_daily     美股日线: code(标普500代码)/date/open/high/low/close/vol
  fut_daily    期货主力连续日线: code(如RB0)/date/open/high/low/close/vol/hold/settle
  crypto_daily 加密货币日线: code(BTCUSDT)/date/open/high/low/close/vol
               vol 为 USDT 计价成交额 (quote volume)
  global_codes   股票池快照 (dataset/code/name)
  global_progress 抓取进度 (dataset, key)

股票池:
  hk      ak.stock_hk_spot()               全部港股 (约2800只)
  us      us_tickers.txt                   标普500成分股 (fetch_global_data 同目录)
  fut     ak.futures_display_main_sina()   全部主力连续合约 (约82个)
  crypto  Binance USDT 现货按 24h 成交额取前 N (默认100, 排除稳定币);
          首次选择后冻结在 crypto_tickers.txt, --crypto-refresh 重选

特性: 断点续跑(global_progress), 失败重试, 多线程抓取+单线程入库, 幂等覆盖

用法:
    python3 fetch_global_data.py --limit 5                 # 冒烟
    python3 fetch_global_data.py                           # 全量(默认3线程)
    python3 fetch_global_data.py --datasets hk,fut         # 指定数据集
    python3 fetch_global_data.py --datasets crypto --crypto-top 100
    python3 fetch_global_data.py --datasets crypto --crypto-refresh  # 重选币池
    python3 fetch_global_data.py --redo                    # 忽略进度重抓
    nohup python3 fetch_global_data.py > global.log 2>&1 &

注意: 新浪源日线均为不复权价, 美股/港股的分红拆股会产生跳变;
      训练侧 (train_markets.py) 对极端收益做截断处理, cross_binary
      标签取截面中位数, 对离群值稳健。
      Binance 日线按 UTC 00:00 切日, 永续/现货连续交易, 无停牌缺口。
"""
import argparse
import os
import random
import re
import sqlite3
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import akshare as ak
import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "stock_cache.db")
TICKER_FILE = os.path.join(HERE, "us_tickers.txt")
CRYPTO_TICKER_FILE = os.path.join(HERE, "crypto_tickers.txt")
START_DEFAULT = "20100101"
BINANCE_BASE = "https://data-api.binance.vision"
CRYPTO_TOP = 100
CRYPTO_MIN_DAYS = 250

STABLE_BASES = {
    "USDT", "USDC", "BUSD", "FDUSD", "TUSD", "DAI", "USDP", "USD1",
    "USDE", "USDS", "PYUSD", "GUSD", "FRAX", "LUSD", "RLUSD", "BFUSD",
    "XUSD", "EUR", "EURI", "AEUR", "GBPT", "TRY", "BRL", "ARS",
    "JPY", "USD", "USTC", "UST",
}
LEVERAGED_SUFFIX = re.compile(r"(3L|3S|5L|5S)$")

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


def universe_crypto(refresh=False):
    if os.path.exists(CRYPTO_TICKER_FILE) and not refresh:
        syms = [s.strip() for s in open(CRYPTO_TICKER_FILE, encoding="utf-8")
                if s.strip() and not s.startswith("#")]
        return [(s, s[:-4] if s.endswith("USDT") else s) for s in syms]

    last_err = None
    for attempt in range(3):
        try:
            info = requests.get(f"{BINANCE_BASE}/api/v3/exchangeInfo",
                                timeout=30).json()
            tick = requests.get(f"{BINANCE_BASE}/api/v3/ticker/24hr",
                                timeout=30).json()
            break
        except Exception as e:
            last_err = e
            time.sleep(2.0 ** attempt + random.random())
    else:
        raise SystemExit(f"Binance 股票池获取失败: {last_err}")

    qv = {r["symbol"]: float(r.get("quoteVolume") or 0) for r in tick}
    pairs = []
    for s in info["symbols"]:
        base = s["baseAsset"]
        if s["quoteAsset"] != "USDT" or s["status"] != "TRADING":
            continue
        if base in STABLE_BASES or LEVERAGED_SUFFIX.search(base):
            continue
        pairs.append((s["symbol"], base))
    pairs.sort(key=lambda x: qv.get(x[0], 0.0), reverse=True)

    min_ms = int((time.time() - CRYPTO_MIN_DAYS * 86_400) * 1000)
    picked = []
    for symbol, base in pairs:
        if len(picked) >= CRYPTO_TOP:
            break
        try:
            r = requests.get(f"{BINANCE_BASE}/api/v3/klines",
                             params={"symbol": symbol, "interval": "1d",
                                     "startTime": 0, "limit": 1}, timeout=20)
            r.raise_for_status()
            first = r.json()
            if first and int(first[0][0]) <= min_ms:
                picked.append((symbol, base))
        except Exception:
            continue
    with open(CRYPTO_TICKER_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(s for s, _ in picked) + "\n")
    return picked


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
    try:
        df = ak.stock_us_daily(symbol=code)
    except Exception:
        if "-" not in code:
            raise
        df = ak.stock_us_daily(symbol=code.replace("-", "."))
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


def fetch_crypto(code, start):
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    rows = []
    while True:
        params = {"symbol": code, "interval": "1d",
                  "startTime": start_ms, "limit": 1000}
        r = requests.get(f"{BINANCE_BASE}/api/v3/klines", params=params,
                         timeout=30)
        if r.status_code == 429:
            time.sleep(5)
            continue
        r.raise_for_status()
        chunk = r.json()
        if not chunk:
            break
        rows.extend(chunk)
        if len(chunk) < 1000:
            break
        start_ms = int(chunk[-1][0]) + 86_400_000
        time.sleep(0.1)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "taker_base",
        "taker_quote", "ignore"])
    now_ms = int(time.time() * 1000)
    df = df[pd.to_numeric(df["close_time"], errors="coerce") < now_ms]
    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.strftime(
        "%Y-%m-%d")
    for c in ["open", "high", "low", "close", "quote_vol"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.rename(columns={"quote_vol": "vol"})
    df.insert(0, "code", code)
    df = df[["code", "date", "open", "high", "low", "close", "vol"]]
    df = df[(df["close"] > 0) & (df["open"] > 0)]
    return df.dropna(subset=["close"]).drop_duplicates("date")


# ---------------------------------------------------------------- 入库
def sql_type(s) -> str:
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_integer_dtype(s):
        return "INTEGER"
    if pd.api.types.is_float_dtype(s):
        return "REAL"
    return "TEXT"


TABLE_OF = {"hk": "hk_daily", "us": "us_daily", "fut": "fut_daily",
            "crypto": "crypto_daily"}


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

    def prune_codes(self, dataset, codes):
        table = TABLE_OF[dataset]
        if not codes:
            return
        ph = ",".join("?" * len(codes))
        with self.con:
            self.con.execute(f'DELETE FROM "{table}" WHERE code NOT IN ({ph})',
                             list(codes))
            self.con.execute(
                f"DELETE FROM global_progress WHERE dataset=? "
                f"AND key NOT IN ({ph})", [dataset] + list(codes))

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
            df = {"hk": fetch_hk, "us": fetch_us, "fut": fetch_fut,
                  "crypto": fetch_crypto}[dataset](key, start)
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
    global CRYPTO_TOP
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB)
    ap.add_argument("--datasets", default="hk,us,fut")
    ap.add_argument("--start", default=START_DEFAULT,
                    help="起始日期 YYYYMMDD (默认 20100101)")
    ap.add_argument("--crypto-top", type=int, default=CRYPTO_TOP,
                    help="加密市场按 24h 成交额取前 N (默认100)")
    ap.add_argument("--crypto-refresh", action="store_true",
                    help="重新按 24h 成交额选择加密币池并写 crypto_tickers.txt")
    ap.add_argument("--limit", type=int, default=0, help="每市场抽样(冒烟)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--retry", type=int, default=3)
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    CRYPTO_TOP = max(1, args.crypto_top)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    for d in datasets:
        if d not in TABLE_OF:
            raise SystemExit(f"未知数据集 {d}, 可选: {', '.join(TABLE_OF)}")

    start = f"{args.start[:4]}-{args.start[4:6]}-{args.start[6:]}"
    writer = Writer(args.db)
    con = writer.con

    tasks = []
    for ds in datasets:
        if ds == "crypto":
            pairs = universe_crypto(args.crypto_refresh)
        else:
            pairs = {"hk": universe_hk, "us": universe_us,
                     "fut": universe_fut}[ds]()
        if args.limit and len(pairs) > args.limit:
            rs = np.random.RandomState(args.seed)
            idx = sorted(rs.choice(len(pairs), args.limit, replace=False))
            pairs = [pairs[i] for i in idx]
        if not (ds == "crypto" and args.limit):
            writer.write_codes(ds, pairs)
        if ds == "crypto" and not args.limit:
            writer.prune_codes(ds, [c for c, _ in pairs])
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
