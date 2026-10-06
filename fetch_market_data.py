#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大盘指数 + 申万一级行业行情/成分映射爬取 (akshare -> stock_cache.db)

表:
  index_bars     大盘指数日线 (code, name, date, open, high, low, close, vol)
  industry_bars  申万一级行业指数日线 (industry_code, industry_name, date, open,
                 high, low, close, vol, amount)
  stock_industry 个股->申万一级行业最新映射 (code, industry_code, industry_name,
                 weight, start_date, updated)

用法:
    python3 fetch_market_data.py               # 全部
    python3 fetch_market_data.py --indices-only
    python3 fetch_market_data.py --industry-only
"""
import argparse
import os
import sqlite3
import time

import akshare as ak
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "stock_cache.db")

INDICES = [
    ("sh000001", "上证指数"), ("sh000016", "上证50"),
    ("sz399001", "深证成指"), ("sz399006", "创业板指"),
    ("sh000300", "沪深300"), ("sh000905", "中证500"),
    ("sh000852", "中证1000"), ("sh000688", "科创50"),
    ("sh000985", "中证全指"), ("bj899050", "北证50"),
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def with_retry(fn, *args, retry=3, **kwargs):
    last = None
    for i in range(retry):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last = e
            time.sleep(1.0 + i * 2.0)
    raise last


def with_prefix(c: str) -> str:
    c = str(c).zfill(6)
    if c.startswith(("60", "68", "90")):
        return "sh" + c
    if c.startswith(("00", "20", "30")):
        return "sz" + c
    return "bj" + c


def write_table(con, table, df, key_cols):
    if df.empty:
        return 0
    cols = list(df.columns)
    ph = ", ".join("?" * len(cols))
    collist = ", ".join(f'"{c}"' for c in cols)
    data = df.astype(object).where(pd.notna(df), None)
    rows = list(data.itertuples(index=False, name=None))
    with con:
        for kc in key_cols:
            con.execute(f'DELETE FROM "{table}" WHERE {kc[0]}=?', (kc[1],))
        con.executemany(
            f'INSERT INTO "{table}" ({collist}) VALUES ({ph})', rows)
    return len(df)


def fetch_indices(con):
    con.execute("""CREATE TABLE IF NOT EXISTS index_bars(
        code TEXT, name TEXT, date TEXT, open REAL, high REAL, low REAL,
        close REAL, vol REAL, PRIMARY KEY(code, date))""")
    for code, name in INDICES:
        try:
            df = with_retry(ak.stock_zh_index_daily, symbol=code)
        except Exception as e:
            log(f"指数 {code} {name} 失败: {type(e).__name__}: {e}")
            continue
        if df is None or df.empty:
            log(f"指数 {code} {name} 无数据")
            continue
        df = df.rename(columns={"volume": "vol"})
        df.insert(0, "code", code)
        df.insert(1, "name", name)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df = df[["code", "name", "date", "open", "high", "low", "close", "vol"]]
        n = write_table(con, "index_bars", df, [("code", code)])
        log(f"指数 {code} {name}: {n:,} 行 "
            f"({df['date'].min()} ~ {df['date'].max()})")


def fetch_industry(con):
    con.execute("""CREATE TABLE IF NOT EXISTS industry_bars(
        industry_code TEXT, industry_name TEXT, date TEXT, open REAL, high REAL,
        low REAL, close REAL, vol REAL, amount REAL, PRIMARY KEY(industry_code, date))""")
    con.execute("""CREATE TABLE IF NOT EXISTS stock_industry(
        code TEXT PRIMARY KEY, industry_code TEXT, industry_name TEXT,
        weight REAL, start_date TEXT, updated TEXT)""")
    first = with_retry(ak.sw_index_first_info)
    log(f"申万一级行业 {len(first)} 个")
    total_bars = total_cons = 0
    for _, row in first.iterrows():
        code = str(row["行业代码"]).split(".")[0]
        name = str(row["行业名称"])
        try:
            bars = with_retry(ak.index_hist_sw, symbol=code, period="day")
        except Exception as e:
            log(f"行业行情 {code} {name} 失败: {type(e).__name__}: {e}")
            bars = None
        if bars is not None and not bars.empty:
            bars = bars.rename(columns={
                "日期": "date", "开盘": "open", "最高": "high", "最低": "low",
                "收盘": "close", "成交量": "vol", "成交额": "amount"})
            bars.insert(0, "industry_code", code)
            bars.insert(1, "industry_name", name)
            bars["date"] = pd.to_datetime(bars["date"]).dt.strftime("%Y-%m-%d")
            keep = ["industry_code", "industry_name", "date", "open", "high",
                    "low", "close", "vol", "amount"]
            bars = bars[keep]
            total_bars += write_table(con, "industry_bars", bars,
                                      [("industry_code", code)])
        try:
            cons = with_retry(ak.index_component_sw, symbol=code)
        except Exception as e:
            log(f"行业成分 {code} {name} 失败: {type(e).__name__}: {e}")
            cons = None
        if cons is not None and not cons.empty:
            cons = cons.rename(columns={
                "证券代码": "raw_code", "最新权重": "weight",
                "计入日期": "start_date"})
            cons["code"] = cons["raw_code"].map(with_prefix)
            cons["industry_code"] = code
            cons["industry_name"] = name
            cons["start_date"] = pd.to_datetime(
                cons["start_date"], errors="coerce").dt.strftime("%Y-%m-%d")
            cons["updated"] = time.strftime("%Y-%m-%d")
            cons = cons[["code", "industry_code", "industry_name", "weight",
                         "start_date", "updated"]].drop_duplicates("code")
            ph = ", ".join("?" * len(cons.columns))
            with con:
                con.executemany(
                    f'INSERT INTO stock_industry(code,industry_code,'
                    f'industry_name,weight,start_date,updated) VALUES({ph}) '
                    f'ON CONFLICT(code) DO UPDATE SET '
                    f'industry_code=excluded.industry_code,'
                    f'industry_name=excluded.industry_name,'
                    f'weight=excluded.weight,start_date=excluded.start_date,'
                    f'updated=excluded.updated',
                    list(cons.astype(object).where(
                        pd.notna(cons), None).itertuples(index=False, name=None)))
            total_cons += len(cons)
        log(f"行业 {code} {name}: 行情 {0 if bars is None else len(bars)} 行, "
            f"成分 {0 if cons is None else len(cons)} 只")
    log(f"行业汇总: 行情 {total_bars:,} 行, 成分映射 {total_cons:,} 条")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB)
    ap.add_argument("--indices-only", action="store_true")
    ap.add_argument("--industry-only", action="store_true")
    args = ap.parse_args()

    con = sqlite3.connect(args.db, timeout=120)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    t0 = time.time()
    if not args.industry_only:
        log("抓取大盘指数 ...")
        fetch_indices(con)
    if not args.indices_only:
        log("抓取申万一级行业 ...")
        fetch_industry(con)
    for t in ("index_bars", "industry_bars", "stock_industry"):
        try:
            n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print(f"  {t}: {n:,} 行")
        except sqlite3.OperationalError:
            pass
    con.close()
    log(f"完成, 用时 {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
