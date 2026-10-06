#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
补充数据爬取: 资金流/融资融券/龙虎榜/业绩预告快报/QVIX/国债收益率

表 (stock_cache.db):
  stock_fund_flow  个股资金流: 主力/超大/大/中/小单净额+净占比 (约最近120个交易日)
  margin_detail    融资融券明细: 融资余额/买入/偿还, 融券余量/卖出/偿还 (2010-03-31起)
  lhb_detail       龙虎榜明细 (2010-01起)
  yjyg             业绩预告 (按报告期, 含公告日期)
  yjkb             业绩快报 (按报告期, 含公告日期)
  qvix             期权波动率指数 QVIX (50ETF/300ETF, 2015起)
  bond_yield       中美国债收益率 (2015起)

特性: 断点续跑(extra_progress), 失败重试, 多线程抓取+单线程入库, 幂等覆盖

用法:
    python3 fetch_extra_data.py --limit 20                 # 冒烟
    python3 fetch_extra_data.py                            # 全量(默认3线程)
    python3 fetch_extra_data.py --datasets margin,lhb      # 指定数据集
    python3 fetch_extra_data.py --redo                     # 忽略进度重抓
    nohup python3 fetch_extra_data.py > extra.log 2>&1 &
"""
import argparse
import calendar
import gc
import os
import random
import socket
import sqlite3
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import akshare as ak
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
socket.setdefaulttimeout(60)

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "stock_cache.db")

CODE_COND = ("(code GLOB 'sh60[0135]*' OR code GLOB 'sh68[89]*' "
             "OR code GLOB 'sz00[0123]*' OR code GLOB 'sz30[01]*' "
             "OR code GLOB 'bj*')")

FLOW_RENAME = {
    "日期": "date", "收盘价": "close", "涨跌幅": "pct_chg",
    "主力净流入-净额": "main_net", "主力净流入-净占比": "main_pct",
    "超大单净流入-净额": "xl_net", "超大单净流入-净占比": "xl_pct",
    "大单净流入-净额": "l_net", "大单净流入-净占比": "l_pct",
    "中单净流入-净额": "m_net", "中单净流入-净占比": "m_pct",
    "小单净流入-净额": "s_net", "小单净流入-净占比": "s_pct",
}
MARGIN_SSE = {
    "信用交易日期": "date", "标的证券代码": "raw_code", "标的证券简称": "name",
    "融资余额": "rz_bal", "融资买入额": "rz_buy", "融资偿还额": "rz_repay",
    "融券余量": "rq_vol", "融券卖出量": "rq_sell", "融券偿还量": "rq_repay",
}
MARGIN_SZSE = {
    "证券代码": "raw_code", "证券简称": "name", "融资买入额": "rz_buy",
    "融资余额": "rz_bal", "融券卖出量": "rq_sell", "融券余量": "rq_vol",
    "融券余额": "rq_bal", "融资融券余额": "total_bal",
}
BOND_RENAME = {
    "日期": "date", "中国国债收益率2年": "cn_2y", "中国国债收益率5年": "cn_5y",
    "中国国债收益率10年": "cn_10y", "中国国债收益率30年": "cn_30y",
    "中国国债收益率10年-2年": "cn_10y_2y", "美国国债收益率10年": "us_10y",
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def with_prefix(c: str) -> str:
    c = str(c).zfill(6)
    if c.startswith(("60", "68", "90")):
        return "sh" + c
    if c.startswith(("00", "20", "30")):
        return "sz" + c
    return "bj" + c


def fmt_date(s) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.strftime("%Y-%m-%d")


# ---------------------------------------------------------------- 抓取
def fetch_flow(code):
    market = code[:2]
    df = ak.stock_individual_fund_flow(stock=code[2:], market=market)
    df = df.rename(columns=FLOW_RENAME)
    df.insert(0, "code", code)
    df["date"] = fmt_date(df["date"])
    return df


def fetch_qvix(code):
    if code == "50etf":
        df = ak.index_option_50etf_qvix()
    else:
        df = ak.index_option_300etf_qvix()
    df.insert(0, "code", code)
    df["date"] = fmt_date(df["date"])
    return df


def fetch_bond(_):
    df = ak.bond_zh_us_rate(start_date="20150101")
    df = df.rename(columns=BOND_RENAME)
    df = df[["date"] + [c for c in BOND_RENAME.values() if c != "date"]]
    df["date"] = fmt_date(df["date"])
    return df


def fetch_yjyg(period):
    df = ak.stock_yjyg_em(date=period)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    if "股票代码" in df.columns:
        df["code"] = df["股票代码"].map(with_prefix)
    df["report_period"] = period
    return df


def fetch_yjkb(period):
    df = ak.stock_yjkb_em(date=period)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    if "股票代码" in df.columns:
        df["code"] = df["股票代码"].map(with_prefix)
    df["report_period"] = period
    return df


def fetch_lhb(month):
    y, m = int(month[:4]), int(month[4:])
    start = month + "01"
    end = f"{month}{calendar.monthrange(y, m)[1]:02d}"
    df = ak.stock_lhb_detail_em(start_date=start, end_date=end)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    if "代码" in df.columns:
        df["code"] = df["代码"].map(with_prefix)
    return df


def fetch_margin(dt, market):
    if market == "sse":
        df = ak.stock_margin_detail_sse(date=dt)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns=MARGIN_SSE)
    else:
        df = ak.stock_margin_detail_szse(date=dt)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns=MARGIN_SZSE)
    df = df.copy()
    # 两融标的含 ETF(51x/15x), 用交易所前缀而非代码段推断
    pre = "sh" if market == "sse" else "sz"
    df["code"] = pre + df["raw_code"].astype(str).str.zfill(6)
    df["date"] = f"{dt[:4]}-{dt[4:6]}-{dt[6:]}"
    df["market"] = market
    return df


FETCHERS = {"flow": fetch_flow, "qvix": fetch_qvix, "yjyg": fetch_yjyg,
            "yjkb": fetch_yjkb, "lhb": fetch_lhb}


# ---------------------------------------------------------------- 任务/删除规则
def build_tasks(con, datasets, limit, seed):
    dates = [r[0] for r in con.execute(
        "SELECT DISTINCT date FROM index_bars WHERE code='sh000001' "
        "AND date >= '2010-03-31' ORDER BY date")]
    codes = [r[0] for r in con.execute(
        f"SELECT code FROM stocks WHERE {CODE_COND} ORDER BY code")]
    if limit and len(codes) > limit:
        rs = np.random.RandomState(seed)
        codes = sorted(rs.choice(codes, size=limit, replace=False).tolist())

    periods = []
    for y in range(2010, 2027):
        for md in ("0331", "0630", "0930", "1231"):
            if f"{y}{md}" <= time.strftime("%Y%m%d"):
                periods.append(f"{y}{md}")
    months = []
    for y in range(2010, 2027):
        for m in range(1, 13):
            if f"{y}{m:02d}" <= time.strftime("%Y%m"):
                months.append(f"{y}{m:02d}")

    tasks = []
    if "qvix" in datasets:
        tasks += [("qvix", "50etf"), ("qvix", "300etf")]
    if "bond" in datasets:
        tasks += [("bond", "all")]
    if "yjyg" in datasets:
        tasks += [("yjyg", p) for p in periods]
    if "yjkb" in datasets:
        tasks += [("yjkb", p) for p in periods]
    if "lhb" in datasets:
        tasks += [("lhb", m) for m in months]
    if "margin" in datasets:
        dts = [d.replace("-", "") for d in dates]
        tasks += [("margin", f"{d}|sse") for d in dts]
        tasks += [("margin", f"{d}|szse") for d in dts]
    if "flow" in datasets:
        tasks += [("flow", c) for c in codes]
    return tasks


def do_fetch(dataset, key):
    if dataset == "qvix":
        return fetch_qvix(key)
    if dataset == "bond":
        return fetch_bond(key)
    if dataset == "yjyg":
        return fetch_yjyg(key)
    if dataset == "yjkb":
        return fetch_yjkb(key)
    if dataset == "lhb":
        return fetch_lhb(key)
    if dataset == "flow":
        return fetch_flow(key)
    if dataset == "margin":
        dt, market = key.split("|")
        return fetch_margin(dt, market)
    raise ValueError(dataset)


TABLE_OF = {"flow": "stock_fund_flow", "qvix": "qvix", "yjyg": "yjyg",
            "yjkb": "yjkb", "lhb": "lhb_detail", "margin": "margin_detail",
            "bond": "bond_yield"}


def delete_spec(dataset, key):
    if dataset == "flow":
        return 'DELETE FROM "stock_fund_flow" WHERE code=?', [key]
    if dataset == "qvix":
        return 'DELETE FROM "qvix" WHERE code=?', [key]
    if dataset == "bond":
        return 'DELETE FROM "bond_yield"', []
    if dataset in ("yjyg", "yjkb"):
        return f'DELETE FROM "{dataset}" WHERE report_period=?', [key]
    if dataset == "lhb":
        y, m = int(key[:4]), int(key[4:])
        end = f"{key}{calendar.monthrange(y, m)[1]:02d}"
        return ('DELETE FROM "lhb_detail" WHERE "上榜日" BETWEEN ? AND ?',
                [key + "01", end])
    if dataset == "margin":
        dt, market = key.split("|")
        d = f"{dt[:4]}-{dt[4:6]}-{dt[6:]}"
        return ('DELETE FROM "margin_detail" WHERE date=? AND market=?',
                [d, market])
    raise ValueError(dataset)


# ---------------------------------------------------------------- 入库
def sql_type(s) -> str:
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_integer_dtype(s):
        return "INTEGER"
    if pd.api.types.is_float_dtype(s):
        return "REAL"
    return "TEXT"


class Writer:
    def __init__(self, db):
        self.con = sqlite3.connect(db, timeout=120)
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("PRAGMA synchronous=NORMAL")
        self.con.execute("""CREATE TABLE IF NOT EXISTS extra_progress(
            dataset TEXT NOT NULL, key TEXT NOT NULL, status TEXT NOT NULL,
            rows INTEGER, error TEXT, finished REAL,
            PRIMARY KEY(dataset, key))""")
        self.cols = {}
        self.con.commit()

    def _ensure(self, table, df):
        if table not in self.cols:
            existing = {r[1].lower(): r[1] for r in self.con.execute(
                f'PRAGMA table_info("{table}")')}
            if not existing:
                cols = ", ".join(f'"{c}" {sql_type(df[c])}' for c in df.columns)
                self.con.execute(f'CREATE TABLE "{table}" ({cols})')
                existing = {str(c).lower(): c for c in df.columns}
            self.cols[table] = existing
        for c in df.columns:
            if str(c).lower() not in self.cols[table]:
                self.con.execute(
                    f'ALTER TABLE "{table}" ADD COLUMN "{c}" {sql_type(df[c])}')
                self.cols[table][str(c).lower()] = c

    def write(self, table, df, dsql, dparams):
        df = df.copy()
        seen, keep = set(), []
        for c in df.columns:
            if str(c).lower() in seen:
                continue
            seen.add(str(c).lower())
            keep.append(c)
        df = df[keep]
        for c in df.columns:
            if c != "code" and pd.api.types.is_datetime64_any_dtype(df[c]):
                df[c] = pd.to_datetime(df[c], errors="coerce").dt.strftime(
                    "%Y-%m-%d %H:%M:%S")
        self._ensure(table, df)
        with self.con:
            self.con.execute(dsql, dparams)
            if len(df):
                collist = ", ".join(f'"{c}"' for c in df.columns)
                ph = ", ".join("?" * len(df.columns))
                data = df.astype(object).where(pd.notna(df), None)
                self.con.executemany(
                    f'INSERT INTO "{table}" ({collist}) VALUES ({ph})',
                    list(data.itertuples(index=False, name=None)))
        return len(df)

    def mark(self, dataset, key, status, rows, error):
        with self.con:
            self.con.execute(
                "INSERT INTO extra_progress(dataset,key,status,rows,error,"
                "finished) VALUES(?,?,?,?,?,?) "
                "ON CONFLICT(dataset,key) DO UPDATE SET status=excluded.status,"
                " rows=excluded.rows, error=excluded.error,"
                " finished=excluded.finished",
                (dataset, key, status, rows, error, time.time()))


def worker(dataset, key, retry, sleep):
    df, err = None, None
    for attempt in range(max(1, retry)):
        try:
            df = do_fetch(dataset, key)
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
    ap.add_argument("--datasets", default="qvix,bond,yjyg,yjkb,lhb,margin,flow")
    ap.add_argument("--limit", type=int, default=0, help="股票随机抽样(flow)")
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

    writer = Writer(args.db)
    con = writer.con
    tasks = build_tasks(con, datasets, args.limit, args.seed)
    done = set()
    if not args.redo:
        done = set(con.execute(
            "SELECT dataset, key FROM extra_progress WHERE status='ok'"))
    tasks = [t for t in tasks if t not in done]
    log(f"待抓取 {len(tasks):,} 个任务 (已完成 {len(done):,})")
    if not tasks:
        return

    t0 = time.time()
    total = len(tasks)
    n_ok = n_fail = 0
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(worker, d, k, args.retry, args.sleep)
                    for d, k in tasks]
            for fut in as_completed(futs):
                dataset, key, df, err = fut.result()
                if err:
                    writer.mark(dataset, key, "fail", None, err)
                    n_fail += 1
                else:
                    table = TABLE_OF[dataset]
                    dsql, dparams = delete_spec(dataset, key)
                    rows = writer.write(table, df, dsql, dparams)
                    writer.mark(dataset, key, "ok", rows, None)
                    n_ok += 1
                n = n_ok + n_fail
                if n % args.log_every == 0 or n == total:
                    el = time.time() - t0
                    speed = n / el if el > 0 else 0
                    eta = (total - n) / speed / 60 if speed > 0 else 0
                    log(f"进度 {n:,}/{total:,} (ok {n_ok:,}, fail {n_fail:,}), "
                        f"{speed:.2f} 任务/s, 已用 {el / 60:.1f} min, "
                        f"预计剩余 {eta:.0f} min")
    except KeyboardInterrupt:
        log("收到中断, 可重跑续传")

    for t in TABLE_OF.values():
        try:
            c = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            print(f"  {t}: {c:,} 行")
        except sqlite3.OperationalError:
            pass
    log(f"本次完成 {n_ok:,} 成功 / {n_fail:,} 失败, "
        f"总用时 {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
