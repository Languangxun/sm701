#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全市场股票历史基本面数据爬取 (akshare -> SQLite, 写入 stock_cache.db)

数据表 (均以 fund_ 为前缀, 每行带 code/date):
  fund_valuation        stock_value_em                       2018年至今每日估值(PE/PB/PS/PCF/市值/股本)
  fund_valuation_baidu  stock_zh_valuation_baidu             百度估值长历史(总市值/PE/PB/PCF, 1999年起)
  fund_abstract         stock_financial_abstract             财务摘要(关键指标, 按报告期, 上市至今)
  fund_indicator        stock_financial_analysis_indicator   新浪财务分析指标(报告期, ~86列)
  fund_indicator_em     stock_financial_analysis_indicator_em 东财财务指标(报告期, ~141列)
  fund_balance          stock_balance_sheet_by_report_em     资产负债表(报告期, 全字段)
  fund_profit           stock_profit_sheet_by_report_em      利润表(报告期, 全字段)
  fund_cashflow         stock_cash_flow_sheet_by_report_em   现金流量表(报告期, 全字段)
  fund_fhps             stock_fhps_detail_em                 分红送配(报告期)
  fund_gdhs             stock_zh_a_gdhs_detail_em            股东户数(统计截止日)

特性:
  - 全量历史: 不限起始年份、不限数据量, 抓到接口返回的全部历史
  - 断点续跑: fund_progress 表记录每只股票每类数据的完成状态, 中断后重跑自动跳过已完成
  - 失败重试: 指数退避自动重试, 单只股票失败不中断整体
  - 多线程抓取 + 单线程入库, 幂等写入(重抓覆盖旧数据)
  - 表结构自动扩展: 东财报表列数随公司类型变化, 新列自动 ALTER TABLE

用法:
    python3 fetch_fundamentals.py --limit 3                       # 冒烟测试
    python3 fetch_fundamentals.py                                 # 全量(默认4线程)
    python3 fetch_fundamentals.py --datasets valuation,fhps,gdhs  # 只抓部分数据集
    python3 fetch_fundamentals.py --workers 6 --sleep 0.1         # 调并发
    python3 fetch_fundamentals.py --redo                          # 忽略进度全部重抓
    nohup python3 fetch_fundamentals.py > fundamental.log 2>&1 &  # 后台长跑
"""
import argparse
import os
import random
import re
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

# 只抓 A 股股票(含北交所), 排除 ETF/基金/指数
CODE_COND = ("(code GLOB 'sh60[0135]*' OR code GLOB 'sh68[89]*' "
             "OR code GLOB 'sz00[0123]*' OR code GLOB 'sz30[01]*' "
             "OR code GLOB 'bj*')")

VAL_EN = {
    "数据日期": "date", "当日收盘价": "close", "当日涨跌幅": "pct_chg",
    "总市值": "total_mv", "流通市值": "float_mv", "总股本": "total_share",
    "流通股本": "float_share", "PE(TTM)": "pe_ttm", "PE(静)": "pe_static",
    "市净率": "pb", "PEG值": "peg", "市现率": "pcf", "市销率": "ps",
}

DATASET_SPECS = {
    "valuation":       ("fund_valuation", "date"),
    "valuation_baidu": ("fund_valuation_baidu", "date"),
    "abstract":        ("fund_abstract", "report_date"),
    "indicator":       ("fund_indicator", "report_date"),
    "indicator_em":    ("fund_indicator_em", "report_date"),
    "balance":         ("fund_balance", "report_date"),
    "profit":          ("fund_profit", "report_date"),
    "cashflow":        ("fund_cashflow", "report_date"),
    "fhps":            ("fund_fhps", "report_date"),
    "gdhs":            ("fund_gdhs", "report_date"),
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def plain(code: str) -> str:
    return code[2:]


def dot(code: str) -> str:
    return f"{code[2:]}.{code[:2].upper()}"


def em(code: str) -> str:
    return code[:2].upper() + code[2:]


def fmt_date(s) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.strftime("%Y-%m-%d")


# ---------------------------------------------------------------- 各数据集抓取
def fetch_valuation(code):
    df = ak.stock_value_em(symbol=plain(code))
    df = df.rename(columns=VAL_EN)
    df["date"] = fmt_date(df["date"])
    return df


def fetch_valuation_baidu(code):
    frames = []
    for ind in ("总市值", "市盈率(TTM)", "市盈率(静)", "市净率", "市现率"):
        d = ak.stock_zh_valuation_baidu(symbol=plain(code), indicator=ind,
                                        period="全部")
        if d is None or d.empty:
            continue
        d = d.copy()
        d["indicator"] = ind
        frames.append(d)
    if not frames:
        return pd.DataFrame(columns=["date", "value", "indicator"])
    df = pd.concat(frames, ignore_index=True)
    df["date"] = fmt_date(df["date"])
    return df[["date", "value", "indicator"]]


def fetch_abstract(code):
    raw = ak.stock_financial_abstract(symbol=plain(code))
    if raw is None or raw.empty:
        return pd.DataFrame()
    period_cols = [c for c in raw.columns if re.fullmatch(r"\d{8}", str(c))]
    if not period_cols:
        return pd.DataFrame()
    names = raw["指标"].astype(str)
    dup = names.duplicated(keep=False)
    names = names.where(~dup, raw["选项"].astype(str) + "|" + names)
    out = raw[period_cols].T
    out.columns = names.tolist()
    out.index = fmt_date(pd.Series(out.index)).to_numpy()
    return out.reset_index().rename(columns={"index": "report_date"})


def fetch_indicator(code):
    df = ak.stock_financial_analysis_indicator(symbol=plain(code))
    df = df.rename(columns={"日期": "report_date"})
    df["report_date"] = fmt_date(df["report_date"])
    return df


def _derive_report_date(df):
    df = df.loc[:, ~df.columns.duplicated()]
    if "REPORT_DATE" in df.columns:
        df["report_date"] = fmt_date(df["REPORT_DATE"])
        df = df.drop(columns=["REPORT_DATE"])
    return df


def fetch_indicator_em(code):
    df = ak.stock_financial_analysis_indicator_em(symbol=dot(code),
                                                  indicator="按报告期")
    return _derive_report_date(df)


def fetch_balance(code):
    df = ak.stock_balance_sheet_by_report_em(symbol=em(code))
    return _derive_report_date(df)


def fetch_profit(code):
    df = ak.stock_profit_sheet_by_report_em(symbol=em(code))
    return _derive_report_date(df)


def fetch_cashflow(code):
    df = ak.stock_cash_flow_sheet_by_report_em(symbol=em(code))
    return _derive_report_date(df)


def fetch_fhps(code):
    df = ak.stock_fhps_detail_em(symbol=plain(code))
    df = df.rename(columns={"报告期": "report_date"})
    df["report_date"] = fmt_date(df["report_date"])
    return df


def fetch_gdhs(code):
    df = ak.stock_zh_a_gdhs_detail_em(symbol=plain(code))
    df = df.rename(columns={"股东户数统计截止日": "report_date"})
    df["report_date"] = fmt_date(df["report_date"])
    return df


FETCHERS = {
    "valuation": fetch_valuation,
    "valuation_baidu": fetch_valuation_baidu,
    "abstract": fetch_abstract,
    "indicator": fetch_indicator,
    "indicator_em": fetch_indicator_em,
    "balance": fetch_balance,
    "profit": fetch_profit,
    "cashflow": fetch_cashflow,
    "fhps": fetch_fhps,
    "gdhs": fetch_gdhs,
}

for _t in (np.int8, np.int16, np.int32, np.int64,
           np.uint8, np.uint16, np.uint32, np.uint64, np.bool_):
    sqlite3.register_adapter(_t, int)
for _t in (np.float16, np.float32, np.float64):
    sqlite3.register_adapter(_t, float)


# ---------------------------------------------------------------- 入库
def normalize(code, df, date_col):
    if df is None:
        return pd.DataFrame({"code": pd.Series(dtype=str), date_col: []})
    df = df.copy()
    df = df.loc[:, ~df.columns.duplicated()]
    seen = set()
    keep = []
    for c in df.columns:
        if str(c).lower() in seen:
            continue
        seen.add(str(c).lower())
        keep.append(c)
    df = df[keep]
    df.insert(0, "code", code)
    for c in df.columns:
        if c == "code":
            continue
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.strftime(
                "%Y-%m-%d %H:%M:%S")
    if date_col in df.columns:
        df[date_col] = fmt_date(df[date_col])
    sub = [c for c in df.columns if c != "code"]
    df = df.dropna(how="all", subset=sub).reset_index(drop=True)
    return df


def sql_type(s) -> str:
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_integer_dtype(s):
        return "INTEGER"
    if pd.api.types.is_float_dtype(s):
        return "REAL"
    return "TEXT"


class Writer:
    """单线程 SQLite 写入器, 幂等: 先删该 code 旧行再插入"""

    def __init__(self, db):
        self.con = sqlite3.connect(db, timeout=120)
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("PRAGMA synchronous=NORMAL")
        self.con.execute("PRAGMA temp_store=MEMORY")
        self.con.execute("""CREATE TABLE IF NOT EXISTS fund_progress(
            code TEXT NOT NULL, dataset TEXT NOT NULL, status TEXT NOT NULL,
            rows INTEGER, error TEXT, started REAL, finished REAL,
            PRIMARY KEY(code, dataset))""")
        self.cols = {}
        for table, date_col in DATASET_SPECS.values():
            self._ensure(table, pd.DataFrame(
                {"code": pd.Series(dtype=str), date_col: pd.Series(dtype=str)}))
        self.con.commit()

    def _ensure(self, table, df):
        if table not in self.cols:
            existing = {r[1].lower(): r[1] for r in self.con.execute(
                f'PRAGMA table_info("{table}")')}
            if not existing:
                cols = ", ".join(f'"{c}" {sql_type(df[c])}' for c in df.columns)
                self.con.execute(f'CREATE TABLE "{table}" ({cols})')
                self.con.execute(
                    f'CREATE INDEX "idx_{table}_code" ON "{table}"(code)')
                existing = {str(c).lower(): c for c in df.columns}
            self.cols[table] = existing
        for c in df.columns:
            if str(c).lower() not in self.cols[table]:
                self.con.execute(
                    f'ALTER TABLE "{table}" ADD COLUMN "{c}" {sql_type(df[c])}')
                self.cols[table][str(c).lower()] = c

    def write(self, code, dataset, df):
        table, date_col = DATASET_SPECS[dataset]
        df = normalize(code, df, date_col)
        self._ensure(table, df)
        rows = len(df)
        with self.con:
            self.con.execute(f'DELETE FROM "{table}" WHERE code=?', (code,))
            if rows:
                collist = ", ".join(f'"{c}"' for c in df.columns)
                ph = ", ".join("?" * len(df.columns))
                data = df.astype(object).where(pd.notna(df), None)
                self.con.executemany(
                    f'INSERT INTO "{table}" ({collist}) VALUES ({ph})',
                    list(data.itertuples(index=False, name=None)))
        return rows

    def mark(self, code, dataset, status, rows, error, started):
        with self.con:
            self.con.execute(
                "INSERT INTO fund_progress(code,dataset,status,rows,error,"
                "started,finished) VALUES(?,?,?,?,?,?,?) "
                "ON CONFLICT(code,dataset) DO UPDATE SET status=excluded.status,"
                " rows=excluded.rows, error=excluded.error,"
                " started=excluded.started, finished=excluded.finished",
                (code, dataset, status, rows, error, started, time.time()))


# ---------------------------------------------------------------- 抓取调度
def worker(code, dataset, retry, sleep):
    started = time.time()
    df, err = None, None
    for attempt in range(max(1, retry)):
        try:
            df = FETCHERS[dataset](code)
            if df is None:
                df = pd.DataFrame()
            if not df.empty:
                return code, dataset, df, None, started
            err = "empty"
        except Exception as e:
            df, err = None, f"{type(e).__name__}: {e}"
        time.sleep(min(60.0, 2.0 ** attempt + random.random()))
    if sleep > 0:
        time.sleep(sleep)
    if err == "empty":
        return code, dataset, pd.DataFrame(), None, started
    return code, dataset, None, err or "unknown error", started


def load_codes(con, args):
    codes = [r[0] for r in con.execute(
        f"SELECT code FROM stocks WHERE {CODE_COND} ORDER BY code")]
    if args.codes:
        want = {c.strip() for c in args.codes.split(",") if c.strip()}
        codes = [c for c in codes if c in want]
    if args.limit and len(codes) > args.limit:
        rs = np.random.RandomState(args.seed)
        codes = sorted(rs.choice(codes, size=args.limit, replace=False).tolist())
    return codes


def print_summary(con):
    df = pd.read_sql_query(
        "SELECT dataset, status, COUNT(*) AS tasks, SUM(rows) AS rows "
        "FROM fund_progress GROUP BY dataset, status "
        "ORDER BY dataset, status", con)
    print("\n================ 抓取进度汇总 ================")
    print(df.to_string(index=False))
    for table, _ in DATASET_SPECS.values():
        n = con.execute(
            f"SELECT COUNT(*) FROM sqlite_master WHERE type='table' "
            f"AND name=?", (table,)).fetchone()[0]
        if n:
            cnt = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            print(f"  {table}: {cnt:,} 行")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB)
    ap.add_argument("--datasets", default=",".join(DATASET_SPECS),
                    help="逗号分隔, 可选: " + ",".join(DATASET_SPECS))
    ap.add_argument("--codes", default="", help="逗号分隔代码, 如 sh600000,sz000001")
    ap.add_argument("--limit", type=int, default=0, help="随机抽取N只股票(冒烟)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--retry", type=int, default=3)
    ap.add_argument("--sleep", type=float, default=0.2,
                    help="每个任务完成后休眠秒数")
    ap.add_argument("--redo", action="store_true", help="忽略进度, 全部重抓")
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    for d in datasets:
        if d not in DATASET_SPECS:
            raise SystemExit(f"未知数据集 {d}, 可选: {', '.join(DATASET_SPECS)}")

    writer = Writer(args.db)
    con = writer.con
    codes = load_codes(con, args)
    if not codes:
        raise SystemExit("没有匹配的股票代码")

    done = set()
    if not args.redo:
        done = set(con.execute(
            "SELECT code, dataset FROM fund_progress WHERE status='ok'"))
    tasks = [(c, d) for c in codes for d in datasets if (c, d) not in done]
    log(f"股票 {len(codes)} 只, 数据集 {len(datasets)} 类, "
        f"待抓取 {len(tasks):,} 个任务, 已完成 {len(codes) * len(datasets) - len(tasks):,}")
    if not tasks:
        print_summary(con)
        return

    t0 = time.time()
    total = len(tasks)
    n_ok = n_fail = 0
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(worker, c, d, args.retry, args.sleep)
                    for c, d in tasks]
            for fut in as_completed(futs):
                code, ds, df, err, started = fut.result()
                if err:
                    writer.mark(code, ds, "fail", None, err, started)
                    n_fail += 1
                else:
                    rows = writer.write(code, ds, df)
                    writer.mark(code, ds, "ok", rows, None, started)
                    n_ok += 1
                k = n_ok + n_fail
                if k % args.log_every == 0 or k == total:
                    el = time.time() - t0
                    speed = k / el if el > 0 else 0
                    eta = (total - k) / speed / 3600 if speed > 0 else 0
                    log(f"进度 {k:,}/{total:,} (ok {n_ok:,}, fail {n_fail:,}), "
                        f"{speed:.2f} 任务/s, 已用 {el / 60:.1f} min, "
                        f"预计剩余 {eta:.1f} h")
    except KeyboardInterrupt:
        log("收到中断, 已写入数据保存在库中, 可直接重跑续传")
    print_summary(con)
    log(f"本次完成 {n_ok:,} 成功 / {n_fail:,} 失败, "
        f"总用时 {(time.time() - t0) / 60:.1f} min")
    log(f"数据库: {args.db}")


if __name__ == "__main__":
    main()
