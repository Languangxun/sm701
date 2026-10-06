#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据质量审计与清洗 (stock_cache.db)

审计对象:
  daily_bars           日K: 重复/非法日期/非正价/OHLC结构
  index_bars           大盘指数日线
  industry_bars        申万一级行业指数日线
  stock_industry       个股->行业映射: 孤儿代码/行业不在行业表
  fund_valuation       每日估值: 重复/非法日期/非正价/PE/PB为零
  fund_valuation_baidu 百度估值: 重复/非正值
  fund_abstract        财务摘要
  fund_indicator       新浪财务指标
  fund_indicator_em    东财财务指标: 公告日早于报告期=脏数据
  fund_balance         资产负债表
  fund_profit          利润表
  fund_cashflow        现金流量表
  fund_fhps            分红送配
  fund_gdhs            股东户数
  fund_progress        抓取状态(仅统计 ok/fail)

模式:
  默认     只扫描并输出报告(不改库)
  --fix    删除确定脏数据: 重复行/非法日期/非正价/全空行/公告早于报告期
  --vacuum --fix 后回收空间
用法:
    python3 clean_data.py
    python3 clean_data.py --fix
    python3 clean_data.py --tables fund_valuation,daily_bars
    python3 clean_data.py --fix --vacuum
"""
import argparse
import os
import sqlite3
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "stock_cache.db")
REPORT_DIR = os.path.join(HERE, "reports")

# 表 -> (日期列, 主键列, 必须>0的列, 是否OHLC结构)
SPECS = {
    "daily_bars":           ("date", ["code", "date"],
                             ["open", "high", "low", "close"], True),
    "index_bars":           ("date", ["code", "date"],
                             ["open", "high", "low", "close"], True),
    "industry_bars":        ("date", ["industry_code", "date"],
                             ["open", "high", "low", "close"], True),
    "stock_industry":       (None, ["code"], [], False),
    "fund_valuation":       ("date", ["code", "date"], ["close"], False),
    # 百度估值的 PE/PB/PCF 为负是亏损股正常值, 不做非正检查
    "fund_valuation_baidu": ("date", ["code", "indicator", "date"], [], False),
    "fund_abstract":        ("report_date", ["code", "report_date"], [], False),
    "fund_indicator":       ("report_date", ["code", "report_date"], [], False),
    "fund_indicator_em":    ("report_date", ["code", "report_date"], [], False),
    "fund_balance":         ("report_date", ["code", "report_date"], [], False),
    "fund_profit":          ("report_date", ["code", "report_date"], [], False),
    "fund_cashflow":        ("report_date", ["code", "report_date"], [], False),
    "fund_fhps":            ("report_date", ["code", "report_date"], [], False),
    "fund_gdhs":            ("report_date", ["code", "report_date"], [], False),
}

NOTICE_TABLES = ["fund_indicator_em", "fund_balance", "fund_profit",
                 "fund_cashflow"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def table_exists(con, t):
    return con.execute("SELECT COUNT(*) FROM sqlite_master WHERE "
                       "type='table' AND name=?", (t,)).fetchone()[0] > 0


def columns_of(con, t):
    return [r[1] for r in con.execute(f'PRAGMA table_info("{t}")')]


def sample_rows(con, t, where, n=3):
    try:
        rows = con.execute(
            f'SELECT * FROM "{t}" WHERE {where} LIMIT {n}').fetchall()
        return [[str(v)[:40] for v in r] for r in rows]
    except sqlite3.Error:
        return []


def audit_table(con, t, spec):
    date_col, keys, pos_cols, ohlc = spec
    cols = columns_of(con, t)
    info = {"table": t, "rows": 0}
    n = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
    info["rows"] = n
    if n == 0:
        return info, [("空表", 0, [])]
    checks = []

    if date_col in cols:
        bad_d = con.execute(
            f'SELECT COUNT(*) FROM "{t}" WHERE "{date_col}" IS NULL '
            f"OR date(\"{date_col}\") IS NULL").fetchone()[0]
        if bad_d:
            checks.append(("非法/空日期", bad_d, sample_rows(
                con, t, f'"{date_col}" IS NULL OR date("{date_col}") IS NULL')))
        dup = con.execute(
            f'SELECT COUNT(*) FROM (SELECT 1 FROM "{t}" GROUP BY '
            f'{", ".join(chr(34)+k+chr(34) for k in keys)} HAVING COUNT(*)>1)'
        ).fetchone()[0]
        if dup:
            checks.append(("重复主键", dup, []))
    for c in pos_cols:
        if c not in cols:
            continue
        bad = con.execute(
            f'SELECT COUNT(*) FROM "{t}" WHERE "{c}" IS NOT NULL '
            f'AND "{c}" <= 0').fetchone()[0]
        if bad:
            checks.append((f"非正{c}", bad, sample_rows(
                con, t, f'"{c}" IS NOT NULL AND "{c}" <= 0')))
    if ohlc and all(c in cols for c in ("open", "high", "low", "close")):
        bad = con.execute(
            f'SELECT COUNT(*) FROM "{t}" WHERE high < low '
            f'OR high < MAX(open, close) OR low > MIN(open, close)'
        ).fetchone()[0]
        if bad:
            checks.append(("OHLC结构错", bad, sample_rows(
                con, t, "high < low OR high < MAX(open, close) "
                "OR low > MIN(open, close)")))
    data_cols = [c for c in cols if c not in keys]
    if 0 < len(data_cols) <= 40:
        where = " AND ".join(f'"{c}" IS NULL' for c in data_cols)
        allnull = con.execute(
            f'SELECT COUNT(*) FROM "{t}" WHERE {where}').fetchone()[0]
        if allnull:
            checks.append(("全空行", allnull, []))
    if t in NOTICE_TABLES and "NOTICE_DATE" in cols:
        bad = con.execute(
            f'SELECT COUNT(*) FROM "{t}" WHERE "NOTICE_DATE" IS NOT NULL '
            f'AND "{date_col}" IS NOT NULL '
            f'AND date("NOTICE_DATE") < date("{date_col}")').fetchone()[0]
        if bad:
            checks.append(("公告日早于报告期", bad, sample_rows(
                con, t, 'date("NOTICE_DATE") < date("' + date_col + '")')))
    if t in ("fund_valuation",):
        for c in ("pe_ttm", "pb"):
            if c in cols:
                z = con.execute(
                    f'SELECT COUNT(*) FROM "{t}" WHERE "{c}" = 0').fetchone()[0]
                if z:
                    checks.append((f"{c}=0(接口缺失占位)", z, []))
    if t.startswith("fund_") and "code" in cols:
        orph = con.execute(
            f'SELECT COUNT(*) FROM "{t}" x LEFT JOIN stocks s '
            f'ON s.code=x.code WHERE s.code IS NULL').fetchone()[0]
        if orph:
            checks.append(("代码不在stocks表", orph, []))
    if t == "stock_industry":
        orph = con.execute(
            "SELECT COUNT(*) FROM stock_industry x LEFT JOIN stocks s "
            "ON s.code=x.code WHERE s.code IS NULL").fetchone()[0]
        if orph:
            checks.append(("代码不在stocks表", orph, []))
        bad_ind = con.execute(
            "SELECT COUNT(*) FROM (SELECT DISTINCT industry_code "
            "FROM stock_industry) x LEFT JOIN (SELECT DISTINCT industry_code "
            "FROM industry_bars) b ON b.industry_code=x.industry_code "
            "WHERE b.industry_code IS NULL").fetchone()[0]
        if bad_ind:
            checks.append(("行业代码无行情", bad_ind, []))
        null_ind = con.execute(
            "SELECT COUNT(*) FROM stock_industry WHERE industry_code IS NULL "
            "OR industry_code=''").fetchone()[0]
        if null_ind:
            checks.append(("行业为空", null_ind, []))

    return info, checks


def fix_table(con, t, spec):
    date_col, keys, pos_cols, ohlc = spec
    cols = columns_of(con, t)
    fixed = {}
    with con:
        if date_col in cols:
            cur = con.execute(
                f'DELETE FROM "{t}" WHERE "{date_col}" IS NULL '
                f'OR date("{date_col}") IS NULL')
            if cur.rowcount:
                fixed["非法/空日期"] = cur.rowcount
        dup_sql = (f'DELETE FROM "{t}" WHERE rowid NOT IN '
                   f'(SELECT MAX(rowid) FROM "{t}" GROUP BY '
                   f'{", ".join(chr(34)+k+chr(34) for k in keys)})')
        cur = con.execute(dup_sql)
        if cur.rowcount:
            fixed["重复主键"] = cur.rowcount
        for c in pos_cols:
            if c in cols:
                cur = con.execute(
                    f'DELETE FROM "{t}" WHERE "{c}" IS NOT NULL AND "{c}"<=0')
                if cur.rowcount:
                    fixed[f"非正{c}"] = cur.rowcount
        if ohlc and all(c in cols for c in ("open", "high", "low", "close")):
            # 最小修复: high/low 撑到包含 open/close, 不删K线
            cur = con.execute(
                f'UPDATE "{t}" SET high = MAX(high, open, close) '
                f'WHERE high IS NOT NULL AND open IS NOT NULL '
                f'AND close IS NOT NULL '
                f'AND (high < MAX(open, close) OR high < low)')
            n1 = cur.rowcount
            cur = con.execute(
                f'UPDATE "{t}" SET low = MIN(low, open, close) '
                f'WHERE low IS NOT NULL AND open IS NOT NULL '
                f'AND close IS NOT NULL '
                f'AND (low > MIN(open, close) OR low > high)')
            if n1 or cur.rowcount:
                fixed["OHLC结构错"] = n1 + cur.rowcount
        data_cols = [c for c in cols if c not in keys]
        if 0 < len(data_cols) <= 40:
            where = " AND ".join(f'"{c}" IS NULL' for c in data_cols)
            cur = con.execute(f'DELETE FROM "{t}" WHERE {where}')
            if cur.rowcount:
                fixed["全空行"] = cur.rowcount
        if t in NOTICE_TABLES and "NOTICE_DATE" in cols:
            # 公告日早于报告期=占位符(1900-01-01)或脏数据: 置空, 训练按缺失处理
            cur = con.execute(
                f'UPDATE "{t}" SET "NOTICE_DATE"=NULL '
                f'WHERE "NOTICE_DATE" IS NOT NULL '
                f'AND "{date_col}" IS NOT NULL '
                f'AND date("NOTICE_DATE") < date("{date_col}")')
            if cur.rowcount:
                fixed["公告日早于报告期"] = cur.rowcount
    return fixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB)
    ap.add_argument("--tables", default="", help="逗号分隔, 默认全部")
    ap.add_argument("--fix", action="store_true", help="执行清理(默认只扫描)")
    ap.add_argument("--vacuum", action="store_true", help="清理后VACUUM")
    args = ap.parse_args()

    con = sqlite3.connect(args.db, timeout=120)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=60000")
    tables = ([t.strip() for t in args.tables.split(",") if t.strip()]
              or list(SPECS))
    body = [f"# 数据清洗报告 {time.strftime('%Y-%m-%d %H:%M')}",
            f"库: `{args.db}`  模式: **{'修复' if args.fix else '只扫描'}**",
            "", "| 表 | 行数 | 问题 | 数量 | 修复 |", "|---|---|---|---|---|"]
    t0 = time.time()
    total_issues = 0

    for t in tables:
        if t == "fund_progress":
            if table_exists(con, t):
                body.append("")
                body.append("## fund_progress (抓取状态, 仅统计)")
                body.append("| dataset | status | tasks | rows |")
                body.append("|---|---|---|---|")
                for r in con.execute(
                        "SELECT dataset, status, COUNT(*), SUM(rows) "
                        "FROM fund_progress GROUP BY dataset, status"):
                    body.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3] or 0} |")
            continue
        if t not in SPECS or not table_exists(con, t):
            log(f"跳过 {t} (不存在)")
            continue
        info, checks = audit_table(con, t, SPECS[t])
        n_issue = sum(c for _, c, _ in checks)
        total_issues += n_issue
        fixed = {}
        if args.fix and n_issue:
            fixed = fix_table(con, t, SPECS[t])
            log(f"{t}: {info['rows']:,} 行, 问题 {n_issue}, 已修复 {fixed}")
        else:
            log(f"{t}: {info['rows']:,} 行, 问题 {n_issue}")
        for name, cnt, s in (checks or [("正常", 0, [])]):
            body.append(f"| {t} | {info['rows']:,} | {name} | {cnt:,} | "
                        f"{fixed.get(name, '')} |")
            for x in s:
                body.append(f"|  |  | 样例: {' '.join(x)} |  |  |")
        body.append("")

    if args.fix and args.vacuum:
        log("VACUUM ...")
        con.execute("VACUUM")
    con.close()

    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR,
                        f"清洗报告_{time.strftime('%Y%m%d')}.md")
    body.append(f"扫描用时 {time.time() - t0:.0f}s")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(body) + "\n")
    log(f"完成, 问题合计 {total_issues}, 报告: {path}")


if __name__ == "__main__":
    main()
