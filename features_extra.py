#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""新爬数据的特征组: 资金流/两融/龙虎榜/业绩事件

防未来函数:
  - 资金流: 收盘后披露 -> 组内错后1个交易日
  - 两融: T+1披露 -> 错后1个交易日
  - 龙虎榜: 当日盘后披露 -> 只用严格早于 t 的上榜记录
  - 业绩预告/快报: 公告日 +1 天生效, 且只保留150个自然日内的事件
"""
import sqlite3

import numpy as np
import pandas as pd

from lgbm_train_backtest import DB, asof_align, log


def _read(sql):
    con = sqlite3.connect(DB)
    df = pd.read_sql_query(sql, con)
    con.close()
    return df


def _to_days(s):
    return (pd.to_datetime(s, errors="coerce")
            .values.astype("datetime64[D]").astype(np.int64))


def build_flow_features(dates, codes):
    log("构建资金流特征 ...")
    df = _read("SELECT code, date, main_pct, xl_pct, l_pct, s_pct "
               "FROM stock_fund_flow")
    if df.empty:
        return None, []
    for c in ("main_pct", "xl_pct", "l_pct", "s_pct"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    g = df.groupby("code", sort=False)
    F = pd.DataFrame({
        "fl_main": df["main_pct"],
        "fl_xl": df["xl_pct"],
        "fl_l": df["l_pct"],
        "fl_s": df["s_pct"],
        "fl_main5": g["main_pct"].transform(lambda s: s.rolling(5).mean()),
        "fl_main20": g["main_pct"].transform(lambda s: s.rolling(20).mean()),
        "fl_xl5": g["xl_pct"].transform(lambda s: s.rolling(5).mean()),
    }).astype(np.float32)
    F = F.groupby(df["code"], sort=False).shift(1)
    right = pd.concat([df[["code", "date"]], F], axis=1)
    names = list(F.columns)
    return asof_align(codes, dates, right, "date", names), names


def build_margin_features(dates, codes):
    log("构建两融特征 ...")
    df = _read("SELECT code, date, rz_bal, rz_buy, rz_repay, rq_bal "
               "FROM margin_detail")
    if df.empty:
        return None, []
    for c in ("rz_bal", "rz_buy", "rz_repay", "rq_bal"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    gc = df["code"]
    rz, rq = df["rz_bal"], df["rq_bal"]
    F = pd.DataFrame({
        "mg_rz_chg1": rz / rz.groupby(gc, sort=False).shift(1) - 1,
        "mg_rz_chg5": rz / rz.groupby(gc, sort=False).shift(5) - 1,
        "mg_rz_chg20": rz / rz.groupby(gc, sort=False).shift(20) - 1,
        "mg_buy_ratio": df["rz_buy"] / (df["rz_buy"] + df["rz_repay"] + 1e-9),
        "mg_rq_chg5": rq / rq.groupby(gc, sort=False).shift(5) - 1,
        "mg_log_rz": np.log(rz.clip(lower=1.0)),
    }).astype(np.float32)
    F = F.groupby(gc, sort=False).shift(1)
    right = pd.concat([df[["code", "date"]], F], axis=1)
    names = list(F.columns)
    return asof_align(codes, dates, right, "date", names), names


def build_lhb_features(dates, codes):
    log("构建龙虎榜特征 ...")
    df = _read('SELECT code, "上榜日" AS date, "龙虎榜净买额" AS net, '
               '"换手率" AS turn, "净买额占总成交比" AS ratio, '
               '"上榜原因" AS reason FROM lhb_detail')
    if df.empty:
        return None, []
    df = df.dropna(subset=["date"]).copy()
    df["net"] = pd.to_numeric(df["net"], errors="coerce").fillna(0.0) / 1e8
    df["turn"] = pd.to_numeric(df["turn"], errors="coerce").fillna(0.0)
    df["ratio"] = pd.to_numeric(df["ratio"], errors="coerce").fillna(0.0)
    r = df["reason"].fillna("")
    df["f_zdf"] = r.str.contains("涨幅偏离").astype(np.float32)
    df["f_ddf"] = r.str.contains("跌幅偏离").astype(np.float32)
    df["f_lxup"] = (r.str.contains("连续") & r.str.contains("涨")).astype(
        np.float32)
    df["f_lxdn"] = (r.str.contains("连续") & r.str.contains("跌")).astype(
        np.float32)
    df["f_hsl"] = r.str.contains("换手率").astype(np.float32)
    df["f_nolim"] = r.str.contains("无价格涨跌幅").astype(np.float32)
    vcols = ["net", "turn", "ratio", "f_zdf", "f_ddf", "f_lxup", "f_lxdn",
             "f_hsl", "f_nolim"]
    agg = df.groupby(["code", "date"], as_index=False).agg(
        **{c: (c, "sum") for c in vcols}, cnt=("net", "size"))
    agg["day"] = _to_days(agg["date"])
    ld = _to_days(dates)
    pos = pd.Series(np.arange(len(dates))).groupby(np.asarray(codes)).indices
    names = ["lhb_cnt28", "lhb_net28", "lhb_turn28", "lhb_ratio28",
             "lhb_zdf28", "lhb_ddf28", "lhb_lxup28", "lhb_lxdn28",
             "lhb_hsl28", "lhb_nolim28", "lhb_cnt84", "lhb_net84"]
    n = len(names)
    out = np.zeros((len(dates), n), dtype=np.float32)
    sums = {c: np.cumsum(agg[c].to_numpy(dtype=np.float64)) for c in vcols}
    sums["__cnt"] = np.cumsum(agg["cnt"].to_numpy(dtype=np.float64))

    def seg(cum, j_hi, j_lo):
        return cum[j_hi - 1] - np.where(j_lo > 0, cum[j_lo - 1], 0.0)

    for code, g in agg.groupby("code", sort=False):
        rows = pos.get(code)
        if rows is None:
            continue
        d = g["day"].to_numpy()
        gi = g.index.to_numpy()
        t = ld[rows]
        j_hi = np.searchsorted(d, t, side="left")
        j_lo = np.searchsorted(d, t - 28, side="left")
        ok = j_hi > 0
        cnt = seg(sums["__cnt"], j_hi, j_lo)
        net = seg(sums["net"], j_hi, j_lo)
        turn = seg(sums["turn"], j_hi, j_lo)
        ratio = seg(sums["ratio"], j_hi, j_lo)
        denom = np.maximum(cnt, 1.0)
        vals = {
            "lhb_cnt28": cnt,
            "lhb_net28": net,
            "lhb_turn28": turn / denom,
            "lhb_ratio28": ratio / denom,
            "lhb_zdf28": seg(sums["f_zdf"], j_hi, j_lo),
            "lhb_ddf28": seg(sums["f_ddf"], j_hi, j_lo),
            "lhb_lxup28": seg(sums["f_lxup"], j_hi, j_lo),
            "lhb_lxdn28": seg(sums["f_lxdn"], j_hi, j_lo),
            "lhb_hsl28": seg(sums["f_hsl"], j_hi, j_lo),
            "lhb_nolim28": seg(sums["f_nolim"], j_hi, j_lo),
        }
        j_lo84 = np.searchsorted(d, t - 84, side="left")
        vals["lhb_cnt84"] = seg(sums["__cnt"], j_hi, j_lo84)
        vals["lhb_net84"] = seg(sums["net"], j_hi, j_lo84)
        for k, name in enumerate(names):
            v = np.where(ok, vals[name], 0.0)
            out[rows, k] = v.astype(np.float32)
    return out, names


def _type_score(t):
    t = str(t)
    for kw in ("预增", "略增", "扭亏", "续盈", "减亏"):
        if kw in t:
            return 1.0
    for kw in ("预减", "略减", "首亏", "续亏", "增亏"):
        if kw in t:
            return -1.0
    return 0.0


def build_event_features(dates, codes):
    log("构建业绩事件特征 ...")
    ld = _to_days(dates)
    yj = _read('SELECT code, "公告日期" AS notice, "预告类型" AS ttype, '
               '"业绩变动幅度" AS amp FROM yjyg')
    kb = _read('SELECT code, "公告日期" AS notice, "净利润-同比增长" AS np_yoy, '
               '"每股收益" AS eps FROM yjkb')
    mats, names = [], []

    if not yj.empty:
        yj = yj.dropna(subset=["notice"]).copy()
        yj["yj_type"] = yj["ttype"].map(_type_score)
        yj["yj_amp"] = pd.to_numeric(
            yj["amp"], errors="coerce").clip(-100, 100) / 100.0
        day = _to_days(yj["notice"]) + 1
        yj["date"] = pd.to_datetime(day, unit="D").strftime("%Y-%m-%d")
        yj = yj.dropna(subset=["date"])
        yj = yj.groupby(["code", "date"], as_index=False).agg(
            yj_type=("yj_type", "mean"), yj_amp=("yj_amp", "mean"))
        yj["day"] = _to_days(yj["date"])
        vals = ["yj_type", "yj_amp", "day"]
        mat = asof_align(codes, dates, yj[["code", "date"] + vals],
                         "date", vals)
        fresh = (ld - mat[:, 2]) <= 150
        mat[~fresh, 0] = np.nan
        mat[~fresh, 1] = np.nan
        mats.append(mat[:, :2])
        names += ["yj_type", "yj_amp"]

    if not kb.empty:
        kb = kb.dropna(subset=["notice"]).copy()
        kb["np_yoy"] = pd.to_numeric(kb["np_yoy"], errors="coerce").clip(
            -500, 500)
        kb["eps"] = pd.to_numeric(kb["eps"], errors="coerce")
        day = _to_days(kb["notice"]) + 1
        kb["date"] = pd.to_datetime(day, unit="D").strftime("%Y-%m-%d")
        kb = kb.dropna(subset=["date"])
        kb = kb.drop_duplicates(["code", "date"], keep="last")
        kb["day"] = _to_days(kb["date"])
        vals = ["np_yoy", "eps", "day"]
        mat = asof_align(codes, dates, kb[["code", "date"] + vals],
                         "date", vals)
        fresh = (ld - mat[:, 2]) <= 150
        mat[~fresh, 0] = np.nan
        mat[~fresh, 1] = np.nan
        mats.append(mat[:, :2])
        names += ["kb_np_yoy", "kb_eps"]

    if not mats:
        return None, []
    return np.hstack(mats), names
