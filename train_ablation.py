#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LightGBM 特征组消融测试 (Ablation Study)

特征组:
  tech      个股技术指标 (lgbm_train_backtest 原有价量指标, 不含市场项)
  pattern   个股 K 线形态 (实体/影线/十字星/锤子/吞没/连阳/涨停/新高低等)
  market    大盘数据 (上证/深成/创业板/沪深300 指数收益、均线、波动 + 市场宽度)
  industry  行业数据 (申万一级行业指数收益、动量、相对强弱、行业排名)
  fund      基本面 (每日 PE/PB/PS/PCF/市值 + 东财报告期指标, 按公告日 point-in-time 对齐)

消融方式 (--mode):
  full     单组 + 逐组累加 + 留一法 (默认, 15 组配置)
  single   每个特征组单独训练
  cum      逐组累加
  loo      全特征留一

防未来函数:
  - 基本面按 NOTICE_DATE (公告日) asof 对齐, 只用 t 日及以前已披露的报告
  - 估值按日期直接对齐, 缺失向前填充
  - 指数/行业/技术指标仅用 t 日收盘及以前数据

全量内存方案:
  - 默认使用全部样本(不采样); 特征按组落盘到 ablation_output/feat_*.npy
  - 训练时按配置逐组读入内存, 峰值内存 ≈ 单配置矩阵, 15G 内存也可跑全量
  - 训练集 AUC 默认在 50 万行子样本上计算, 验证集用全量

用法:
    python3 train_ablation.py --limit 300 --rounds 300      # 冒烟
    python3 train_ablation.py --mode full                    # 全量消融
    python3 train_ablation.py --mode single --groups tech,pattern,fund
    python3 train_ablation.py --mode horizon --groups tech,pattern,market,industry,fund,flow,margin,lhb,events
        # T+1/T+5/T+10 三个模型, IC=逐日RankIC对比T日实际收益
    python3 train_ablation.py --mode walkforward --groups tech,pattern,industry,fund
        # walk-forward多窗口(2022/2023验证)+2024+holdout定稿评估
    python3 train_ablation.py --mode horizon --calibrate platt --regime breadth --max-dd 0.15
        # Platt校准(2018-2022训练/2023校准/2024+评估)+状态过滤+回撤控制
    python3 train_ablation.py --mode horizon --exec topen --groups tech
        # 盘中信号锚定开盘价: 信号日=买入日/open[t]执行 (仅horizon/walkforward)
    python3 train_ablation.py --train-sample 3000000         # 手动采样加速
"""
import argparse
import gc
import json
import os
import sqlite3
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from lgbm_train_backtest import (COST, DATA_START, MIN_HOLD, SEED, TRAIN_END,
                                 _aligned, ThermalCallback, asof_align,
                                 build_features, daily_ic, load_bars, log,
                                 neutralize_features, perf, run_period)
from features_extra import (build_event_features, build_flow_features,
                            build_lhb_features, build_margin_features)

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "stock_cache.db")
OUT_DIR = os.path.join(HERE, "ablation_output")

GROUPS = ["tech", "pattern", "market", "industry", "fund",
          "flow", "margin", "lhb", "events", "tech_norm", "pattern_norm"]
# market 组是每日常数特征(同一天所有股票值相同), 混入横截面选股模型会占据
# 早期分裂、日内预测趋同、TopK退化——默认不进选股, 仅择时研究时手动指定
DEFAULT_GROUPS = "tech,pattern,industry,fund"
MKT_COLS = ["mkt1", "mkt5", "mkt20", "breadth5"]

MARKET_INDICES = ["sh000001", "sz399001", "sz399006", "sh000300"]
REL_INDEX = "sh000300"

FUND_VAL_COLS = ["pe_ttm", "pb", "ps", "pcf", "peg", "total_mv", "float_mv"]
FUND_IND_COLS = [
    "ROEJQ", "ROEKCJQ", "XSMLL", "XSJLL", "ZZCJLL", "TOTALOPERATEREVETZ",
    "PARENTNETPROFITTZ", "KCFJCXSYJLRTZ", "EPSJB", "BPS", "MGJYXJJE",
    "ZCFZL", "LD", "SD", "XJLLB", "JYXJLYYSR", "YSZKYYSR", "ZZCZZTS",
    "CHZZTS", "YSZKZZTS", "ROIC", "TOTAL_ROI",
]
FUND_IND_RENAME = {
    "ROEJQ": "roe", "ROEKCJQ": "roe_kcfj", "XSMLL": "gross_margin",
    "XSJLL": "net_margin", "ZZCJLL": "roa", "TOTALOPERATEREVETZ": "rev_yoy",
    "PARENTNETPROFITTZ": "np_yoy", "KCFJCXSYJLRTZ": "np_kf_yoy",
    "EPSJB": "eps", "BPS": "bps", "MGJYXJJE": "ocf_ps", "ZCFZL": "debt_ratio",
    "LD": "current_ratio", "SD": "quick_ratio", "XJLLB": "ocf_ratio",
    "JYXJLYYSR": "ocf_to_rev", "YSZKYYSR": "ar_to_rev", "ZZCZZTS": "asset_turn_days",
    "CHZZTS": "inv_turn_days", "YSZKZZTS": "ar_turn_days", "ROIC": "roic",
    "TOTAL_ROI": "total_roi",
}
INDUSTRY_FEATS = ["ind_ret1", "ind_ret5", "ind_ret20", "ind_ma20r", "ind_vol20",
                  "ind_volr", "ind_amt_r", "ind_ret5_rel", "ind_ret20_rel",
                  "ind_rank20"]


def to_matrix(X: pd.DataFrame, cols=None) -> np.ndarray:
    cols = list(X.columns) if cols is None else list(cols)
    vals = np.empty((len(X), len(cols)), dtype=np.float32)
    for j, c in enumerate(cols):
        vals[:, j] = X[c].to_numpy(dtype=np.float32, copy=False)
    gc.collect()
    return vals


def reindex_dates(mat, mat_dates, target_dates):
    idx = pd.Index(mat_dates).get_indexer(target_dates)
    out = np.full((len(target_dates), mat.shape[1]), np.nan, dtype=np.float32)
    ok = idx >= 0
    out[ok] = mat[idx[ok]]
    return out


# ---------------------------------------------------------------- K线形态
def build_pattern_features(df):
    log("构建K线形态特征 ...")
    gb = df.groupby("code", sort=False)
    codes = df["code"].cat.codes
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    pc = gb["close"].shift(1)
    po = gb["open"].shift(1)
    rng = (h - l).replace(0.0, np.nan)
    body = c - o
    body_abs = body.abs()
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    F = {}
    F["p_body"] = body / (pc + 1e-9)
    F["p_upper_sh"] = upper / (pc + 1e-9)
    F["p_lower_sh"] = lower / (pc + 1e-9)
    F["p_body_pos"] = body / (rng + 1e-9)
    F["p_doji"] = (body_abs <= 0.1 * rng).astype(np.float32)
    F["p_hammer"] = ((lower >= 2.0 * body_abs)
                     & (upper <= body_abs)).astype(np.float32)
    F["p_shooting"] = ((upper >= 2.0 * body_abs)
                       & (lower <= body_abs)).astype(np.float32)
    F["p_bull_engulf"] = ((c > o) & (pc < po) & (c >= po)
                          & (o <= pc)).astype(np.float32)
    F["p_bear_engulf"] = ((c < o) & (pc > po) & (c <= po)
                          & (o >= pc)).astype(np.float32)
    up = c > o
    dn = c < o
    F["p_three_up"] = (_aligned(up.astype(np.float32).groupby(codes)
                                .rolling(3).sum()) >= 3).astype(np.float32)
    F["p_three_dn"] = (_aligned(dn.astype(np.float32).groupby(codes)
                                .rolling(3).sum()) >= 3).astype(np.float32)

    def _streak(s):
        return s.groupby((s != s.shift()).cumsum()).cumsum().astype(np.float32)

    F["p_up_streak"] = up.groupby(codes, sort=False).transform(_streak)
    F["p_dn_streak"] = dn.groupby(codes, sort=False).transform(_streak)
    ret1 = c / (pc + 1e-9) - 1.0
    lim = df["lim"].astype(np.float64)
    is_limit_up = (ret1 >= lim - 0.005).astype(np.float32)
    is_limit_dn = (ret1 <= -lim + 0.005).astype(np.float32)
    F["p_limit_up20"] = _aligned(is_limit_up.groupby(codes)
                                 .rolling(20).sum())
    F["p_limit_dn20"] = _aligned(is_limit_dn.groupby(codes)
                                 .rolling(20).sum())
    F["p_yang_ratio20"] = _aligned(up.astype(np.float32).groupby(codes)
                                   .rolling(20).mean())
    gap = o / (pc + 1e-9) - 1.0
    F["p_gap_up20"] = _aligned((gap > 0.02).astype(np.float32)
                               .groupby(codes).rolling(20).sum())
    hh = _aligned(gb["close"].rolling(20).max()).groupby(
        codes, sort=False).shift(1)
    ll = _aligned(gb["close"].rolling(20).min()).groupby(
        codes, sort=False).shift(1)
    F["p_dist_high20"] = c / (hh + 1e-9) - 1.0
    F["p_dist_low20"] = c / (ll + 1e-9) - 1.0
    out = pd.DataFrame(index=df.index, dtype=np.float32)
    for k, v in F.items():
        out[k] = v.astype(np.float32)
    del F, gb
    gc.collect()
    return out


# ---------------------------------------------------------------- 大盘
def build_market_features(dates):
    log("构建大盘特征 ...")
    con = sqlite3.connect(DB)
    frames = []
    for code in MARKET_INDICES:
        d = pd.read_sql_query(
            "SELECT date, close FROM index_bars WHERE code=? ORDER BY date",
            con, params=[code])
        close = d["close"].astype(float)
        r1 = close.pct_change()
        f = pd.DataFrame({
            f"m_{code}_ret1": r1,
            f"m_{code}_ret5": close.pct_change(5),
            f"m_{code}_ret20": close.pct_change(20),
            f"m_{code}_ret60": close.pct_change(60),
            f"m_{code}_ma20r": close / close.rolling(20).mean() - 1.0,
            f"m_{code}_vol20": r1.rolling(20).std(),
        })
        f.index = d["date"].values
        frames.append(f)
    con.close()
    mkt = pd.concat(frames, axis=1)
    mkt = mkt.loc[:, ~mkt.columns.duplicated()]
    return mkt, mkt.columns.tolist()


# ---------------------------------------------------------------- 行业
def build_industry_features(dates, codes):
    log("构建行业特征 ...")
    con = sqlite3.connect(DB)
    bars = pd.read_sql_query(
        "SELECT industry_code, date, close, amount FROM industry_bars "
        "ORDER BY industry_code, date", con)
    # 行业映射用当前快照(而非按 start_date 严格asof): start_date 是指数纳入日
    # 而非行业变更日, 严格对齐会把2018-2021年大部分股票的行业特征清空;
    # 残余偏差是成分股历史变更, 相对可接受
    mp = pd.read_sql_query("SELECT code, industry_code FROM stock_industry", con)
    con.close()
    gb = bars.groupby("industry_code", sort=False)
    close = bars["close"].astype(float)
    r1 = gb["close"].pct_change()
    bars["ind_ret1"] = r1.astype(np.float32)
    bars["ind_ret5"] = close.groupby(bars["industry_code"]).pct_change(5).astype(np.float32)
    bars["ind_ret20"] = close.groupby(bars["industry_code"]).pct_change(20).astype(np.float32)
    ma20 = gb["close"].transform(lambda s: s.rolling(20).mean())
    bars["ind_ma20r"] = (close / ma20 - 1.0).astype(np.float32)
    bars["ind_vol20"] = gb["close"].transform(
        lambda s: s.pct_change().rolling(20).std()).astype(np.float32)
    bars["ind_volr"] = (
        gb["close"].transform(lambda s: s.pct_change().rolling(20).std())
        / (gb["close"].transform(lambda s: s.pct_change().rolling(60).std()) + 1e-9)
    ).astype(np.float32)
    amt = bars["amount"].astype(float)
    bars["ind_amt_r"] = (amt / gb["amount"].transform(
        lambda s: s.astype(float).rolling(20).mean()) - 1.0).astype(np.float32)
    mkt_ret5 = bars.groupby("date")["ind_ret5"].transform("mean")
    mkt_ret20 = bars.groupby("date")["ind_ret20"].transform("mean")
    bars["ind_ret5_rel"] = (bars["ind_ret5"] - mkt_ret5).astype(np.float32)
    bars["ind_ret20_rel"] = (bars["ind_ret20"] - mkt_ret20).astype(np.float32)
    bars["ind_rank20"] = bars.groupby("date")["ind_ret20"].rank(pct=True).astype(np.float32)
    feat = bars.set_index(["industry_code", "date"])[INDUSTRY_FEATS]

    ind_of = pd.Series(codes).map(dict(zip(mp["code"], mp["industry_code"])))
    mi = pd.MultiIndex.from_arrays([ind_of.values, dates])
    arr = feat.reindex(mi)
    del bars, feat, gb
    gc.collect()
    return arr


# ---------------------------------------------------------------- 基本面
def build_fund_features(dates, codes):
    log("构建基本面特征 ...")
    con = sqlite3.connect(DB)

    def has(table):
        return con.execute("SELECT COUNT(*) FROM sqlite_master WHERE "
                           "type='table' AND name=?", (table,)).fetchone()[0] > 0

    names, mats = [], []
    if has("fund_valuation"):
        val = pd.read_sql_query(
            "SELECT code, date, " + ", ".join(FUND_VAL_COLS) +
            " FROM fund_valuation", con)
        vnames = ["ep", "bp", "sp", "cfp", "peg", "log_mv", "float_ratio"]
        raw = asof_align(codes, dates, val, "date", FUND_VAL_COLS)
        v = np.full_like(raw, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            for j in range(4):
                np.divide(1.0, raw[:, j], out=v[:, j],
                          where=raw[:, j] != 0)
            v[:, 4] = raw[:, 4]
            mv = np.where(raw[:, 5] > 0, raw[:, 5], np.nan)
            v[:, 5] = np.log(mv)
            v[:, 6] = np.where(mv > 0, raw[:, 6] / mv, np.nan)
        lo = np.array([-1.0, 0.0, -2.0, -2.0, -10.0, 0.0, 0.0])
        hi = np.array([1.0, 5.0, 2.0, 2.0, 10.0, 30.0, 1.0])
        v = np.clip(v, lo, hi)
        names += vnames
        mats.append(v)
        del val, raw
        gc.collect()
    else:
        log("警告: fund_valuation 不存在, 估值特征跳过")

    if has("fund_indicator_em"):
        cols = ", ".join(f'"{c}"' for c in FUND_IND_COLS)
        ind = pd.read_sql_query(
            f"SELECT code, NOTICE_DATE AS date, {cols} FROM fund_indicator_em "
            f"WHERE NOTICE_DATE IS NOT NULL", con)
        # 公告须严格早于t才可用(多为盘后披露, 同日公告在t收盘时未知)
        ind["date"] = (pd.to_datetime(ind["date"], errors="coerce")
                       - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
        ind = ind.dropna(subset=["date"])
        arr = asof_align(codes, dates, ind, "date", FUND_IND_COLS)
        # 报告期指标多为百分比/倍数, 直接使用, 极端值截断(NaN 保留给LGBM)
        arr = np.clip(arr, -1e3, 1e3)
        names += [FUND_IND_RENAME.get(c, c) for c in FUND_IND_COLS]
        mats.append(arr)
        del ind
        gc.collect()
    else:
        log("警告: fund_indicator_em 不存在, 财务指标特征跳过")

    con.close()
    if not mats:
        raise SystemExit("基本面数据表不存在, 请先运行 fetch_fundamentals.py")
    return mats, names


# ------------------------------------------------- 基本面横截面标准化/行业中性化
def load_industry_map():
    """个股->申万一级行业映射(当前快照, 与行业特征同一口径, 残余偏差见README §5.4)"""
    con = sqlite3.connect(DB)
    try:
        mp = pd.read_sql_query("SELECT code, industry_code FROM stock_industry",
                               con)
    finally:
        con.close()
    return dict(zip(mp["code"], mp["industry_code"]))


def neutralize_fund_features(mat, dates, codes, mode="cross", ind_map=None):
    """基本面横截面标准化/行业中性化 (PIT安全: 只用 t 日当天的横截面)

    - cross:    z = (x - 当日均值) / 当日标准差
    - industry: 先减行业内均值(无映射的行用全市场均值), 再除以当日标准差
    NaN 保留(不填充, 交给 LightGBM); 结果 clip ±5;
    日内有效样本 <10 或标准差 ~0 的列跳过(保留原值)"""
    if mode == "none":
        return mat
    out = np.array(mat, dtype=np.float32, copy=True)
    dates_a = np.asarray(dates)
    order = np.argsort(dates_a, kind="stable")
    sd = dates_a[order]
    uniq, cnts = np.unique(sd, return_counts=True)
    ind_arr = None
    if mode == "industry":
        ind_arr = np.array(
            [ind_map.get(str(c), "") if ind_map else "" for c in
             np.asarray(codes)], dtype=object)
    pos = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for _, c in zip(uniq, cnts):
            rows = order[pos:pos + c]
            pos += c
            b = out[rows].astype(np.float64)
            n = (~np.isnan(b)).sum(axis=0)
            mu = np.nanmean(b, axis=0)
            sd_ = np.nanstd(b, axis=0)
            if mode == "industry" and ind_arr is not None:
                gi = ind_arr[rows]
                for ind in np.unique(gi):
                    m = gi == ind
                    if m.sum() < 2:
                        continue
                    sub = b[m]
                    gm = np.nanmean(sub, axis=0)
                    use = ~np.isnan(gm)
                    sub[:, use] -= gm[use]
                    b[m] = sub
            else:
                good = (n >= 10) & np.isfinite(mu)
                b[:, good] -= mu[good]
            good = ((n >= 10) & np.isfinite(sd_) & (sd_ > 1e-12))
            if good.any():
                b[:, good] /= sd_[good]
                np.clip(b[:, good], -5.0, 5.0, out=b[:, good])
            out[rows] = b.astype(np.float32)
    return out


def yearly_metrics(dates, y, p, ret):
    """分年度验证: 每年 n/AUC/acc + 年内逐日 RankIC(对比T日实际收益)"""
    df = pd.DataFrame({"date": np.asarray(dates),
                       "y": np.asarray(y, dtype=np.float32),
                       "p": np.asarray(p, dtype=np.float64),
                       "r": np.asarray(ret, dtype=np.float32)})
    df["year"] = df["date"].str.slice(0, 4)
    rows = []
    for yy, g in df.groupby("year"):
        n = len(g)
        auc = (roc_auc_score(g["y"], g["p"])
               if g["y"].nunique() > 1 else float("nan"))
        acc = float(((g["p"] > 0.5).astype(np.int8) == g["y"]).mean())
        sub = pd.DataFrame({"date": g["date"], "prob": g["p"],
                            "next_ret": g["r"]})
        ic = daily_ic(sub)
        rows.append({"year": yy, "n": n, "auc": float(auc), "acc": acc,
                     "ic": float(ic.mean()) if len(ic) else 0.0,
                     "ic_n": int(len(ic))})
    del df
    gc.collect()
    return pd.DataFrame(rows)


def log_yearly_table(tab, label):
    for _, r in tab.iterrows():
        log(f"[{label}/{r['year']}] n={r['n']:,} auc={r['auc']:.4f} "
            f"acc={r['acc']:.4f} ic={r['ic']:.4f} (ic_n={r['ic_n']})")


def shift_group_for_open_exec(mat, icode):
    """topen口径特征前移: 每只股整体后移一行 (t行用t-1收盘及以前的特征),
    每股首行填NaN. 调用方须另行补上当日开盘缺口 gap_t (9:25已知)."""
    mat = np.asarray(mat, dtype=np.float32)
    same = np.empty(len(icode), dtype=bool)
    same[0] = False
    same[1:] = icode[1:] == icode[:-1]
    out = np.full_like(mat, np.nan)
    out[1:] = np.where(same[1:, None], mat[:-1], np.nan)
    return out


# ---------------------------------------------------------------- 多周期标签
def compute_tradable_open(df):
    """t开盘可买 (盘中信号锚定开盘价口径): 当日有开盘、非开盘涨停封板、
    距上一有行情日≤5天. 开盘价9:25集合竞价后已知, 当日可执行, 不算前视
    (回测按开盘价成交为乐观估计, 实盘建议另加滑点)"""
    n = len(df)
    o = df["open"].to_numpy(dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)
    cd = df["code"].cat.codes.values
    lim = df["lim"].values.astype(np.float64)
    date_idx = pd.factorize(df["date"], sort=True)[0]
    same_prev = np.empty(n, dtype=bool)
    same_prev[0] = False
    same_prev[1:] = cd[1:] == cd[:-1]
    pc = np.full(n, np.nan, dtype=np.float64)
    pc[same_prev] = c[np.where(same_prev)[0] - 1]
    gap_prev = np.full(n, 999, dtype=np.int32)
    gap_prev[same_prev] = (date_idx[same_prev]
                           - date_idx[np.where(same_prev)[0] - 1])
    log(f"t开盘可买样本占比 "
        f"{float((np.isfinite(o) & np.isfinite(pc) & (gap_prev <= 5) & (o < pc * (1.0 + lim - 0.005))).mean()):.3f}")
    return pd.Series(
        np.isfinite(o) & np.isfinite(pc) & (gap_prev <= 5)
        & (o < pc * (1.0 + lim - 0.005)), index=df.index)


def build_multilabels(df, tradable, horizons=(1, 5, 10), exec_lag=1):
    """多周期标签 (T+1执行, 无日内换仓)

    exec_lag=1 (默认): t收盘信号、t+1开盘买, R_h = close[t+h]/open[t+1]-1
    exec_lag=0 (盘中信号锚定开盘价): 信号日t=买入日、以open[t]为执行价,
        R_h = close[t+h]/open[t]-1 (h>=1; T+1卖出规则下最短过夜到t+1收盘,
        持仓共h+1段). 此时传入的tradable须是t开盘口径
        (见compute_tradable_open)

    mask_h: 入场可买(tradable, 与执行口径一致) 且持有窗内无>5天停牌
            且i+h仍是同只股票. R为float32(NaN=无效), Y为R>0"""
    n = len(df)
    o = df["open"].to_numpy(dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)
    cd = df["code"].cat.codes.values
    date_idx = pd.factorize(df["date"], sort=True)[0]
    gap = np.full(n, 999, dtype=np.int32)
    gap[:-1] = date_idx[1:] - date_idx[:-1]
    gap[:-1][cd[:-1] != cd[1:]] = 999
    cbad = np.concatenate([[0], np.cumsum((gap > 5).astype(np.int64))])
    entry = tradable.values
    out = {}
    for h in sorted(set(horizons)):
        if h < 1:
            raise ValueError(f"horizon须>=1: {h}")
        R = np.full(n, np.nan, dtype=np.float64)
        idx = np.arange(n - h)
        same = cd[idx] == cd[idx + h]
        sel = idx[same]
        if exec_lag == 1:
            R[sel] = c[sel + h] / o[sel + 1] - 1.0
        else:
            R[sel] = c[sel + h] / o[sel] - 1.0
        m = np.zeros(n, dtype=bool)
        m[sel] = (cbad[sel + h] - cbad[sel] == 0)
        m &= np.isfinite(R) & entry
        out[h] = (R.astype(np.float32), (R > 0).astype(np.int8), m)
    return out


# ---------------------------------------------------------------- 消融配置
def make_configs(mode, groups):
    order = [g for g in GROUPS if g in groups]
    configs = []
    if mode in ("full", "single"):
        configs += [(g,) for g in order]
    if mode in ("full", "cum"):
        cum = []
        for g in order:
            cum.append(g)
            configs.append(tuple(cum))
    if mode in ("full", "loo"):
        for g in order:
            configs.append(tuple(x for x in order if x != g))
    seen, out = set(), []
    for c in configs:
        key = "+".join(c)
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def save_group(name, arr, cols):
    p = os.path.join(OUT_DIR, f"feat_{name}.npy")
    np.save(p, arr)
    with open(os.path.join(OUT_DIR, f"feat_{name}.cols.json"), "w",
              encoding="utf-8") as f:
        json.dump(list(cols), f, ensure_ascii=False)
    log(f"特征组 {name}: {len(cols)} 列 -> {os.path.basename(p)}")
    del arr
    gc.collect()


def build_config(names, cfg, rows):
    """按配置从落盘特征文件中读入指定行, 拼成矩阵 (峰值=单矩阵)"""
    total = sum(len(names[g]) for g in cfg)
    out = np.empty((len(rows), total), dtype=np.float32)
    j = 0
    for g in cfg:
        k = len(names[g])
        m = np.load(os.path.join(OUT_DIR, f"feat_{g}.npy"), mmap_mode="r")
        out[:, j:j + k] = m[rows]
        del m
        j += k
    gc.collect()
    return out


def ece_of(y, p, n_bins=10):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    e, n = 0.0, len(y)
    for b in range(n_bins):
        m = idx == b
        if m.sum():
            e += m.sum() / n * abs(float(p[m].mean()) - float(y[m].mean()))
    return float(e)


def fit_platt(p_cal, y_cal):
    """Platt 校准: 在校准集 (默认2023年, 未参与训练) 上拟合 Logistic 映射"""
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression().fit(
        np.asarray(p_cal, dtype=np.float64).reshape(-1, 1),
        np.asarray(y_cal, dtype=np.float64))
    return lr


def apply_platt(lr, p):
    return lr.predict_proba(
        np.asarray(p, dtype=np.float64).reshape(-1, 1))[:, 1]


def thr_table(y, p, thrs=(0.50, 0.55, 0.60, 0.65)):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return {t: (float(y[p >= t].mean()) if (p >= t).any() else 0.0,
                int((p >= t).sum())) for t in thrs}


def build_regime_series(dates, mkt_base, kind="breadth"):
    """市场状态序列 (PIT安全: 只用 t 日收盘可得的 breadth5)

    - breadth: breadth5 > 0.5 为 risk-on
    - trend:   breadth5 > 自身20日均线为 risk-on (预热期默认 True)
    返回 pd.Series(date -> bool); risk-off 当天不开新仓, 在持 vintage 继续持有"""
    b = (pd.Series(mkt_base[:, MKT_COLS.index("breadth5")],
                   index=np.asarray(dates)).groupby(level=0).first())
    if kind == "trend":
        ma = b.rolling(20, min_periods=20).mean()
        return (b > ma).fillna(True)
    return b > 0.5


def _hold_perf_from_selection(sel_rows, dates_all, bars_oc, bench, h,
                              strategy, label, max_dd=0.0, buy_lag=1):
    """由信号行 vintage 构造重叠持仓曲线 (单策略).

    buy_lag=1: t+1开盘买, 持h段 (默认, 与R_h=close[t+h]/open[t+1]-1一致)
    buy_lag=0: 信号日t开盘买 (盘中信号锚定开盘价), 持h+1段至t+h收盘,
        与R_h=close[t+h]/open[t]-1一致
    max_dd > 0 时做因果回撤控制: 组合相对峰值回撤跌破 -max_dd 后空仓,
    直到回到阈值内 (逐日顺序模拟, 无前视)"""
    o_full, c_full = bars_oc[:, 0], bars_oc[:, 1]
    r0 = np.asarray(sel_rows)
    if len(r0) == 0:
        return None, None
    e = r0 + buy_lag
    n_seg = h if buy_lag == 1 else h + 1
    x = e + n_seg - 1
    rr = np.empty((len(r0), n_seg), dtype=np.float64)
    rr[:, 0] = c_full[e] / o_full[e] - 1.0
    for k in range(1, n_seg):
        rr[:, k] = c_full[e + k] / c_full[e + k - 1] - 1.0
    dd = np.stack([dates_all[e[i]:e[i] + n_seg] for i in range(len(r0))])
    long = pd.DataFrame({"date": dd.ravel(), "ret": rr.ravel()})
    g = long.groupby("date")["ret"].mean()
    act = long.groupby("date").size()
    ent = pd.Series(1, index=dates_all[e]).groupby(level=0).sum()
    ext = pd.Series(1, index=dates_all[x]).groupby(level=0).sum()
    to = (0.5 * (ent.add(ext, fill_value=0.0) / act)
          ).reindex(g.index).fillna(0.0)
    to_v, g_v = to.values, g.values
    if max_dd and max_dd > 0:
        # 因果模拟: 超回撤则空仓 (收益与换手同时置零)
        eq, peak = 1.0, 1.0
        active = np.zeros(len(g_v), dtype=bool)
        for i, r in enumerate(g_v - COST * to_v):
            if eq / peak - 1.0 <= -max_dd:
                continue
            active[i] = True
            eq *= (1.0 + r)
            peak = max(peak, eq)
        g_s = pd.Series(g_v * active, index=g.index)
        to = pd.Series(to_v * active, index=g.index)
    else:
        g_s = g
    m = perf(g_s, bench.reindex(g.index).fillna(0.0), to, act)
    m.update(h=h, period=label, strategy=strategy, days=len(g))
    curve = pd.DataFrame(
        {"date": g.index, "h": h, "strategy": strategy,
         "ret_net": (g_s.values - COST * to.values),
         "equity": (1.0 + g_s.values - COST * to.values).cumprod()})
    return m, curve


def _liquidity_amount_thr(K, liquidity):
    """TopK 流动性/容量联合阈值(元): 日成交额 >= max(下限, 1/(K*max_pct))."""
    if liquidity is None:
        return 0.0
    cap = 1.0 / (K * liquidity.get("max_turnover_pct", 0.05))
    return max(liquidity.get("min_amount", 0.0), cap)


def run_hold_backtest(dates_all, codes_all, mask_rows, prob, bars_oc,
                      bench, h, topks=(5, 10, 20, 50), label="",
                      regime=None, max_dd=0.0, buy_lag=1,
                      bars_amt=None, liquidity=None):
    """重叠持仓回测: 每个信号日t收盘选TopK, t+1开盘买入持有h天(t+h收盘卖)

    每日组合收益=在持vintage当日收益的等权平均; 成本按进/出分摊
    (每端COST/2, 与run_period的单程COST口径一致). 仍是T+1调仓, 无日内.
    buy_lag=0 时为盘中信号锚定开盘价口径 (信号日t开盘买, 持h+1段).
    regime 非空时 risk-off 当天不开新仓; max_dd>0 时做因果回撤控制.
    liquidity 非空时对 K==5 额外做流动性/容量过滤, 输出 Hold{h}Top5Liq"""
    bt = pd.DataFrame({"date": dates_all[mask_rows],
                       "code": codes_all[mask_rows],
                       "prob": prob, "row": mask_rows})
    if regime is not None:
        on = set(regime[regime].index)
        bt = bt[bt["date"].isin(on)]
    def _run_one_K(df_sub, K, name):
        sel = df_sub[df_sub["rk"] <= K]
        if sel.empty:
            return
        m, curve = _hold_perf_from_selection(
            sel["row"].to_numpy(), dates_all, bars_oc, bench, h,
            name, label, max_dd, buy_lag)
        if m is None:
            return
        m["K"] = K
        rows_out.append(m)
        curves.append(curve)

    rows_out, curves = [], []
    for K in topks:
        base = bt.copy()
        base["rk"] = base.groupby("date")["prob"].rank(ascending=False,
                                                       method="first")
        # 先生成无流动性约束的 Top5
        if K == 5 and liquidity is not None and bars_amt is not None:
            _run_one_K(base, K, f"Hold{h}Top{K}")
            # 再生成流动性约束后的 Top5Liq
            thr = _liquidity_amount_thr(K, liquidity)
            liq_ok = bars_amt[base["row"].to_numpy()] >= thr
            liq = base[liq_ok].copy()
            if not liq.empty:
                liq["rk"] = liq.groupby("date")["prob"].rank(
                    ascending=False, method="first")
                n_drop = (~liq_ok).sum()
                log(f"[{label}/Hold{h}Top{K}Liq] 流动性过滤剔除 {n_drop:,} 行 "
                    f"(thr>= {thr:,.0f} 元)")
                _run_one_K(liq, K, f"Hold{h}Top{K}Liq")
            else:
                log(f"[{label}] Top5 流动性过滤后无候选 (thr={thr:,.0f})")
        else:
            _run_one_K(base, K, f"Hold{h}Top{K}")
    return (pd.DataFrame(rows_out),
            pd.concat(curves, ignore_index=True) if curves
            else pd.DataFrame())


def run_hold_threshold_backtest(dates_all, codes_all, mask_rows, prob,
                                bars_oc, bench, h,
                                thrs=(0.50, 0.55, 0.60), label="",
                                regime=None, max_dd=0.0, buy_lag=1,
                                min_n=MIN_HOLD, min_cov=0.0):
    """阈值类重叠持仓回测: 信号日选 prob>=thr (入选不足 min_n 只则当天空仓),
    t+1开盘买入持有h天; 每日组合=在持vintage等权平均; 成本与 TopK 口径一致.
    buy_lag=0 时为盘中信号锚定开盘价口径 (信号日t开盘买, 持h+1段).
    min_cov>0 时覆盖天数不足的策略整行丢弃 (杀小样本假象); 每行附带
    cover (有持仓信号日占比) 与 active_days 供审查"""
    bt = pd.DataFrame({"date": dates_all[mask_rows],
                       "code": codes_all[mask_rows],
                       "prob": prob, "row": mask_rows})
    if regime is not None:
        on = set(regime[regime].index)
        bt = bt[bt["date"].isin(on)]
    all_days = pd.Series(dates_all[mask_rows]).nunique()
    rows_out, curves = [], []
    for thr in thrs:
        sel = bt[bt["prob"] >= thr]
        cnt = sel.groupby("date").size()
        sel = sel[~sel["date"].isin(set(cnt[cnt < min_n].index))]
        if sel.empty:
            continue
        active_days = int(sel["date"].nunique())
        cover = active_days / max(int(all_days), 1)
        if cover < min_cov:
            log(f"[{label}/P{thr:.2f}] 覆盖 {cover:.1%} < {min_cov:.0%}, "
                f"丢弃 (小样本假象)")
            continue
        m, curve = _hold_perf_from_selection(
            sel["row"].to_numpy(), dates_all, bars_oc, bench, h,
            f"Hold{h}P{thr:.2f}", label, max_dd, buy_lag)
        if m is None:
            continue
        m["thr"] = thr
        m["cover"] = cover
        m["active_days"] = active_days
        rows_out.append(m)
        curves.append(curve)
    return (pd.DataFrame(rows_out),
            pd.concat(curves, ignore_index=True) if curves
            else pd.DataFrame())


def run_horizon_mode(args, params, thermal_cb, groups, names, labels,
                     warm, year, dates, codes, nr_all, bars_oc, t0,
                     regime=None, bars_amt=None, liquidity=None):
    """多周期模式: 每个horizon独立训练一个二分类模型(特征=全部请求组)

    报告: valid AUC/logloss/acc + 全验证集逐日RankIC(对比T日实际收益)
    + 分年度验证表 + TopK/阈值两套重叠持仓回测.
    --calibrate platt 时切分改为 2018~(CY-1)训练 / CY年校准 / (CY+1)~评估,
    回测与阈值重搜都用校准后概率; regime/max_dd 作用于重叠持仓回测"""
    allg = tuple(groups)
    horizons = sorted(labels)
    rankic_mode = args.objective == "rankic"
    log(f"horizon模式: 周期 {horizons}, 特征组 {list(allg)}")
    cfgs = []
    if args.per_group:
        cfgs += [(g,) for g in groups]
    cfgs.append(allg)
    seen, dedup = set(), []
    for c in cfgs:
        if c not in seen:
            seen.add(c)
            dedup.append(c)
    cfgs = dedup
    log(f"配置 {len(cfgs)} 个: {[c[0] if len(c) == 1 else 'all' for c in cfgs]}")
    calib = getattr(args, "calibrate", "none") != "none"
    cy = int(getattr(args, "calib_year", 2023))
    va_min_year = (cy + 1) if calib else int(TRAIN_END[:4]) + 1
    tr_max_year = (cy - 1) if calib else int(TRAIN_END[:4])
    age_ref = pd.Timestamp(f"{tr_max_year}-12-31")
    if calib:
        log(f"Platt校准流程: 训练2018~{tr_max_year} / 校准{cy}年 / "
            f"评估{va_min_year}年起")
    if regime is not None:
        log(f"regime过滤开启: risk-off当天不开新仓 "
            f"(回撤控制 max_dd={getattr(args, 'max_dd', 0.0)})")
    hold_thrs = tuple(float(v) for v in
                      getattr(args, "hold_thrs", "0.50,0.55,0.60").split(",")
                      if v.strip())
    buy_lag = 1 if getattr(args, "exec", "t1open") == "t1open" else 0
    psuf = "" if buy_lag == 1 else "@open"
    if buy_lag == 0:
        log("执行口径: 盘中信号锚定开盘价 (信号日=买入日, open[t]执行, "
            "R_h=close[t+h]/open[t]-1)")
    rs = np.random.RandomState(args.seed)
    vb = np.flatnonzero(np.isfinite(nr_all) & warm
                        & (year >= va_min_year).values)
    bench = pd.Series(nr_all[vb], index=dates[vb]).groupby(level=0).mean()
    log(f"市场基准: {len(bench)} 个交易日")
    rows, hold_all, hold_curves = [], [], []
    for h in horizons:
        R, Y, M0 = labels[h]
        M = M0 & warm
        yv = Y.astype(np.float32)
        tr_all = np.flatnonzero(M & (year <= tr_max_year).values)
        va_all = np.flatnonzero(M & (year >= va_min_year).values)
        cal_all = (np.flatnonzero(M & (year == cy).values) if calib
                   else np.array([], dtype=int))
        tr_h = (np.sort(rs.choice(
            tr_all, size=min(args.train_sample, len(tr_all)),
            replace=False)) if args.train_sample else tr_all)
        va_h = (np.sort(rs.choice(
            va_all, size=min(args.valid_sample, len(va_all)),
            replace=False)) if args.valid_sample else va_all)
        cal_h = cal_all
        age = (age_ref - pd.to_datetime(dates[tr_h])).days.values
        w_h = (0.5 ** (age / 756.0)).astype(np.float32)
        bench_h = bench
        if buy_lag == 0:
            rb = np.flatnonzero(M & (year >= va_min_year).values)
            bench_h = pd.Series(R[rb], index=dates[rb]).groupby(level=0).mean()

        def train_one(cfg, flabel):
            feat_names = [nm for g in cfg for nm in names[g]]
            log(f"[h={h}/{flabel}] 特征 {len(feat_names)}, "
                f"训练 {len(tr_h):,} / 验证 {len(va_h):,}"
                + (f" / 校准 {len(cal_h):,}" if calib else ""))
            Xtr = build_config(names, cfg, tr_h)
            dtr = lgb.Dataset(Xtr, label=yv[tr_h], weight=w_h,
                              feature_name=feat_names)
            del Xtr
            gc.collect()
            Xva = build_config(names, cfg, va_h)
            dva = lgb.Dataset(Xva, label=yv[va_h], reference=dtr)
            del Xva
            gc.collect()
            train_kw = {"params": params, "train_set": dtr,
                        "num_boost_round": args.rounds, "valid_sets": [dva]}
            if rankic_mode:
                from rank_ic_objective import (make_lambdarank_ic_objective,
                                               make_rank_auc_eval,
                                               make_rank_ic_eval)
                p = dict(params)
                p["objective"] = make_lambdarank_ic_objective(
                    R[tr_h], dates[tr_h])
                p["metric"] = "None"
                train_kw["params"] = p
                train_kw["feval"] = [make_rank_ic_eval(R[va_h], dates[va_h]),
                                     make_rank_auc_eval(yv[va_h])]
            model = lgb.train(
                **train_kw,
                callbacks=[lgb.early_stopping(150, first_metric_only=True,
                                              verbose=False),
                           thermal_cb,
                           lgb.log_evaluation(200)])
            Xva = build_config(names, cfg, va_h)
            p_raw = model.predict(Xva, num_iteration=model.best_iteration)
            if rankic_mode:
                p_raw = 1.0 / (1.0 + np.exp(-np.clip(p_raw, -50, 50)))
            del Xva
            gc.collect()
            p_va = p_raw
            prow = {}
            if calib:
                Xc = build_config(names, cfg, cal_h)
                p_cal = model.predict(Xc,
                                      num_iteration=model.best_iteration)
                if rankic_mode:
                    p_cal = 1.0 / (1.0 + np.exp(-np.clip(p_cal, -50, 50)))
                del Xc
                gc.collect()
                lr = fit_platt(p_cal, yv[cal_h])
                p_va = apply_platt(lr, p_raw).astype(np.float64)
                prow = {"platt_a": float(lr.coef_[0, 0]),
                        "platt_b": float(lr.intercept_[0]),
                        "calib_n": len(cal_h),
                        "raw_logloss": log_loss(yv[va_h], p_raw),
                        "raw_ece": ece_of(yv[va_h], p_raw),
                        "logloss": log_loss(yv[va_h], p_va),
                        "ece": ece_of(yv[va_h], p_va)}
                log(f"[h={h}/{flabel}] Platt(a={prow['platt_a']:.3f},"
                    f"b={prow['platt_b']:.3f}) 校准n={len(cal_h):,}: "
                    f"LogLoss {prow['raw_logloss']:.4f}->"
                    f"{prow['logloss']:.4f}, ECE {prow['raw_ece']:.4f}->"
                    f"{prow['ece']:.4f}")
                th = thr_table(yv[va_h], p_va)
                log(f"[h={h}/{flabel}] 校准后阈值胜率: " + ", ".join(
                    f"P≥{t:.2f} {v[0]:.4f}(n={v[1]:,})" for t, v in th.items()))
            bt = pd.DataFrame({"date": dates[va_h], "prob": p_va,
                               "next_ret": R[va_h]})
            ic = daily_ic(bt)
            ic_mean = float(ic.mean()) if len(ic) else 0.0
            ic_ir = float(ic_mean / (ic.std() + 1e-12)) if len(ic) else 0.0
            row = {"horizon": h, "config": flabel, "n_feat": len(feat_names),
                   "n_train": len(tr_h), "n_valid": len(va_h),
                   "best_iter": model.best_iteration,
                   "valid_auc": roc_auc_score(yv[va_h], p_va),
                   "valid_logloss": log_loss(yv[va_h], p_va),
                   "valid_acc": accuracy_score(yv[va_h], p_va > 0.5),
                   "base_rate": float(yv[va_h].mean()),
                   "ic": ic_mean, "ic_ir": ic_ir}
            row.update(prow)
            ytab = yearly_metrics(dates[va_h], yv[va_h], p_va, R[va_h])
            log_yearly_table(ytab, f"h={h}/{flabel}")
            for _, r in ytab.iterrows():
                row[f"auc_{r['year']}"] = float(r["auc"])
                row[f"ic_{r['year']}"] = float(r["ic"])
            if model.best_iteration <= 5:
                log(f"[h={h}/{flabel}] 提示: best_iter="
                    f"{model.best_iteration} 接近单树, 上面的分年度表即"
                    f"稳定性验证, 高IC需各年同号才算数")
            if h == 1 and not args.no_backtest:
                bt2 = pd.DataFrame({
                    "date": dates[va_h], "code": codes[va_h], "prob": p_va,
                    "next_ret": R[va_h], "y": yv[va_h]})
                s, _, _, _, _, _ = run_period(bt2, f"h={h}/{flabel}{psuf}")
                for strat in ("Top20", "Top50", "P>=0.55"):
                    m = s[s["strategy"] == strat]
                    if not m.empty:
                        key = strat.replace(">=", "").replace(".", "")
                        row[f"{key}_ann_net"] = float(m.iloc[0]["ann_net"])
                        row[f"{key}_sharpe"] = float(m.iloc[0]["sharpe"])
                row["mkt_ann_net"] = float(
                    s[s["strategy"] == "市场等权"].iloc[0]["ann_net"])
                del bt2, s
            if not args.no_backtest:
                hs, hc = run_hold_backtest(
                    dates, codes, va_h, p_va, bars_oc, bench_h, h,
                    topks=(5, 10, 20, 50), label=f"h={h}/{flabel}{psuf}",
                    regime=regime, max_dd=args.max_dd, buy_lag=buy_lag,
                    bars_amt=bars_amt, liquidity=liquidity)
                for _, m in hs.iterrows():
                    key = "liq" if "Liq" in str(m.get("strategy", "")) else m['K']
                    row[f"hold{key}_ann"] = float(m["ann_net"])
                    row[f"hold{key}_sharpe"] = float(m["sharpe"])
                    row[f"hold{key}_mdd"] = float(m["mdd"])
                hold_all.append(hs)
                hold_curves.append(hc)
                ts, tc = run_hold_threshold_backtest(
                    dates, codes, va_h, p_va, bars_oc, bench_h, h,
                    thrs=hold_thrs, label=f"h={h}/{flabel}{psuf}",
                    regime=regime, max_dd=args.max_dd, buy_lag=buy_lag,
                    min_n=args.hold_min_n, min_cov=args.hold_min_cov)
                for _, m in ts.iterrows():
                    key = f"holdP{m['thr']:.2f}".replace(".", "")
                    row[f"{key}_ann"] = float(m["ann_net"])
                    row[f"{key}_sharpe"] = float(m["sharpe"])
                    row[f"{key}_mdd"] = float(m["mdd"])
                    row[f"{key}_cov"] = float(m["cover"])
                if len(ts):
                    hold_all.append(ts)
                    hold_curves.append(tc)
            rows.append(row)
            log(f"[h={h}/{flabel}] 完成: valid_auc {row['valid_auc']:.4f}, "
                f"IC {row['ic']:.4f}, best_iter {row['best_iter']}, "
                f"用时 {(time.time() - t0) / 60:.1f} min")
            model.save_model(
                os.path.join(OUT_DIR, f"lgbm_h{h}_{flabel}.txt"))
            del dtr, dva, model, bt
            gc.collect()

        for cfg in cfgs:
            flabel = cfg[0] if len(cfg) == 1 else "all"
            train_one(cfg, flabel)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT_DIR, "horizon_results.csv"), index=False)
    if hold_all:
        pd.concat(hold_all, ignore_index=True).to_csv(
            os.path.join(OUT_DIR, "holdbacktest_summary.csv"), index=False)
    if hold_curves:
        pd.concat([c for c in hold_curves if len(c)], ignore_index=True).to_csv(
            os.path.join(OUT_DIR, "holdbacktest_daily.csv"), index=False)
    print("\n================ 多周期结果 "
          "(IC=全验证集逐日RankIC, 对比T日实际收益) ================")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    with open(os.path.join(OUT_DIR, "horizon_run_config.json"), "w",
              encoding="utf-8") as f:
        json.dump({"mode": "horizon", "groups": groups,
                   "horizons": horizons, "params": params,
                   "limit": args.limit,
                   "fund_norm": getattr(args, "fund_norm", "cross"),
                   "calibrate": getattr(args, "calibrate", "none"),
                   "calib_year": getattr(args, "calib_year", 2023),
                   "regime": getattr(args, "regime", "none"),
                   "max_dd": getattr(args, "max_dd", 0.0),
                   "hold_thrs": getattr(args, "hold_thrs", "0.50,0.55,0.60"),
                   "exec": getattr(args, "exec", "t1open")},
                  f, ensure_ascii=False, indent=2)
    if not args.keep_features:
        for g in names:
            for suffix in (".npy", ".cols.json"):
                try:
                    os.remove(os.path.join(OUT_DIR, f"feat_{g}{suffix}"))
                except OSError:
                    pass
    print(f"\n输出目录: {OUT_DIR}")
    log(f"全部完成, 总用时 {(time.time() - t0) / 60:.1f} min")


def run_walkforward_mode(args, params, thermal_cb, groups, names, labels,
                         warm, year, dates, codes, nr_all, bars_oc, t0,
                         regime=None, bars_amt=None, liquidity=None):
    """walk-forward 多窗口 + holdout 定稿评估.

    窗口: 对 --wf-valid-years 的每个验证年 vy, 用 2018~(vy-1) 训练、
    vy 年验证; 最后用全量训练、2024+ holdout 定稿评估.
    --calibrate platt 时最终模型改用 2018~(CY-1)训练 / CY年校准 / 评估."""
    allg = tuple(groups)
    horizons = sorted(labels)
    rankic_mode = args.objective == "rankic"
    vy_list = [int(v) for v in args.wf_valid_years.split(",") if v.strip()]
    log(f"walk-forward: 验证年 {vy_list}, 周期 {horizons}, "
        f"特征组 {list(allg)}")
    calib = getattr(args, "calibrate", "none") != "none"
    cy = int(getattr(args, "calib_year", 2023))
    va_min_year = (cy + 1) if calib else int(TRAIN_END[:4]) + 1
    tr_max_year = (cy - 1) if calib else int(TRAIN_END[:4])
    hold_thrs = tuple(float(v) for v in
                      getattr(args, "hold_thrs", "0.50,0.55,0.60").split(",")
                      if v.strip())
    buy_lag = 1 if getattr(args, "exec", "t1open") == "t1open" else 0
    psuf = "" if buy_lag == 1 else "@open"
    if buy_lag == 0:
        log("执行口径: 盘中信号锚定开盘价 (信号日=买入日, open[t]执行, "
            "R_h=close[t+h]/open[t]-1)")
    rows, hold_all, hold_curves = [], [], []

    def fit_window(R, yv, tr_idx, va_idx, w, h, tag, cal_idx=None,
                   do_period=False):
        feat_names = [nm for g in allg for nm in names[g]]
        log(f"[h={h}/{tag}] 特征 {len(feat_names)}, 训练 {len(tr_idx):,} / "
            f"验证 {len(va_idx):,}"
            + (f" / 校准 {len(cal_idx):,}" if cal_idx is not None else ""))
        Xtr = build_config(names, allg, tr_idx)
        dtr = lgb.Dataset(Xtr, label=yv[tr_idx], weight=w,
                          feature_name=feat_names)
        del Xtr
        gc.collect()
        Xva = build_config(names, allg, va_idx)
        dva = lgb.Dataset(Xva, label=yv[va_idx], reference=dtr)
        del Xva
        gc.collect()
        train_kw = {"params": params, "train_set": dtr,
                    "num_boost_round": args.rounds, "valid_sets": [dva]}
        if rankic_mode:
            from rank_ic_objective import (make_lambdarank_ic_objective,
                                           make_rank_auc_eval,
                                           make_rank_ic_eval)
            p = dict(params)
            p["objective"] = make_lambdarank_ic_objective(
                R[tr_idx], dates[tr_idx])
            p["metric"] = "None"
            train_kw["params"] = p
            train_kw["feval"] = [make_rank_ic_eval(R[va_idx], dates[va_idx]),
                                 make_rank_auc_eval(yv[va_idx])]
        model = lgb.train(
            **train_kw,
            callbacks=[lgb.early_stopping(150, first_metric_only=True,
                                          verbose=False),
                       thermal_cb,
                       lgb.log_evaluation(200)])
        Xva = build_config(names, allg, va_idx)
        p_raw = model.predict(Xva, num_iteration=model.best_iteration)
        if rankic_mode:
            p_raw = 1.0 / (1.0 + np.exp(-np.clip(p_raw, -50, 50)))
        del Xva
        gc.collect()
        p_va = p_raw
        prow = {}
        if cal_idx is not None and len(cal_idx):
            Xc = build_config(names, allg, cal_idx)
            p_cal = model.predict(Xc, num_iteration=model.best_iteration)
            if rankic_mode:
                p_cal = 1.0 / (1.0 + np.exp(-np.clip(p_cal, -50, 50)))
            del Xc
            gc.collect()
            lr = fit_platt(p_cal, yv[cal_idx])
            p_va = apply_platt(lr, p_raw).astype(np.float64)
            prow = {"platt_a": float(lr.coef_[0, 0]),
                    "platt_b": float(lr.intercept_[0]),
                    "calib_n": len(cal_idx),
                    "raw_logloss": log_loss(yv[va_idx], p_raw),
                    "logloss": log_loss(yv[va_idx], p_va),
                    "ece": ece_of(yv[va_idx], p_va)}
            log(f"[h={h}/{tag}] Platt: LogLoss {prow['raw_logloss']:.4f}->"
                f"{prow['logloss']:.4f}, ECE {prow['ece']:.4f}")
        bt = pd.DataFrame({"date": dates[va_idx], "prob": p_va,
                           "next_ret": R[va_idx]})
        ic = daily_ic(bt)
        ic_mean = float(ic.mean()) if len(ic) else 0.0
        row = {"horizon": h, "window": tag, "n_feat": len(feat_names),
               "n_train": len(tr_idx), "n_valid": len(va_idx),
               "best_iter": model.best_iteration,
               "valid_auc": roc_auc_score(yv[va_idx], p_va),
               "valid_logloss": log_loss(yv[va_idx], p_va),
               "valid_acc": accuracy_score(yv[va_idx], p_va > 0.5),
               "base_rate": float(yv[va_idx].mean()),
               "ic": ic_mean,
               "ic_ir": float(ic_mean / (ic.std() + 1e-12)) if len(ic) else 0.0}
        row.update(prow)
        ytab = yearly_metrics(dates[va_idx], yv[va_idx], p_va, R[va_idx])
        log_yearly_table(ytab, f"h={h}/{tag}")
        vb = np.flatnonzero(np.isfinite(nr_all) & warm
                            & np.isin(year, ytab["year"].astype(int)
                                      .to_numpy()))
        bench = pd.Series(nr_all[vb], index=dates[vb]).groupby(level=0).mean()
        if buy_lag == 0:
            bench = pd.Series(R[va_idx], index=dates[va_idx]
                              ).groupby(level=0).mean()
        if do_period and h == 1 and not args.no_backtest:
            bt2 = pd.DataFrame({
                "date": dates[va_idx], "code": codes[va_idx], "prob": p_va,
                "next_ret": R[va_idx], "y": yv[va_idx]})
            s, _, _, _, _, _ = run_period(bt2, f"h={h}/{tag}{psuf}")
            for strat in ("Top20", "Top50", "P>=0.55"):
                m = s[s["strategy"] == strat]
                if not m.empty:
                    key = strat.replace(">=", "").replace(".", "")
                    row[f"{key}_ann_net"] = float(m.iloc[0]["ann_net"])
                    row[f"{key}_sharpe"] = float(m.iloc[0]["sharpe"])
            row["mkt_ann_net"] = float(
                s[s["strategy"] == "市场等权"].iloc[0]["ann_net"])
            del bt2, s
        if not args.no_backtest:
            hs, hc = run_hold_backtest(
                dates, codes, va_idx, p_va, bars_oc, bench, h,
                topks=(5, 10, 20, 50), label=f"h={h}/{tag}{psuf}",
                regime=regime, max_dd=args.max_dd, buy_lag=buy_lag,
                bars_amt=bars_amt, liquidity=liquidity)
            for _, m in hs.iterrows():
                key = "liq" if "Liq" in str(m.get("strategy", "")) else m['K']
                row[f"hold{key}_ann"] = float(m["ann_net"])
                row[f"hold{key}_sharpe"] = float(m["sharpe"])
                row[f"hold{key}_mdd"] = float(m["mdd"])
            hold_all.append(hs)
            hold_curves.append(hc)
            ts, tc = run_hold_threshold_backtest(
                dates, codes, va_idx, p_va, bars_oc, bench, h,
                thrs=hold_thrs, label=f"h={h}/{tag}{psuf}",
                regime=regime, max_dd=args.max_dd, buy_lag=buy_lag,
                min_n=args.hold_min_n, min_cov=args.hold_min_cov)
            for _, m in ts.iterrows():
                key = f"holdP{m['thr']:.2f}".replace(".", "")
                row[f"{key}_ann"] = float(m["ann_net"])
                row[f"{key}_sharpe"] = float(m["sharpe"])
                row[f"{key}_mdd"] = float(m["mdd"])
                row[f"{key}_cov"] = float(m["cover"])
            if len(ts):
                hold_all.append(ts)
                hold_curves.append(tc)
        rows.append(row)
        log(f"[h={h}/{tag}] 完成: valid_auc {row['valid_auc']:.4f}, "
            f"IC {row['ic']:.4f}, best_iter {row['best_iter']}, "
            f"用时 {(time.time() - t0) / 60:.1f} min")
        model.save_model(os.path.join(
            OUT_DIR, f"lgbm_wf_h{h}_{tag.replace('/', '_')}.txt"))
        del dtr, dva, model, bt
        gc.collect()

    rs = np.random.RandomState(args.seed)
    for h in horizons:
        R, Y, M0 = labels[h]
        M = M0 & warm
        yv = Y.astype(np.float32)
        for vy in vy_list:
            tr_all = np.flatnonzero(M & (year <= vy - 1).values)
            va_all = np.flatnonzero(M & (year == vy).values)
            if len(tr_all) == 0 or len(va_all) == 0:
                log(f"[h={h}/valid{vy}] 样本不足, 跳过")
                continue
            tr_i = (np.sort(rs.choice(
                tr_all, size=min(args.train_sample, len(tr_all)),
                replace=False)) if args.train_sample else tr_all)
            va_i = (np.sort(rs.choice(
                va_all, size=min(args.valid_sample, len(va_all)),
                replace=False)) if args.valid_sample else va_all)
            age = (pd.Timestamp(f"{vy - 1}-12-31")
                   - pd.to_datetime(dates[tr_i])).days.values
            w = (0.5 ** (age / 756.0)).astype(np.float32)
            fit_window(R, yv, tr_i, va_i, w, h, f"train~{vy - 1}/valid{vy}")
        tr_all = np.flatnonzero(M & (year <= tr_max_year).values)
        va_all = np.flatnonzero(M & (year >= va_min_year).values)
        cal_all = (np.flatnonzero(M & (year == cy).values) if calib
                   else None)
        tr_i = (np.sort(rs.choice(
            tr_all, size=min(args.train_sample, len(tr_all)),
            replace=False)) if args.train_sample else tr_all)
        va_i = (np.sort(rs.choice(
            va_all, size=min(args.valid_sample, len(va_all)),
            replace=False)) if args.valid_sample else va_all)
        age = (pd.Timestamp(f"{tr_max_year}-12-31")
               - pd.to_datetime(dates[tr_i])).days.values
        w = (0.5 ** (age / 756.0)).astype(np.float32)
        fit_window(R, yv, tr_i, va_i, w, h, "holdout", cal_idx=cal_all,
                   do_period=True)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT_DIR, "walkforward_results.csv"), index=False)
    if hold_all:
        pd.concat(hold_all, ignore_index=True).to_csv(
            os.path.join(OUT_DIR, "holdbacktest_wf_summary.csv"), index=False)
    if hold_curves:
        pd.concat([c for c in hold_curves if len(c)], ignore_index=True).to_csv(
            os.path.join(OUT_DIR, "holdbacktest_wf_daily.csv"), index=False)
    print("\n================ walk-forward + holdout 结果 ================")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    with open(os.path.join(OUT_DIR, "walkforward_run_config.json"), "w",
              encoding="utf-8") as f:
        json.dump({"mode": "walkforward", "groups": groups,
                   "horizons": horizons, "valid_years": vy_list,
                   "params": params, "limit": args.limit,
                   "fund_norm": getattr(args, "fund_norm", "cross"),
                   "calibrate": getattr(args, "calibrate", "none"),
                   "calib_year": getattr(args, "calib_year", 2023),
                   "regime": getattr(args, "regime", "none"),
                   "max_dd": getattr(args, "max_dd", 0.0),
                   "exec": getattr(args, "exec", "t1open")},
                  f, ensure_ascii=False, indent=2)
    if not args.keep_features:
        for g in names:
            for suffix in (".npy", ".cols.json"):
                try:
                    os.remove(os.path.join(OUT_DIR, f"feat_{g}{suffix}"))
                except OSError:
                    pass
    print(f"\n输出目录: {OUT_DIR}")
    log(f"全部完成, 总用时 {(time.time() - t0) / 60:.1f} min")


def main():
    global OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="随机抽取N只股票(冒烟)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--mode", default="full",
                    choices=["full", "single", "cum", "loo", "horizon",
                             "walkforward"])
    ap.add_argument("--horizons", default="1,5,10",
                    help="horizon模式的预测周期(天), 逗号分隔. "
                    "R_h=close[t+h]/open[t+1]-1")
    ap.add_argument("--per-group", action="store_true",
                    help="horizon模式下每个特征组单独训练(找每周期最强组)")
    ap.add_argument("--groups", default=DEFAULT_GROUPS,
                    help="逗号分隔, 可选: " + ",".join(GROUPS))
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--leaves", type=int, default=31)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--threads", type=int, default=6, help="LightGBM线程数")
    ap.add_argument("--throttle", type=float, default=0.0,
                    help="每轮提升后休眠秒数, 降低发热")
    ap.add_argument("--temp-limit", type=float, default=92.0,
                    help="CPU温度超过该值自动挂起降温")
    ap.add_argument("--temp-resume", type=float, default=85.0,
                    help="降温到该值以下继续")
    ap.add_argument("--train-sample", type=int, default=0,
                    help="训练采样行数, 0=全量(默认)")
    ap.add_argument("--valid-sample", type=int, default=0,
                    help="验证采样行数, 0=全量(默认)")
    ap.add_argument("--train-auc-sample", type=int, default=500_000,
                    help="训练集AUC评估子样本行数, 0=全量")
    ap.add_argument("--keep-features", action="store_true",
                    help="保留落盘的特征矩阵文件")
    ap.add_argument("--no-backtest", action="store_true")
    ap.add_argument("--fund-norm", default="none",
                    choices=["none", "cross", "industry"],
                    help="基本面横截面处理: none=原始值 (默认, walk-forward裁决: "
                    "树模型不需要标准化, cross会洗掉全市场估值水位本身的"
                    "牛熊信息); cross=当日z-score, industry=行业中性化后"
                    "z-score (只用t日横截面, PIT安全)")
    ap.add_argument("--tech-norm", default="none",
                    choices=["none", "cross", "industry"],
                    help="技术指标tech组横截面处理: none=原始值(默认), "
                    "cross=当日z-score, industry=行业中性化后z-score")
    ap.add_argument("--pattern-norm", default="none",
                    choices=["none", "cross", "industry"],
                    help="K线形态pattern组横截面处理: none=原始值(默认), "
                    "cross=当日z-score, industry=行业中性化后z-score")
    ap.add_argument("--calibrate", default="none",
                    choices=["none", "platt"],
                    help="Platt校准: 2018~(CY-1)训练/CY年校准/(CY+1)~评估, "
                    "回测与阈值重搜用校准后概率")
    ap.add_argument("--calib-year", type=int, default=2023,
                    help="校准年CY (默认2023)")
    ap.add_argument("--regime", default="none",
                    choices=["none", "breadth", "trend"],
                    help="市场状态过滤: risk-off当天不开新仓(重叠持仓回测)")
    ap.add_argument("--max-dd", type=float, default=0.0,
                    help="因果回撤控制: 组合回撤跌破该值后空仓, 0=关闭")
    ap.add_argument("--hold-thrs", default="0.50,0.55,0.60",
                    help="阈值类重叠持仓回测的概率阈值, 逗号分隔")
    ap.add_argument("--hold-min-n", type=int, default=5,
                    help="阈值策略单日最少入选只数, 不足则当天空仓")
    ap.add_argument("--hold-min-cov", type=float, default=0.2,
                    help="阈值策略最小覆盖天数占比, 不足整行丢弃 (杀小样本假象; "
                    "walk-forward实证: 835%%/14993%%类极值覆盖均<15%%)")
    ap.add_argument("--liquidity", action="store_true",
                    help="对 Top5 额外施加流动性/容量约束 (生成 Hold{h}Top5Liq)")
    ap.add_argument("--liq-min-amount", type=float, default=10_000_000.0,
                    help="Top5 流动性约束: 个股日成交额下限(元), 默认1000万")
    ap.add_argument("--liq-max-turnover-pct", type=float, default=0.05,
                    help="Top5 容量约束: 单票仓位占个股日成交额上限, 默认5%%")
    ap.add_argument("--liq-impact-bps", type=float, default=0.0,
                    help="Top5 冲击成本附加(年化双边bps), 默认0; 例如10表示在"
                    "成交额不足时额外扣10bps成本")
    ap.add_argument("--wf-valid-years", default="2022,2023",
                    help="walk-forward验证年, 逗号分隔")
    ap.add_argument("--exec", default="t1open",
                    choices=["t1open", "topen"],
                    help="执行口径: t1open=t收盘信号/t+1开盘买 "
                    "(R=close[t+h]/open[t+1]-1); topen=盘中信号锚定开盘价, "
                    "信号日=买入日/open[t]执行 (R=close[t+h]/open[t]-1), "
                    "仅horizon/walkforward可用")
    ap.add_argument("--out-dir", default=OUT_DIR,
                    help="输出目录, 默认 ablation_output")
    ap.add_argument("--objective", default="binary",
                    choices=["binary", "rankic"],
                    help="训练目标: binary=二分类(默认), rankic=LambdaRankIC"
                    "直接优化逐日Rank IC")
    args = ap.parse_args()

    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    # 若用户显式指定 tech_norm/pattern_norm, 自动开启对应 norm 参数
    if "tech_norm" in groups and args.tech_norm == "none":
        args.tech_norm = "cross"
    if "pattern_norm" in groups and args.pattern_norm == "none":
        args.pattern_norm = "cross"
    # tech/pattern 若指定了 norm, 内部实际使用 tech_norm/pattern_norm 文件
    if args.tech_norm != "none":
        groups = ["tech_norm" if g == "tech" else g for g in groups]
    if args.pattern_norm != "none":
        groups = ["pattern_norm" if g == "pattern" else g for g in groups]
    # 去重并保持顺序
    seen, dedup = set(), []
    for g in groups:
        if g not in seen:
            seen.add(g)
            dedup.append(g)
    groups = dedup
    for g in groups:
        if g not in GROUPS:
            raise SystemExit(f"未知特征组 {g}, 可选: {', '.join(GROUPS)}")
    if args.exec != "t1open" and args.mode not in ("horizon", "walkforward"):
        raise SystemExit("--exec topen 仅 horizon/walkforward 模式可用")
    if "market" in groups:
        log("警告: market 组是每日常数特征, 选股模型应剔除(见README §5.3/§11); "
            "仅大盘择时研究时保留, 回测出现恒定预测/固定篮子即退化信号")

    OUT_DIR = args.out_dir
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    df = load_bars(args.limit or None, args.seed)
    X, next_ret, y, tradable = build_features(df)
    year = df["date"].str.slice(0, 4).astype(int)
    dates = df["date"].values
    code_cat = df["code"].cat.categories.to_numpy()
    codes = code_cat[df["code"].cat.codes.to_numpy()]
    n = len(df)
    gap_t = None
    if args.exec == "topen":
        # 当日开盘缺口 (9:25集合竞价后已知, topen口径下唯一可用的当日信息)
        o = df["open"].to_numpy(dtype=np.float64)
        c = df["close"].to_numpy(dtype=np.float64)
        cd = df["code"].cat.codes.values
        same = np.empty(n, dtype=bool)
        same[0] = False
        same[1:] = cd[1:] == cd[:-1]
        gap_t = np.full(n, np.nan, dtype=np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            gap_t[same] = (o[same] / c[np.where(same)[0] - 1] - 1.0)
        del o, c, cd, same
        gc.collect()

    base = ((year >= int(DATA_START[:4])) & next_ret.notna() & tradable
            & X["ret60"].notna()).values
    tr_all = np.flatnonzero(base & (year <= int(TRAIN_END[:4])).values)
    va_all = np.flatnonzero(base & ~(year <= int(TRAIN_END[:4])).values)
    rs = np.random.RandomState(args.seed)
    tr_idx = (rs.choice(tr_all, size=min(args.train_sample, len(tr_all)),
                        replace=False) if args.train_sample else tr_all)
    va_idx = (rs.choice(va_all, size=min(args.valid_sample, len(va_all)),
                        replace=False) if args.valid_sample else va_all)
    tr_idx.sort()
    va_idx.sort()
    log(f"样本: 训练全集 {len(tr_all):,} -> 采样 {len(tr_idx):,}, "
        f"验证全集 {len(va_all):,} -> 采样 {len(va_idx):,}")

    names = {}

    def put(name, arr, cols):
        save_group(name, arr, cols)
        names[name] = list(cols)

    tech_cols = [c for c in X.columns if c not in MKT_COLS]
    put("tech", to_matrix(X, tech_cols), tech_cols)
    if args.tech_norm != "none":
        log(f"tech 组横截面处理: {args.tech_norm} ...")
        ind_map = (load_industry_map()
                   if args.tech_norm == "industry" else None)
        tech_norm_mat = neutralize_features(
            to_matrix(X, tech_cols), dates, codes,
            ind_map=ind_map, mode=args.tech_norm)
        put("tech_norm", tech_norm_mat, [f"{c}_n" for c in tech_cols])
        del tech_norm_mat
        gc.collect()
    mkt_base = to_matrix(X, MKT_COLS)
    warm = ((year >= int(DATA_START[:4])) & X["ret60"].notna()).values
    regime = None
    if args.regime != "none":
        regime = build_regime_series(dates, mkt_base, args.regime)
        if args.exec == "topen":
            # topen口径: 9:25时t收盘未知, 状态用上一交易日的 (首日默认开仓)
            regime = regime.shift(1).fillna(True)
        log(f"regime({args.regime}): risk-on {int(regime.sum())}/{len(regime)} 天")
    del X
    gc.collect()

    if "pattern" in groups or "pattern_norm" in groups:
        pat = build_pattern_features(df)
        put("pattern", to_matrix(pat), list(pat.columns))
        if args.pattern_norm != "none":
            log(f"pattern 组横截面处理: {args.pattern_norm} ...")
            ind_map = (load_industry_map()
                       if args.pattern_norm == "industry" else None)
            pat_norm_mat = neutralize_features(
                to_matrix(pat), dates, codes,
                ind_map=ind_map, mode=args.pattern_norm)
            put("pattern_norm", pat_norm_mat,
                [f"{c}_n" for c in pat.columns])
            del pat_norm_mat
            gc.collect()
        del pat
        gc.collect()

    y_all = y.to_numpy(dtype=np.float32)
    nr_all = next_ret.to_numpy(dtype=np.float32)
    bars_oc = np.stack([df["open"].to_numpy(dtype=np.float64),
                        df["close"].to_numpy(dtype=np.float64)], axis=1)
    bars_amt = (df["vol"].to_numpy(dtype=np.float64)
                * df["close"].to_numpy(dtype=np.float64))
    liquidity = None
    if args.liquidity:
        liquidity = {
            "min_amount": args.liq_min_amount,
            "max_turnover_pct": args.liq_max_turnover_pct,
            "impact_bps": args.liq_impact_bps,
        }
        log(f"Top5 流动性约束: 日成交额 >= {args.liq_min_amount:,.0f} 元, "
            f"容量 <= {args.liq_max_turnover_pct:.1%} 日成交额, "
            f"冲击成本 +{args.liq_impact_bps:.1f} bps")
    labels = None
    if args.mode in ("horizon", "walkforward"):
        horizons = [int(v) for v in args.horizons.split(",") if v.strip()]
        if args.exec == "topen":
            log("执行口径: 盘中信号锚定开盘价 (信号日=买入日, open[t]执行)")
            labels = build_multilabels(df, compute_tradable_open(df),
                                       tuple(horizons), exec_lag=0)
        else:
            labels = build_multilabels(df, tradable, tuple(horizons),
                                       exec_lag=1)
        log(f"多周期标签: {sorted(labels)}")
        for h, (R, Y, M) in labels.items():
            log(f"  h={h}: 训练 {(M & (year <= int(TRAIN_END[:4])).values).sum():,} / "
                f"验证 {(M & ~(year <= int(TRAIN_END[:4])).values).sum():,}")
    del df, next_ret, base
    gc.collect()

    if "market" in groups:
        mkt, mkt_cols = build_market_features(dates)
        mkt_full = reindex_dates(mkt.to_numpy(np.float32), mkt.index.values, dates)
        del mkt
        gc.collect()
        put("market", np.hstack([mkt_base, mkt_full]), MKT_COLS + mkt_cols)
        del mkt_full
        gc.collect()
    del mkt_base
    gc.collect()
    if "industry" in groups:
        ind = build_industry_features(dates, codes)
        put("industry", ind.to_numpy(np.float32), INDUSTRY_FEATS)
        del ind
        gc.collect()
    if "fund" in groups:
        fund_mats, fund_names = build_fund_features(dates, codes)
        fund_mat = np.hstack(fund_mats)
        del fund_mats
        gc.collect()
        if args.fund_norm != "none":
            ind_map = (load_industry_map()
                       if args.fund_norm == "industry" else None)
            log(f"基本面横截面标准化: {args.fund_norm} ...")
            fund_mat = neutralize_fund_features(fund_mat, dates, codes,
                                                args.fund_norm, ind_map)
        put("fund", fund_mat, fund_names)
        del fund_mat
        gc.collect()
    for g, builder in (("flow", build_flow_features),
                       ("margin", build_margin_features),
                       ("lhb", build_lhb_features),
                       ("events", build_event_features)):
        if g not in groups:
            continue
        mat, fn = builder(dates, codes)
        if mat is None or len(fn) == 0:
            log(f"警告: {g} 数据表为空, 跳过该组")
            groups.remove(g)
        else:
            finite = float(np.isfinite(
                np.asarray(mat, dtype=np.float32)).mean())
            if finite < 0.01:
                log(f"警告: {g} 有效值仅 {finite:.2%}, 基本无信息量, 跳过该组")
                groups.remove(g)
                del mat
                gc.collect()
            else:
                put(g, mat, fn)

    if args.exec == "topen":
        # topen口径: 9:25时t收盘未知, 所有特征组按股前移一日
        # (t行用t-1收盘及以前), 再补上当日开盘缺口gap_t (tech组末列)
        log("执行口径topen: 特征按股前移一日 + 补gap_t ...")
        icode = pd.factorize(codes)[0]
        for g in list(names):
            p = os.path.join(OUT_DIR, f"feat_{g}.npy")
            mat = np.load(p)
            mat = shift_group_for_open_exec(mat, icode)
            if g == "tech":
                mat = np.hstack([mat, np.asarray(gap_t, dtype=np.float32)
                                 [:, None]])
                names[g] = list(names[g]) + ["gap_t"]
                with open(os.path.join(OUT_DIR, "feat_tech.cols.json"), "w",
                          encoding="utf-8") as f:
                    json.dump(list(names[g]), f, ensure_ascii=False)
                log(f"特征组 tech: {len(names[g])} 列 (含gap_t)")
            np.save(p, mat)
            del mat
            gc.collect()
        del icode
        gc.collect()
    del gap_t
    gc.collect()

    y_tr, y_va = y_all[tr_idx], y_all[va_idx]
    age = (pd.Timestamp(TRAIN_END) - pd.to_datetime(dates[tr_idx])).days.values
    w_tr = (0.5 ** (age / 756.0)).astype(np.float32)
    tr_sub = tr_idx
    if args.train_auc_sample and len(tr_idx) > args.train_auc_sample:
        tr_sub = np.sort(np.random.RandomState(args.seed + 1).choice(
            tr_idx, size=args.train_auc_sample, replace=False))

    rankic_mode = args.objective == "rankic"
    params = {
        "objective": "binary", "metric": ["auc", "binary_logloss"],
        "learning_rate": args.lr, "num_leaves": args.leaves,
        "min_data_in_leaf": 1000, "feature_fraction": 0.6,
        "bagging_fraction": 0.7, "bagging_freq": 1, "lambda_l2": 20.0,
        "max_bin": 127, "num_threads": args.threads, "verbosity": -1,
        "seed": args.seed, "feature_fraction_seed": args.seed,
        "bagging_seed": args.seed,
    }
    if rankic_mode:
        params["objective"] = "binary"  # placeholder, 实际在训练时替换为函数
        params["metric"] = "None"
        log("训练目标: rankic (LambdaRankIC 直接优化逐日Rank IC)")
    thermal_cb = ThermalCallback(limit=args.temp_limit,
                                 resume=args.temp_resume,
                                 throttle=args.throttle, every=5)

    if args.mode == "horizon":
        run_horizon_mode(args, params, thermal_cb, groups, names, labels,
                         warm, year, dates, codes, nr_all, bars_oc, t0,
                         regime=regime, bars_amt=bars_amt,
                         liquidity=liquidity)
        return
    if args.mode == "walkforward":
        run_walkforward_mode(args, params, thermal_cb, groups, names, labels,
                             warm, year, dates, codes, nr_all, bars_oc, t0,
                             regime=regime, bars_amt=bars_amt,
                             liquidity=liquidity)
        return

    configs = make_configs(args.mode, groups)
    log(f"消融配置 {len(configs)} 个: {['+'.join(c) for c in configs]}")

    rows, imps = [], {}
    for cfg in configs:
        label = "+".join(cfg)
        feat_names = [nm for g in cfg for nm in names[g]]
        Xtr = build_config(names, cfg, tr_idx)
        log(f"[{label}] 特征 {Xtr.shape[1]}, 训练 {len(Xtr):,} 行 ...")
        dtr = lgb.Dataset(Xtr, label=y_tr, weight=w_tr,
                          feature_name=feat_names)
        del Xtr
        gc.collect()
        Xva = build_config(names, cfg, va_idx)
        dva = lgb.Dataset(Xva, label=y_va, reference=dtr)
        del Xva
        gc.collect()
        model = lgb.train(
            params, dtr, num_boost_round=args.rounds, valid_sets=[dva],
            callbacks=[lgb.early_stopping(150, first_metric_only=True,
                                          verbose=False),
                       thermal_cb,
                       lgb.log_evaluation(200)])
        Xva = build_config(names, cfg, va_idx)
        p_va = model.predict(Xva, num_iteration=model.best_iteration)
        del Xva
        gc.collect()
        Xtr = build_config(names, cfg, tr_sub)
        p_tr = model.predict(Xtr, num_iteration=model.best_iteration)
        del Xtr
        gc.collect()
        row = {
            "config": label, "n_feat": len(feat_names),
            "best_iter": model.best_iteration,
            "train_auc": roc_auc_score(y_all[tr_sub], p_tr),
            "valid_auc": roc_auc_score(y_va, p_va),
            "valid_logloss": log_loss(y_va, p_va),
            "valid_acc": accuracy_score(y_va, p_va > 0.5),
            "base_rate": float(y_va.mean()),
        }
        imp = pd.DataFrame({
            "feature": model.feature_name(),
            "gain": model.feature_importance("gain"),
            "split": model.feature_importance("split"),
        }).sort_values("gain", ascending=False)
        imp["config"] = label
        imps[label] = imp

        if not args.no_backtest:
            bt = pd.DataFrame({
                "date": dates[va_idx], "code": codes[va_idx],
                "prob": p_va, "next_ret": nr_all[va_idx], "y": y_va})
            s, _, _, _, ic, icir = run_period(bt, label)
            row["ic"] = ic
            row["ic_ir"] = icir
            for strat in ("Top20", "Top50", "P>=0.55"):
                m = s[s["strategy"] == strat]
                if not m.empty:
                    key = strat.replace(">=", "").replace(".", "")
                    row[f"{key}_ann_net"] = float(m.iloc[0]["ann_net"])
                    row[f"{key}_sharpe"] = float(m.iloc[0]["sharpe"])
            row["mkt_ann_net"] = float(
                s[s["strategy"] == "市场等权"].iloc[0]["ann_net"])
            del bt, s

        rows.append(row)
        log(f"[{label}] 完成: valid_auc {row['valid_auc']:.4f}, "
            f"logloss {row['valid_logloss']:.4f}, best_iter {row['best_iter']}, "
            f"用时 {(time.time() - t0) / 60:.1f} min")
        if cfg == tuple(groups):
            model.save_model(os.path.join(OUT_DIR, "lgbm_all.txt"))
        del dtr, dva, model, p_tr, p_va
        gc.collect()

    res = pd.DataFrame(rows).sort_values("valid_auc", ascending=False)
    res.to_csv(os.path.join(OUT_DIR, "ablation_results.csv"), index=False)
    pd.concat(imps.values(), ignore_index=True).to_csv(
        os.path.join(OUT_DIR, "ablation_importance.csv"), index=False)
    print("\n================ 消融结果 (按验证集AUC排序) ================")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    with open(os.path.join(OUT_DIR, "run_config.json"), "w",
              encoding="utf-8") as f:
        json.dump({"mode": args.mode, "groups": groups,
                   "configs": ["+".join(c) for c in configs],
                   "params": params, "train_rows": len(tr_idx),
                   "valid_rows": len(va_idx),
                   "train_auc_rows": len(tr_sub), "limit": args.limit},
                  f, ensure_ascii=False, indent=2)
    if not args.keep_features:
        for g in names:
            for suffix in (".npy", ".cols.json"):
                try:
                    os.remove(os.path.join(OUT_DIR, f"feat_{g}{suffix}"))
                except OSError:
                    pass
    print(f"\n输出目录: {OUT_DIR}")
    log(f"全部完成, 总用时 {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
