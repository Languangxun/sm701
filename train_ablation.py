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
    python3 train_ablation.py --train-sample 3000000         # 手动采样加速
"""
import argparse
import gc
import json
import os
import sqlite3
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from lgbm_train_backtest import (COST, DATA_START, SEED, TRAIN_END,
                                 _aligned, ThermalCallback, asof_align,
                                 build_features, daily_ic, load_bars, log,
                                 perf, run_period)
from features_extra import (build_event_features, build_flow_features,
                            build_lhb_features, build_margin_features)

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "stock_cache.db")
OUT_DIR = os.path.join(HERE, "ablation_output")

GROUPS = ["tech", "pattern", "market", "industry", "fund",
          "flow", "margin", "lhb", "events"]
DEFAULT_GROUPS = "tech,pattern,market,industry,fund"
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


# ---------------------------------------------------------------- 多周期标签
def build_multilabels(df, tradable, horizons=(1, 5, 10)):
    """多周期标签: R_h = close[t+h]/open[t+1]-1 (t+1开盘买入, 持有h天)

    mask_h: 入场可买(tradable, 与t+1规则一致) 且持有窗内无>5天停牌
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
        R[sel] = c[sel + h] / o[sel + 1] - 1.0
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


def run_hold_backtest(dates_all, codes_all, mask_rows, prob, bars_oc,
                      bench, h, topks=(10, 20, 50), label=""):
    """重叠持仓回测: 每个信号日t收盘选TopK, t+1开盘买入持有h天(t+h收盘卖)

    每日组合收益=在持vintage当日收益的等权平均; 成本按进/出分摊
    (每端COST/2, 与run_period的单程COST口径一致). 仍是T+1调仓, 无日内"""
    bt = pd.DataFrame({"date": dates_all[mask_rows],
                       "code": codes_all[mask_rows],
                       "prob": prob, "row": mask_rows})
    bt["rk"] = bt.groupby("date")["prob"].rank(ascending=False,
                                               method="first")
    o_full, c_full = bars_oc[:, 0], bars_oc[:, 1]
    rows_out, curves = [], []
    for K in topks:
        sel = bt[bt["rk"] <= K]
        if sel.empty:
            continue
        r0 = sel["row"].to_numpy()
        e, x = r0 + 1, r0 + h
        rr = np.empty((len(sel), h), dtype=np.float64)
        rr[:, 0] = c_full[e] / o_full[e] - 1.0
        for k in range(1, h):
            rr[:, k] = c_full[e + k] / c_full[e + k - 1] - 1.0
        dd = np.stack([dates_all[e[i]:e[i] + h] for i in range(len(sel))])
        long = pd.DataFrame({"date": dd.ravel(), "ret": rr.ravel()})
        g = long.groupby("date")["ret"].mean()
        act = long.groupby("date").size()
        ent = pd.Series(1, index=dates_all[e]).groupby(level=0).sum()
        ext = pd.Series(1, index=dates_all[x]).groupby(level=0).sum()
        to = (0.5 * (ent.add(ext, fill_value=0.0) / act)
              ).reindex(g.index).fillna(0.0)
        m = perf(g, bench.reindex(g.index).fillna(0.0), to, act)
        m.update(h=h, K=K, period=label,
                 strategy=f"Hold{h}Top{K}", days=len(g))
        rows_out.append(m)
        curves.append(pd.DataFrame(
            {"date": g.index, "h": h, "K": K,
             "ret_net": (g.values - COST * to.values),
             "equity": (1.0 + g.values - COST * to.values).cumprod()}))
    return (pd.DataFrame(rows_out),
            pd.concat(curves, ignore_index=True) if curves
            else pd.DataFrame())


def run_horizon_mode(args, params, thermal_cb, groups, names, labels,
                     warm, year, dates, codes, nr_all, bars_oc, t0):
    """多周期模式: 每个horizon独立训练一个二分类模型(特征=全部请求组)

    报告: valid AUC/logloss/acc + 全验证集逐日RankIC(对比T日实际收益)"""
    allg = tuple(groups)
    horizons = sorted(labels)
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
    rs = np.random.RandomState(args.seed)
    vb = np.flatnonzero(np.isfinite(nr_all) & warm
                        & ~(year <= int(TRAIN_END[:4])).values)
    bench = pd.Series(nr_all[vb], index=dates[vb]).groupby(level=0).mean()
    log(f"市场基准: {len(bench)} 个交易日")
    rows, hold_all, hold_curves = [], [], []
    for h in horizons:
        R, Y, M0 = labels[h]
        M = M0 & warm
        yv = Y.astype(np.float32)
        tr_all = np.flatnonzero(M & (year <= int(TRAIN_END[:4])).values)
        va_all = np.flatnonzero(M & ~(year <= int(TRAIN_END[:4])).values)
        tr_h = (np.sort(rs.choice(
            tr_all, size=min(args.train_sample, len(tr_all)),
            replace=False)) if args.train_sample else tr_all)
        va_h = (np.sort(rs.choice(
            va_all, size=min(args.valid_sample, len(va_all)),
            replace=False)) if args.valid_sample else va_all)
        age = (pd.Timestamp(TRAIN_END)
               - pd.to_datetime(dates[tr_h])).days.values
        w_h = (0.5 ** (age / 756.0)).astype(np.float32)

        def train_one(cfg, flabel):
            feat_names = [nm for g in cfg for nm in names[g]]
            log(f"[h={h}/{flabel}] 特征 {len(feat_names)}, "
                f"训练 {len(tr_h):,} / 验证 {len(va_h):,}")
            Xtr = build_config(names, cfg, tr_h)
            dtr = lgb.Dataset(Xtr, label=yv[tr_h], weight=w_h,
                              feature_name=feat_names)
            del Xtr
            gc.collect()
            Xva = build_config(names, cfg, va_h)
            dva = lgb.Dataset(Xva, label=yv[va_h], reference=dtr)
            del Xva
            gc.collect()
            model = lgb.train(
                params, dtr, num_boost_round=args.rounds, valid_sets=[dva],
                callbacks=[lgb.early_stopping(150, first_metric_only=True,
                                              verbose=False),
                           thermal_cb,
                           lgb.log_evaluation(200)])
            Xva = build_config(names, cfg, va_h)
            p_va = model.predict(Xva, num_iteration=model.best_iteration)
            del Xva
            gc.collect()
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
            if h == 1 and not args.no_backtest:
                bt2 = pd.DataFrame({
                    "date": dates[va_h], "code": codes[va_h], "prob": p_va,
                    "next_ret": R[va_h], "y": yv[va_h]})
                s, _, _, _, _, _ = run_period(bt2, f"h={h}/{flabel}")
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
                    dates, codes, va_h, p_va, bars_oc, bench, h,
                    topks=(10, 20, 50), label=f"h={h}/{flabel}")
                for _, m in hs.iterrows():
                    row[f"hold{m['K']}_ann"] = float(m["ann_net"])
                    row[f"hold{m['K']}_sharpe"] = float(m["sharpe"])
                    row[f"hold{m['K']}_mdd"] = float(m["mdd"])
                hold_all.append(hs)
                hold_curves.append(hc)
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
                   "limit": args.limit}, f, ensure_ascii=False, indent=2)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="随机抽取N只股票(冒烟)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--mode", default="full",
                    choices=["full", "single", "cum", "loo", "horizon"])
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
    args = ap.parse_args()

    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    for g in groups:
        if g not in GROUPS:
            raise SystemExit(f"未知特征组 {g}, 可选: {', '.join(GROUPS)}")

    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    df = load_bars(args.limit or None, args.seed)
    X, next_ret, y, tradable = build_features(df)
    year = df["date"].str.slice(0, 4).astype(int)
    dates = df["date"].values
    code_cat = df["code"].cat.categories.to_numpy()
    codes = code_cat[df["code"].cat.codes.to_numpy()]
    n = len(df)

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
    mkt_base = to_matrix(X, MKT_COLS)
    warm = ((year >= int(DATA_START[:4])) & X["ret60"].notna()).values
    del X
    gc.collect()

    if "pattern" in groups:
        pat = build_pattern_features(df)
        put("pattern", to_matrix(pat), list(pat.columns))
        del pat
        gc.collect()

    y_all = y.to_numpy(dtype=np.float32)
    nr_all = next_ret.to_numpy(dtype=np.float32)
    bars_oc = np.stack([df["open"].to_numpy(dtype=np.float64),
                        df["close"].to_numpy(dtype=np.float64)], axis=1)
    labels = None
    if args.mode == "horizon":
        horizons = [int(v) for v in args.horizons.split(",") if v.strip()]
        labels = build_multilabels(df, tradable, tuple(horizons))
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
        put("fund", np.hstack(fund_mats), fund_names)
        del fund_mats
        gc.collect()
    for g, builder in (("flow", build_flow_features),
                       ("margin", build_margin_features),
                       ("lhb", build_lhb_features),
                       ("events", build_event_features)):
        if g not in groups:
            continue
        mat, fn = builder(dates, codes)
        if mat is None:
            log(f"警告: {g} 数据表为空, 跳过该组")
            groups.remove(g)
        else:
            put(g, mat, fn)

    y_tr, y_va = y_all[tr_idx], y_all[va_idx]
    age = (pd.Timestamp(TRAIN_END) - pd.to_datetime(dates[tr_idx])).days.values
    w_tr = (0.5 ** (age / 756.0)).astype(np.float32)
    tr_sub = tr_idx
    if args.train_auc_sample and len(tr_idx) > args.train_auc_sample:
        tr_sub = np.sort(np.random.RandomState(args.seed + 1).choice(
            tr_idx, size=args.train_auc_sample, replace=False))

    params = {
        "objective": "binary", "metric": ["auc", "binary_logloss"],
        "learning_rate": args.lr, "num_leaves": args.leaves,
        "min_data_in_leaf": 1000, "feature_fraction": 0.6,
        "bagging_fraction": 0.7, "bagging_freq": 1, "lambda_l2": 20.0,
        "max_bin": 127, "num_threads": args.threads, "verbosity": -1,
        "seed": args.seed, "feature_fraction_seed": args.seed,
        "bagging_seed": args.seed,
    }
    thermal_cb = ThermalCallback(limit=args.temp_limit,
                                 resume=args.temp_resume,
                                 throttle=args.throttle, every=5)

    if args.mode == "horizon":
        run_horizon_mode(args, params, thermal_cb, groups, names, labels,
                         warm, year, dates, codes, nr_all, bars_oc, t0)
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
