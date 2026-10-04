#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LightGBM 上涨概率模型 + 训练集/验证集分段回测

- 特征: 价量技术指标 + 市场环境 (仅用 t 日及以前数据)
- 标签: 次日 开盘->收盘 涨跌 (close[t+1] > open[t+1]) -> 二分类, 输出 P(上涨)
- 切分: 训练集 2018-01-01 ~ 2023-12-31, 验证集 2024-01-01 之后 (纯样本外)
- 回测: t 日收盘出信号, t+1 开盘买入, t+1 收盘卖出 (等权, TopK/概率阈值, 含换手成本)
  可交易性过滤: 次日开盘一字/涨停封板无法买入、停牌超过7个自然日的样本剔除

用法:
    python3 lgbm_train_backtest.py                # 全量
    python3 lgbm_train_backtest.py --limit 300 --rounds 300   # 快速冒烟测试
"""
import argparse
import gc
import json
import os
import sqlite3
import time
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "stock_cache.db")
OUT_DIR = os.path.join(HERE, "lgbm_output")

LOAD_START = "2016-01-01"   # 特征预热起点
DATA_START = "2018-01-01"   # 样本起点
TRAIN_END = "2023-12-31"    # 训练集截止 (含)
COST = 0.0015               # 单次换仓成本 (买+卖, 万15)
MIN_HOLD = 5                # 阈值策略当日入选不足N只则空仓
SEED = 42

STOCK_COND = ("(s.code GLOB 'sh60[0135]*' OR s.code GLOB 'sh68[89]*' "
              "OR s.code GLOB 'sz00[0123]*' OR s.code GLOB 'sz30[01]*' "
              "OR s.code GLOB 'bj[489]*')")

FEATURES = [
    "ret1", "ret5", "ret10", "ret20", "ret60",
    "ma5r", "ma10r", "ma20r", "ma60r", "ma5_20", "ma20_60",
    "vol20", "vol60", "volr", "rsi6", "rsi14",
    "macd_dif", "macd_dea", "macd_hist",
    "kdj_k", "kdj_d", "kdj_j",
    "bollpos", "hilo20", "atr_n",
    "vr5", "vr20", "logvol", "amp", "gap", "cpos", "updays20",
    "mkt1", "mkt5", "mkt20", "breadth5",
    "r_ret5", "r_ret20", "r_ret60", "r_ma20r", "r_rsi14",
    "r_vr20", "r_vol20", "r_logvol",
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- 数据加载
def calc_limit(df, con):
    """逐行涨跌停幅度: 主板10%, 创业板/科创板20%, 北交所30%, ST主板5%"""
    nm = pd.read_sql_query("SELECT code, name FROM stocks", con)
    st_map = dict(zip(nm["code"], nm["name"].fillna("").str.contains("ST", case=False)))
    cats = df["code"].cat.categories
    lim_c = np.full(len(cats), 0.10, dtype=np.float64)
    for i, cc in enumerate(cats):
        s = str(cc)
        if s.startswith(("sh688", "sh689", "sz300", "sz301")):
            lim_c[i] = 0.20
        elif s.startswith("bj"):
            lim_c[i] = 0.30
        elif st_map.get(s, False):
            lim_c[i] = 0.05
    return lim_c[df["code"].cat.codes.values]


def load_bars(limit=None, seed=SEED):
    con = sqlite3.connect(DB)
    q = f"""SELECT b.code, b.date, b.open, b.high, b.low, b.close, b.vol
            FROM daily_bars b JOIN stocks s ON s.code = b.code
            WHERE b.date >= ? AND {STOCK_COND}"""
    params = [LOAD_START]
    if limit:
        all_codes = pd.read_sql_query(
            f"SELECT code FROM stocks s WHERE {STOCK_COND}", con)["code"].tolist()
        rs = np.random.RandomState(seed)
        codes = sorted(rs.choice(all_codes, size=min(limit, len(all_codes)),
                                 replace=False).tolist())
        q += " AND b.code IN (%s)" % ",".join(["?"] * len(codes))
        params += codes
    q += " ORDER BY b.code, b.date"
    log("读取行情数据 ...")
    df = pd.read_sql_query(q, con, params=params)
    df["code"] = df["code"].astype("category")
    df["lim"] = calc_limit(df, con).astype(np.float32)
    con.close()
    for c in ["open", "high", "low", "close", "vol"]:
        df[c] = df[c].astype("float64")
    log(f"行情数据: {len(df):,} 行, {df['code'].nunique()} 只股票")
    return df


def _aligned(s):
    if isinstance(s.index, pd.MultiIndex):
        return s.reset_index(level=0, drop=True)
    return s


def _gr(gb, col, w, how):
    return _aligned(getattr(gb[col].rolling(w), how)())


def _ewm(gb, col, **kw):
    return gb[col].transform(lambda s: s.ewm(**kw).mean())


# ---------------------------------------------------------------- 特征工程
def _rm(series, codes, w):
    return _aligned(series.groupby(codes, sort=False).rolling(w).mean())


def _compute_feats(df):
    """所有特征以 float32 存储; 中间量在本函数返回后统一释放"""
    gb = df.groupby("code", sort=False)
    codes = df["code"].cat.codes
    dts = df["date"]
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["vol"]

    F = {}

    def put(name, s):
        F[name] = s.astype(np.float32)

    put("ret1", gb["close"].pct_change())
    put("ret5", gb["close"].pct_change(5))
    put("ret10", gb["close"].pct_change(10))
    put("ret20", gb["close"].pct_change(20))
    put("ret60", gb["close"].pct_change(60))

    ma5 = _gr(gb, "close", 5, "mean")
    ma10 = _gr(gb, "close", 10, "mean")
    ma20 = _gr(gb, "close", 20, "mean")
    ma60 = _gr(gb, "close", 60, "mean")
    put("ma5r", c / ma5 - 1.0)
    put("ma10r", c / ma10 - 1.0)
    put("ma20r", c / ma20 - 1.0)
    put("ma60r", c / ma60 - 1.0)
    put("ma5_20", ma5 / ma20 - 1.0)
    put("ma20_60", ma20 / ma60 - 1.0)
    del ma5, ma10, ma60

    std20 = _gr(gb, "close", 20, "std")
    std60 = _gr(gb, "close", 60, "std")
    put("vol20", std20)
    put("vol60", std60)
    put("volr", std20 / (std60 + 1e-9))
    del std60

    delta = gb["close"].diff()
    up = delta.clip(lower=0.0)
    dn = (-delta).clip(lower=0.0)
    for n in (6, 14):
        fu = _rm(up, codes, n)
        fd = _rm(dn, codes, n)
        put(f"rsi{n}", 100.0 * fu / (fu + fd + 1e-9))
    del delta, up, dn

    ema12 = _ewm(gb, "close", span=12, adjust=False)
    ema26 = _ewm(gb, "close", span=26, adjust=False)
    dif = ema12 - ema26
    dea = dif.groupby(codes, sort=False).transform(
        lambda s: s.ewm(span=9, adjust=False).mean())
    put("macd_dif", dif / c)
    put("macd_dea", dea / c)
    put("macd_hist", (dif - dea) * 2.0 / c)
    del ema12, ema26, dif, dea

    llv = _gr(gb, "low", 9, "min")
    hhv = _gr(gb, "high", 9, "max")
    rsv = (c - llv) / (hhv - llv + 1e-9) * 100.0
    k = rsv.groupby(codes, sort=False).transform(
        lambda s: s.ewm(com=2, adjust=False).mean())
    dd = k.groupby(codes, sort=False).transform(
        lambda s: s.ewm(com=2, adjust=False).mean())
    put("kdj_k", k)
    put("kdj_d", dd)
    put("kdj_j", 3.0 * k - 2.0 * dd)
    del llv, hhv, rsv, k, dd

    cmin20 = _gr(gb, "close", 20, "min")
    cmax20 = _gr(gb, "close", 20, "max")
    put("bollpos", (c - ma20) / (2.0 * std20 + 1e-9))
    put("hilo20", (c - cmin20) / (cmax20 - cmin20 + 1e-9))
    del ma20, std20, cmin20, cmax20

    prev_close = gb["close"].shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_close).abs(),
                    (l - prev_close).abs()], axis=1).max(axis=1)
    put("atr_n", _rm(tr, codes, 14) / c)
    put("amp", (h - l) / (prev_close + 1e-9))
    put("gap", df["open"] / (prev_close + 1e-9) - 1.0)
    put("cpos", (c - l) / (h - l + 1e-9))
    del tr, prev_close

    voll5 = _gr(gb, "vol", 5, "mean")
    voll20 = _gr(gb, "vol", 20, "mean")
    put("vr5", v / (voll5 + 1e-9))
    put("vr20", v / (voll20 + 1e-9))
    put("logvol", np.log1p(v))
    del voll5, voll20

    put("updays20", _rm((F["ret1"] > 0).astype(np.float32), codes, 20))

    put("mkt1", F["ret1"].groupby(dts).transform("mean"))
    put("mkt5", F["ret5"].groupby(dts).transform("mean"))
    put("mkt20", F["ret20"].groupby(dts).transform("mean"))
    put("breadth5", (F["ret1"] > 0).astype(np.float32).groupby(dts).transform("mean"))

    # 横截面排名特征 (当日全市场分位, 抗市场风格漂移)
    for k_ in ["ret5", "ret20", "ret60", "ma20r", "rsi14",
               "vr20", "vol20", "logvol"]:
        F["r_" + k_] = F[k_].groupby(dts).rank(pct=True).astype(np.float32)
    return F


def build_features(df):
    log("构建特征 ...")
    F = _compute_feats(df)
    log("组装特征矩阵 ...")
    X = pd.DataFrame(index=df.index, dtype=np.float32)
    for k_, val in F.items():
        X[k_] = val
    del F
    gc.collect()

    # ---- 标签: 次日 开盘->收盘 收益 (t日信号, t+1开盘买, t+1收盘卖) ----
    gb = df.groupby("code", sort=False)
    c = df["close"]
    lim = df["lim"].values.astype(np.float64)
    open1 = gb["open"].shift(-1)
    close1 = gb["close"].shift(-1)
    next_ret = (close1 / open1 - 1.0).astype(np.float32)
    y = (next_ret > 0).astype(np.int8)

    # 组内下一行间隔(交易日数), 用于剔除长期停牌
    date_idx = pd.factorize(df["date"], sort=True)[0]
    cd = df["code"].cat.codes.values
    same_next = np.empty(len(df), dtype=bool)
    same_next[:-1] = cd[:-1] == cd[1:]
    same_next[-1] = False
    gap = np.full(len(df), 999, dtype=np.int32)
    gap[:-1] = date_idx[1:] - date_idx[:-1]
    gap[~same_next] = 999

    tradable = pd.Series(
        next_ret.notna().values & (gap <= 5)
        & (open1.values < c.values * (1.0 + lim - 0.005)), index=df.index)
    bad = (~tradable) & next_ret.notna()
    log(f"可交易样本占比 {tradable.mean():.3f}, 剔除(次日开盘涨停/长停牌) {bad.sum():,} 行")
    log(f"特征矩阵: {X.shape[0]:,} x {X.shape[1]}")
    return X, next_ret, y, tradable


# ---------------------------------------------------------------- 回测
def turnover_series(sel, dates):
    sets = {d: set(g) for d, g in sel.groupby("date", observed=True)["code"]}
    out = np.zeros(len(dates))
    prev = set()
    for i, d in enumerate(dates):
        cur = sets.get(d, set())
        if cur:
            out[i] = 1.0 - (len(cur & prev) / len(cur)) if prev else 1.0
        prev = cur
    return pd.Series(out, index=dates)


def perf(daily, bench, turnover, hold=None):
    daily = daily.fillna(0.0)
    turnover = turnover.fillna(0.0)
    net = daily - COST * turnover
    eq = (1.0 + net).cumprod()
    n = len(net)
    if n == 0:
        return {}
    total = eq.iloc[-1] - 1.0
    ann = (1.0 + total) ** (252.0 / n) - 1.0
    std = net.std()
    sharpe = float(net.mean() / std * np.sqrt(252)) if std > 0 else 0.0
    mdd = float((eq / eq.cummax() - 1.0).min())
    b = bench.reindex(net.index).fillna(0.0)
    b_ann = (1.0 + b).prod() ** (252.0 / n) - 1.0
    if hold is not None:
        active = hold.reindex(net.index).fillna(0) > 0
        win = float((net[active] > 0).mean()) if active.any() else 0.0
        avg_hold = float(hold.mean())
    else:
        win = float((net > 0).mean())
        avg_hold = 0.0
    return {
        "days": n,
        "ann_gross": float((1.0 + daily).prod() ** (252.0 / n) - 1.0),
        "ann_net": float(ann),
        "excess_ann": float(ann - b_ann),
        "sharpe": sharpe,
        "mdd": mdd,
        "win_daily": win,
        "avg_turnover": float(turnover.mean()),
        "avg_hold": avg_hold,
    }


def daily_ic(bt):
    sub = bt[["date", "prob", "next_ret"]].copy()
    sub["rp"] = sub.groupby("date")["prob"].rank()
    sub["rr"] = sub.groupby("date")["next_ret"].rank()

    def _c(d):
        if len(d) < 50 or d["rp"].std() == 0 or d["rr"].std() == 0:
            return np.nan
        return float(np.corrcoef(d["rp"], d["rr"])[0, 1])

    return sub.groupby("date").apply(_c).dropna()


def run_period(bt_orig, label, topk=(5, 10, 20, 50), thrs=(0.50, 0.55, 0.60)):
    bt = bt_orig.dropna(subset=["prob", "next_ret"]).copy()
    dates = np.sort(bt["date"].unique())
    bench = bt.groupby("date")["next_ret"].mean()
    bt.sort_values(["date", "prob"], ascending=[True, False], inplace=True)
    bt["rk"] = bt.groupby("date").cumcount()

    rows, curves = [], []

    ic = daily_ic(bt)
    ic_mean = ic.mean()
    ic_ir = ic_mean / (ic.std() + 1e-12)

    for K in topk:
        sel = bt[bt["rk"] < K]
        daily = sel.groupby("date")["next_ret"].mean().reindex(dates).fillna(0.0)
        hold = sel.groupby("date").size().reindex(dates).fillna(0.0)
        to = turnover_series(sel, dates)
        m = perf(daily, bench, to, hold)
        m.update(period=label, strategy=f"Top{K}")
        rows.append(m)
        curves.append(pd.DataFrame({"date": dates, "period": label,
                                    "strategy": f"Top{K}",
                                    "ret_net": (daily.values - COST * to.values),
                                    "equity": (1.0 + daily.values - COST * to.values).cumprod()}))

    for thr in thrs:
        sel = bt[bt["prob"] >= thr]
        cnt = sel.groupby("date").size()
        sel = sel[~sel["date"].isin(set(cnt[cnt < MIN_HOLD].index))]
        daily = sel.groupby("date")["next_ret"].mean().reindex(dates).fillna(0.0)
        hold = sel.groupby("date").size().reindex(dates).fillna(0.0)
        to = turnover_series(sel, dates)
        m = perf(daily, bench, to, hold)
        m.update(period=label, strategy=f"P>={thr:.2f}")
        rows.append(m)
        curves.append(pd.DataFrame({"date": dates, "period": label,
                                    "strategy": f"P>={thr:.2f}",
                                    "ret_net": (daily.values - COST * to.values),
                                    "equity": (1.0 + daily.values - COST * to.values).cumprod()}))

    mb = perf(bench, bench, pd.Series(0.0, index=dates),
              bt.groupby("date").size().reindex(dates).fillna(0.0))
    mb.update(period=label, strategy="市场等权")
    rows.append(mb)
    bench_eq = (1.0 + bench.reindex(dates).fillna(0.0)).cumprod()
    curves.append(pd.DataFrame({"date": dates, "period": label, "strategy": "市场等权",
                                "ret_net": bench.reindex(dates).fillna(0.0).values,
                                "equity": bench_eq.values}))

    # 概率分档
    bt["bucket"] = pd.qcut(bt["prob"], 10, labels=False, duplicates="drop")
    dec = bt.groupby("bucket")["next_ret"].agg(["mean", "count"]).reset_index()
    dec["period"] = label

    # 分年度 AUC
    bt["year"] = bt["date"].str.slice(0, 4)
    aucs = []
    for y_, g in bt.groupby("year"):
        if g["y"].nunique() > 1:
            aucs.append({"period": label, "year": y_,
                         "auc": roc_auc_score(g["y"], g["prob"]),
                         "acc": accuracy_score(g["y"], g["prob"] > 0.5),
                         "n": len(g)})

    daily_all = pd.concat(curves, ignore_index=True)
    return pd.DataFrame(rows), daily_all, dec, pd.DataFrame(aucs), ic_mean, ic_ir


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="随机抽取N只股票(冒烟测试)")
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--leaves", type=int, default=31)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    df = load_bars(args.limit or None, args.seed)
    X, next_ret, y, tradable = build_features(df)

    year = df["date"].str.slice(0, 4).astype(int)
    date = df["date"]
    code = df["code"]
    base = ((year >= int(DATA_START[:4])) & next_ret.notna() & tradable
            & X["ret60"].notna())
    m_tr = base & (year <= int(TRAIN_END[:4]))
    m_va = base & ~m_tr

    log(f"样本: 训练 {int(m_tr.sum()):,} / 验证 {int(m_va.sum()):,}")
    Xtr, Xva = X.loc[m_tr], X.loc[m_va]
    ytr, yva = y[m_tr].values, y[m_va].values
    del X
    gc.collect()

    params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "learning_rate": args.lr,
        "num_leaves": args.leaves,
        "min_data_in_leaf": 1000,
        "feature_fraction": 0.6,
        "bagging_fraction": 0.7,
        "bagging_freq": 1,
        "lambda_l2": 20.0,
        "max_bin": 127,
        "num_threads": os.cpu_count(),
        "verbosity": -1,
        "seed": SEED,
        "feature_fraction_seed": SEED,
        "bagging_seed": SEED,
    }
    age = (pd.Timestamp(TRAIN_END) - pd.to_datetime(date[m_tr])).dt.days.values
    wtr = 0.5 ** (age / 756.0)  # 时间衰减: 半衰期3年
    log("开始训练 LightGBM ...")
    dtr = lgb.Dataset(Xtr, label=ytr, weight=wtr, feature_name=list(Xtr.columns))
    dva = lgb.Dataset(Xva, label=yva, reference=dtr)
    model = lgb.train(params, dtr, num_boost_round=args.rounds,
                      valid_sets=[dva],
                      callbacks=[lgb.early_stopping(150, first_metric_only=True,
                                                     verbose=True),
                                 lgb.log_evaluation(100)])
    log(f"训练完成, best_iter={model.best_iteration}, 用时 {(time.time()-t0)/60:.1f} min")

    model.save_model(os.path.join(OUT_DIR, "lgbm_model.txt"))
    imp = pd.DataFrame({
        "feature": model.feature_name(),
        "gain": model.feature_importance("gain"),
        "split": model.feature_importance("split"),
    }).sort_values("gain", ascending=False)
    imp.to_csv(os.path.join(OUT_DIR, "feature_importance.csv"), index=False)

    ptr = model.predict(Xtr, num_iteration=model.best_iteration)
    pva = model.predict(Xva, num_iteration=model.best_iteration)

    evals = []
    for name, yy, pp in [("训练集", ytr, ptr), ("验证集", yva, pva)]:
        evals.append({"period": name, "n": len(yy),
                      "auc": roc_auc_score(yy, pp),
                      "logloss": log_loss(yy, pp),
                      "acc": accuracy_score(yy, pp > 0.5),
                      "base_rate": float(yy.mean())})
    ev = pd.DataFrame(evals)
    print("\n================ 模型评估 ================")
    print(ev.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # 回测
    bt_tr = pd.DataFrame({"date": date[m_tr].values, "code": code[m_tr].values,
                          "prob": ptr, "next_ret": next_ret[m_tr].values, "y": ytr})
    bt_va = pd.DataFrame({"date": date[m_va].values, "code": code[m_va].values,
                          "prob": pva, "next_ret": next_ret[m_va].values, "y": yva})

    log("回测 (训练集) ...")
    s_tr, c_tr, d_tr, a_tr, ic_tr, icir_tr = run_period(bt_tr, "训练集")
    log("回测 (验证集) ...")
    s_va, c_va, d_va, a_va, ic_va, icir_va = run_period(bt_va, "验证集")

    summary = pd.concat([s_tr, s_va], ignore_index=True)
    print("\n============ 回测汇总 (t+1开盘买/收盘卖, 等权, 成本万15, 剔除涨停开盘) ============")
    cols = ["strategy", "days", "avg_hold", "ann_gross", "ann_net",
            "excess_ann", "sharpe", "mdd", "win_daily", "avg_turnover"]
    for p in summary["period"].unique():
        print(f"\n--- {p} ---")
        sub = summary[summary["period"] == p]
        print(sub[cols].to_string(index=False,
              float_format=lambda v: f"{v:.4f}"))

    print("\n---------------- IC (RankIC) ----------------")
    print(f"训练集: IC均值 {ic_tr:.4f}, ICIR {icir_tr:.2f} (年化t≈{icir_tr*np.sqrt(s_tr.iloc[0]['days']):.1f})")
    print(f"验证集: IC均值 {ic_va:.4f}, ICIR {icir_va:.2f} (年化t≈{icir_va*np.sqrt(s_va.iloc[0]['days']):.1f})")

    print("\n---------------- 分年度 AUC ----------------")
    print(pd.concat([a_tr, a_va], ignore_index=True).to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n---------------- 概率分档平均次日收益 ----------------")
    for name, d in [("训练集", d_tr), ("验证集", d_va)]:
        print(f"--- {name} ---")
        print(d.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    summary.to_csv(os.path.join(OUT_DIR, "backtest_summary.csv"), index=False)
    pd.concat([c_tr, c_va], ignore_index=True).to_csv(
        os.path.join(OUT_DIR, "daily_equity.csv"), index=False)
    pd.concat([d_tr, d_va], ignore_index=True).to_csv(
        os.path.join(OUT_DIR, "prob_decile.csv"), index=False)
    pd.concat([a_tr, a_va], ignore_index=True).to_csv(
        os.path.join(OUT_DIR, "yearly_auc.csv"), index=False)
    bt_va.to_csv(os.path.join(OUT_DIR, "valid_predictions.csv.gz"),
                 index=False, compression="gzip")
    with open(os.path.join(OUT_DIR, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump({"params": params, "best_iter": model.best_iteration,
                   "features": list(Xtr.columns), "cost": COST, "limit": args.limit,
                   "label": "close[t+1] > open[t+1] (t+1开盘买, t+1收盘卖)",
                   "train": f"{DATA_START}~{TRAIN_END}",
                   "valid": "2024-01-01~"}, f,
                  ensure_ascii=False, indent=2)

    print(f"\n输出目录: {OUT_DIR}")
    for fn in ["lgbm_model.txt", "feature_importance.csv", "backtest_summary.csv",
               "daily_equity.csv", "prob_decile.csv", "yearly_auc.csv",
               "valid_predictions.csv.gz", "run_config.json"]:
        print("  -", fn)
    log(f"全部完成, 总用时 {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
