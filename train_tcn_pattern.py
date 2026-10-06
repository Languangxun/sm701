#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TCN(序列) vs LGBM(标量) 形态消融对比

同一批训练/验证采样行、同一标签与切分下比较:
  - TCN:  每行取该股最近 lookback 天的特征序列 (tech/pattern/tech+pattern)
  - LGBM: 同一行的特征标量 (与 train_ablation 一致)
防未来函数: 序列只回看 t 日及以前 (因果卷积 + 股票起始位置截断)

用法:
    python3 train_tcn_pattern.py --limit 300 --epochs 1          # 冒烟
    python3 train_tcn_pattern.py                                  # 全量采样对比
    python3 train_tcn_pattern.py --configs pattern --device cuda
"""
import argparse
import os
import threading
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import lightgbm as lgb
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from lgbm_train_backtest import (DATA_START, SEED, TRAIN_END,
                                 ThermalCallback, build_features, load_bars,
                                 log, run_period, thermal_wait)
from lgbm_tcn_ensemble import (TCN, make_batch_fn, pair_lists, parse_ints,
                               standardize, stock_start_idx)
from train_ablation import MKT_COLS, build_pattern_features, to_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "tcn_output")
GROUPS = {"tech": "tech", "pattern": "pattern"}


def evaluate(model, make_batch, idx, batch, device):
    model.eval()
    out = np.empty(len(idx), dtype=np.float32)
    with torch.inference_mode():
        for i in range(0, len(idx), batch):
            xb, _, _ = make_batch(idx[i:i + batch])
            out[i:i + batch] = torch.sigmoid(
                model(xb.to(device))).float().cpu().numpy()
    return out


def train_tcn(args, vals, start, tr_idx, va_idx, y_all, w_all, n_feat, device):
    channels, dilations = pair_lists(parse_ints(args.channels),
                                     parse_ints(args.dilations))
    torch.manual_seed(args.seed)
    model = TCN(n_feat, channels, args.kernel, dilations,
                args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.wd)
    lossf = nn.BCEWithLogitsLoss(reduction="none")
    make_batch = make_batch_fn(vals, start, y_all, w_all, args.lookback)
    best_auc, best_state = -1.0, None
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = np.random.RandomState(args.seed * 1000 + ep).permutation(
            len(tr_idx))
        run_loss, seen = 0.0, 0
        t_ep = time.time()
        n_steps = (len(perm) + args.batch_size - 1) // args.batch_size
        for i in range(0, len(perm), args.batch_size):
            rows = tr_idx[perm[i:i + args.batch_size]]
            xb, yb, wb = make_batch(rows)
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = (lossf(model(xb), yb) * wb).sum() / wb.sum().clamp_min(1e-9)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run_loss += loss.item() * len(rows)
            seen += len(rows)
            if args.throttle > 0:
                time.sleep(args.throttle)
            step = i // args.batch_size
            if step % 20 == 0:
                thermal_wait(args.temp_limit, args.temp_resume)
            if step % args.log_every == 0 or step == n_steps - 1:
                log(f"    epoch {ep} step {step}/{n_steps} "
                    f"loss {run_loss / max(seen, 1):.5f} "
                    f"({seen:,}/{len(perm):,})")
        p_va = evaluate(model, make_batch, va_idx, 8192, device)
        auc = roc_auc_score(y_all[va_idx], p_va)
        log(f"    epoch {ep}: loss {run_loss / max(seen, 1):.5f}, "
            f"val_auc {auc:.4f}, {(time.time() - t_ep) / 60:.1f} min")
        if auc > best_auc:
            best_auc, best_state = auc, {
                k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_auc, make_batch


SHOW_COLS = ["model", "config", "n_feat", "valid_auc", "valid_logloss",
             "valid_acc", "ic", "ic_ir", "Top20_ann_net", "Top20_sharpe",
             "P055_ann_net", "mkt_ann_net"]

BACK_COLS = ["strategy", "days", "avg_hold", "ann_gross", "ann_net",
             "excess_ann", "sharpe", "mdd", "win_daily", "avg_turnover"]


def write_report(res, sums):
    path = os.path.join(OUT_DIR, "backtest_report.md")
    lines = [f"# TCN(序列) vs LGBM(标量) 回测报告 "
             f"{time.strftime('%Y-%m-%d %H:%M')}", "",
             f"- 样本: 训练/验证为同一批采样行, lookback 见 run_config",
             f"- 策略: t+1 开盘买/收盘卖, 等权, 成本万15, 剔除涨停开盘", "",
             "## 主对比", ""]
    lines.append(res.reindex(columns=SHOW_COLS).to_markdown(
        index=False, floatfmt=".4f"))
    all_back = []
    for cfg, model, g in sums:
        all_back.append(g.assign(config=cfg, model=model))
    if all_back:
        full = pd.concat(all_back, ignore_index=True)
        full.to_csv(os.path.join(OUT_DIR, "backtest_summary.csv"), index=False)
        lines += ["", "## 分策略回测明细", ""]
        for (cfg, model), g in full.groupby(["config", "model"],
                                            sort=False):
            lines += [f"### {cfg} - {model}", ""]
            lines.append(g[BACK_COLS].to_markdown(index=False,
                                                  floatfmt=".4f"))
            lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"回测报告: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--configs", default="pattern,tech,tech+pattern")
    ap.add_argument("--train-sample", type=int, default=1_000_000)
    ap.add_argument("--valid-sample", type=int, default=300_000)
    ap.add_argument("--lookback", type=int, default=20)
    ap.add_argument("--channels", default="64,64,64,64")
    ap.add_argument("--dilations", default="1,2,4,8")
    ap.add_argument("--kernel", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--threads", type=int, default=4,
                    help="CPU线程数(LGBM/张量准备)")
    ap.add_argument("--throttle", type=float, default=0.05,
                    help="每批训练后休眠秒数, 降低发热")
    ap.add_argument("--temp-limit", type=float, default=92.0)
    ap.add_argument("--temp-resume", type=float, default=85.0)
    ap.add_argument("--log-every", type=int, default=200,
                    help="每N批打印一次进度")
    ap.add_argument("--no-parallel", action="store_true",
                    help="关闭 LGBM(CPU) 与 TCN(GPU) 并行")
    ap.add_argument("--skip-lgbm", action="store_true",
                    help="不跑LGBM基线(消融里已有全量结果), 只训练TCN")
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    log(f"设备 {device}")
    torch.set_num_threads(args.threads)
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    df = load_bars(args.limit or None, args.seed)
    X, next_ret, y, tradable = build_features(df)
    year = df["date"].str.slice(0, 4).astype(int)
    dates = df["date"].values
    code_cat = df["code"].cat.categories.to_numpy()
    codes = code_cat[df["code"].cat.codes.to_numpy()]

    base = ((year >= int(DATA_START[:4])) & next_ret.notna() & tradable
            & X["ret60"].notna()).values
    tr_all = np.flatnonzero(base & (year <= int(TRAIN_END[:4])).values)
    va_all = np.flatnonzero(base & ~(year <= int(TRAIN_END[:4])).values)
    rs = np.random.RandomState(args.seed)
    tr_idx = np.sort(rs.choice(tr_all, size=min(args.train_sample, len(tr_all)),
                               replace=False))
    va_idx = np.sort(rs.choice(va_all, size=min(args.valid_sample, len(va_all)),
                               replace=False))
    y_tr, y_va = y.values[tr_idx].astype(np.float32), \
        y.values[va_idx].astype(np.float32)
    nr_all = next_ret.to_numpy(dtype=np.float32)
    y_all = y.to_numpy(dtype=np.float32)
    age = (pd.Timestamp(TRAIN_END) - pd.to_datetime(dates[tr_idx])).days.values
    w_tr = (0.5 ** (age / 756.0)).astype(np.float32)
    w_all = np.ones(len(dates), dtype=np.float32)
    w_all[tr_idx] = w_tr

    tech_cols = [c for c in X.columns if c not in MKT_COLS]
    tech = to_matrix(X, tech_cols)
    del X
    pat = build_pattern_features(df)
    pattern_cols = list(pat.columns)
    patm = to_matrix(pat, pattern_cols)
    del pat, df
    full = np.hstack([tech, patm])
    all_cols = tech_cols + pattern_cols
    col_of = {c: i for i, c in enumerate(all_cols)}
    del tech, patm

    start = stock_start_idx(codes)
    log(f"样本: 训练 {len(tr_idx):,} / 验证 {len(va_idx):,}, "
        f"特征矩阵 {full.shape}")

    params = {
        "objective": "binary", "metric": ["auc", "binary_logloss"],
        "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 1000,
        "feature_fraction": 0.6, "bagging_fraction": 0.7, "bagging_freq": 1,
        "lambda_l2": 20.0, "max_bin": 127, "num_threads": args.threads,
        "verbosity": -1, "seed": args.seed,
    }
    thermal_cb = ThermalCallback(limit=args.temp_limit,
                                 resume=args.temp_resume,
                                 throttle=0.0, every=5)

    rows, sums = [], []
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    for cfg in configs:
        groups = cfg.split("+")
        for g in groups:
            if g not in GROUPS:
                raise SystemExit(f"未知组 {g}, 可选 tech/pattern")
        cols = [c for g in groups for c in
                (tech_cols if g == "tech" else pattern_cols)]
        idx = [col_of[c] for c in cols]
        vals = np.ascontiguousarray(full[:, idx])
        mean, std = standardize(vals, tr_idx)
        log(f"[{cfg}] 特征 {len(cols)}, TCN(GPU) 与 LGBM(CPU) "
            f"{'并行' if not args.no_parallel else '串行'}训练 ...")

        def run_lgbm():
            dtr = lgb.Dataset(vals[tr_idx], label=y_tr, weight=w_tr,
                              feature_name=cols)
            dva = lgb.Dataset(vals[va_idx], label=y_va, reference=dtr)
            lgbm = lgb.train(params, dtr, num_boost_round=args.rounds,
                             valid_sets=[dva],
                             callbacks=[lgb.early_stopping(
                                 150, first_metric_only=True, verbose=False),
                                 thermal_cb])
            lgbm_box["pl"] = lgbm.predict(
                vals[va_idx], num_iteration=lgbm.best_iteration)
            lgbm_box["iter"] = lgbm.best_iteration

        lgbm_box = {}
        th = None
        if not args.skip_lgbm and not args.no_parallel:
            th = threading.Thread(target=run_lgbm, daemon=True)
            th.start()
        model, best_auc, make_batch = train_tcn(
            args, vals, start, tr_idx, va_idx, y_all, w_all, len(cols), device)
        p_va = evaluate(model, make_batch, va_idx, 8192, device)
        row = {"model": "TCN", "config": cfg, "n_feat": len(cols),
               "valid_auc": roc_auc_score(y_va, p_va),
               "valid_logloss": log_loss(y_va, p_va),
               "valid_acc": accuracy_score(y_va, p_va > 0.5)}
        log(f"[{cfg}] TCN valid_auc {row['valid_auc']:.4f}")
        rows.append(row)
        pairs = [("TCN", p_va)]
        if not args.skip_lgbm:
            if th is not None:
                th.join()
            else:
                run_lgbm()
            pl = lgbm_box["pl"]
            rows.append({
                "model": "LGBM", "config": cfg, "n_feat": len(cols),
                "best_iter": lgbm_box.get("iter"),
                "valid_auc": roc_auc_score(y_va, pl),
                "valid_logloss": log_loss(y_va, pl),
                "valid_acc": accuracy_score(y_va, pl > 0.5)})
            log(f"[{cfg}] LGBM valid_auc {rows[-1]['valid_auc']:.4f}")
            pairs.append(("LGBM", pl))

        for name, p in pairs:
            bt = pd.DataFrame({"date": dates[va_idx], "code": codes[va_idx],
                               "prob": p, "next_ret": nr_all[va_idx],
                               "y": y_va})
            s, _, _, _, ic, icir = run_period(bt, f"{cfg}-{name}")
            r = next(x for x in rows if x["model"] == name
                     and x["config"] == cfg)
            r["ic"] = ic
            r["ic_ir"] = icir
            for strat in ("Top20", "Top50", "P>=0.55"):
                m = s[s["strategy"] == strat]
                if not m.empty:
                    key = strat.replace(">=", "").replace(".", "")
                    r[f"{key}_ann_net"] = float(m.iloc[0]["ann_net"])
                    r[f"{key}_sharpe"] = float(m.iloc[0]["sharpe"])
            r["mkt_ann_net"] = float(
                s[s["strategy"] == "市场等权"].iloc[0]["ann_net"])
            sums.append((cfg, name, s))
            del bt
        del model, vals
        torch.cuda.empty_cache()

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT_DIR, "tcn_results.csv"), index=False)
    write_report(res, sums)
    print("\n============ TCN(序列) vs LGBM(标量) ============")
    print(res.reindex(columns=SHOW_COLS).to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n输出目录: {OUT_DIR}")
    log(f"全部完成, 总用时 {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
