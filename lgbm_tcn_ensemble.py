#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LGBM + TCN 集成模型 + 训练/验证分段回测

在 lgbm_train_backtest.py 的基础上增加 TCN (Temporal Convolutional Network):
- 数据/特征/标签/切分/回测逻辑直接复用 lgbm_train_backtest.py
- LGBM: 默认复用 lgbm_output/lgbm_model.txt, 也可 --retrain-lgbm 重新训练
- TCN: 对每只股票最近 L 个交易日的特征序列建模 (膨胀因果卷积 + 残差块)
- 融合: p = (1-w)*p_lgbm + w*p_tcn, 另附横截面排名融合, 与单模型分别回测对比

用法:
    python3 lgbm_tcn_ensemble.py --limit 300 --epochs 3 --tcn-weight 0.5   # 冒烟
    python3 lgbm_tcn_ensemble.py --device cuda                             # 全量
    python3 lgbm_tcn_ensemble.py --device cuda --resume                    # 崩溃后从检查点续训
"""
import argparse
import copy
import gc
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from lgbm_train_backtest import (COST, DATA_START, FEATURES, MIN_HOLD, SEED,
                                 TRAIN_END, build_features, load_bars, log,
                                 run_period)

try:
    import torch
    import torch.nn as nn
except ImportError as e:
    raise SystemExit(f"需要 PyTorch: pip install torch ({e})")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "ensemble_output")
LGBM_DEFAULT = os.path.join(HERE, "lgbm_output", "lgbm_model.txt")


# ---------------------------------------------------------------- TCN 模型
class Chomp1d(nn.Module):
    """裁掉卷积右侧多出的未来时间步, 保证因果性"""

    def __init__(self, chomp: int):
        super().__init__()
        self.chomp = chomp

    def forward(self, x):
        return x[:, :, :-self.chomp].contiguous() if self.chomp > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel, dilation, dropout):
        super().__init__()
        try:
            from torch.nn.utils.parametrizations import weight_norm
        except ImportError:
            from torch.nn.utils import weight_norm
        pad = (kernel - 1) * dilation
        self.conv1 = weight_norm(nn.Conv1d(n_in, n_out, kernel,
                                           dilation=dilation, padding=pad))
        self.conv2 = weight_norm(nn.Conv1d(n_out, n_out, kernel,
                                           dilation=dilation, padding=pad))
        self.net = nn.Sequential(
            self.conv1, Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
            self.conv2, Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout))
        self.downsample = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
        self.relu = nn.ReLU()
        for conv in (self.conv1, self.conv2):
            for p in conv.parameters():
                if p.dim() > 1:
                    nn.init.normal_(p, 0.0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(self, n_feat, channels, kernel, dilations, dropout):
        super().__init__()
        blocks = []
        n_in = n_feat
        for n_out, d in zip(channels, dilations):
            blocks.append(TemporalBlock(n_in, n_out, kernel, d, dropout))
            n_in = n_out
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_in, 1))

    def forward(self, x):
        # x: (B, T, C) -> (B, C, T); 取最后一个时间步 (t 日) 输出
        y = self.tcn(x.transpose(1, 2))
        return self.head(y[:, :, -1]).squeeze(-1)


# ---------------------------------------------------------------- 数据处理
def to_matrix(X: pd.DataFrame) -> np.ndarray:
    """DataFrame -> 单块 float32 矩阵, 逐列赋值避免整表复制"""
    vals = np.empty((len(X), X.shape[1]), dtype=np.float32)
    for j, c in enumerate(X.columns):
        vals[:, j] = X[c].to_numpy(dtype=np.float32, copy=False)
    gc.collect()
    return vals


def stock_start_idx(codes: np.ndarray) -> np.ndarray:
    """每行所属股票的起始行号 (输入按 code, date 排序)"""
    n = len(codes)
    change = np.empty(n, dtype=bool)
    change[0] = True
    change[1:] = codes[1:] != codes[:-1]
    return np.maximum.accumulate(np.where(change, np.arange(n), 0)).astype(np.int32)


def standardize(vals: np.ndarray, idx: np.ndarray, chunk: int = 2_000_000):
    """用训练集统计量做列标准化, 缺失填 0 (就地修改)"""
    c = vals.shape[1]
    n = np.zeros(c, dtype=np.float64)
    s = np.zeros(c, dtype=np.float64)
    ss = np.zeros(c, dtype=np.float64)
    for i in range(0, len(idx), chunk):
        b = vals[idx[i:i + chunk]]
        m = np.isfinite(b)
        b = np.where(m, b, 0.0)
        n += m.sum(axis=0)
        s += b.sum(axis=0, dtype=np.float64)
        ss += (b * b).sum(axis=0, dtype=np.float64)
    n = np.maximum(n, 1.0)
    mean = (s / n).astype(np.float32)
    var = np.maximum(ss / n - (s / n) ** 2, 1e-12)
    std = np.sqrt(var).astype(np.float32)
    vals -= mean
    vals /= std
    np.nan_to_num(vals, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return mean, std


def parse_ints(s: str):
    return [int(v) for v in s.split(",") if v.strip()]


def pair_lists(channels, dilations):
    """channels 与 dilations 按块一一对应; 单元素时自动扩展到对方长度"""
    if len(channels) == 1:
        channels = channels * len(dilations)
    if len(dilations) == 1:
        dilations = dilations * len(channels)
    if len(channels) != len(dilations):
        raise ValueError(f"channels({len(channels)}) 与 dilations({len(dilations)}) 长度需一致")
    return channels, dilations


# ---------------------------------------------------------------- 训练/预测
def make_batch_fn(vals, start, y_all, w_all, lookback):
    offs = np.arange(lookback)

    def make_batch(rows: np.ndarray):
        r = rows.astype(np.int64)
        w_idx = np.maximum(r[:, None] - lookback + 1 + offs[None, :],
                           start[r][:, None])
        x = vals[w_idx]
        return (torch.from_numpy(x), torch.from_numpy(y_all[r]),
                torch.from_numpy(w_all[r]))

    return make_batch


def train_tcn(args, vals, start, tr_idx, va_idx, y_all, w_all, n_feat, device,
              ckpt_path=None):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    channels, dilations = pair_lists(parse_ints(args.channels),
                                     parse_ints(args.dilations))
    model = TCN(n_feat, channels, args.kernel, dilations, args.dropout).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    log(f"TCN 参数量 {n_par:,}, 通道 {channels}, 膨胀 {dilations}, lookback {args.lookback}")

    use = tr_idx[::args.train_stride] if args.train_stride > 1 else tr_idx
    if args.max_train and len(use) > args.max_train:
        use = use[:args.max_train]
    make_batch = make_batch_fn(vals, start, y_all, w_all, args.lookback)
    opt = torch.optim.AdamW(model.parameters(), lr=args.tcn_lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=1)
    lossf = nn.BCEWithLogitsLoss(reduction="none")

    @torch.inference_mode()
    def predict(idx, batch=8192):
        model.eval()
        out = np.empty(len(idx), dtype=np.float32)
        for i in range(0, len(idx), batch):
            xb, _, _ = make_batch(idx[i:i + batch])
            out[i:i + batch] = torch.sigmoid(
                model(xb.to(device))).float().cpu().numpy()
            if args.throttle > 0:
                time.sleep(args.throttle)
        return out

    best_auc, best_state, bad, hist = -1.0, None, 0, []
    start_ep, start_step = 0, 0
    if args.resume and ckpt_path and os.path.exists(ckpt_path):
        try:
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        except Exception as e:
            ck = None
            log(f"检查点损坏, 忽略: {e}")
        if ck:
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            best_auc, best_state = ck["best_auc"], ck["best_state"]
            bad, hist = ck["bad"], ck["hist"]
            start_ep, start_step = ck["epoch"], ck["step"]
            log(f"从检查点恢复: 已完成 epoch {start_ep}, step {start_step}, "
                f"best_auc {best_auc:.4f}")

    def save_ckpt(epoch, step):
        if not ckpt_path:
            return
        tmp = ckpt_path + ".tmp"
        torch.save({"epoch": epoch, "step": step, "model": model.state_dict(),
                    "opt": opt.state_dict(), "best_auc": best_auc,
                    "best_state": best_state, "bad": bad, "hist": hist}, tmp)
        os.replace(tmp, ckpt_path)

    t0 = time.time()
    first_ep = start_ep if (start_ep > 0 and start_step > 0) else start_ep + 1
    for ep in range(first_ep, args.epochs + 1):
        model.train()
        perm = np.random.RandomState(args.seed * 1000 + ep).permutation(len(use))
        n_steps = (len(perm) + args.batch_size - 1) // args.batch_size
        first_step = start_step if ep == start_ep else 0
        run_loss, seen = 0.0, 0
        for step in range(first_step, n_steps):
            rows = use[perm[step * args.batch_size:(step + 1) * args.batch_size]]
            xb, yb, wb = make_batch(rows)
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = (lossf(model(xb), yb) * wb).sum() / wb.sum().clamp_min(1e-9)
            loss.backward()
            if args.clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            if args.throttle > 0:
                time.sleep(args.throttle)
            run_loss += loss.item() * len(rows)
            seen += len(rows)
            if step % 500 == 0:
                log(f"  epoch {ep} step {step}/{n_steps} "
                    f"{seen:,}/{len(perm):,} loss {run_loss / max(seen, 1):.5f}")
            if ckpt_path and (step + 1) % 200 == 0:
                save_ckpt(ep, step + 1)
        pva = predict(va_idx)
        auc = roc_auc_score(y_all[va_idx], pva)
        ll = log_loss(y_all[va_idx], pva)
        sched.step(auc)
        hist.append({"epoch": ep, "train_loss": run_loss / max(seen, 1),
                     "val_auc": auc, "val_logloss": ll,
                     "lr": opt.param_groups[0]["lr"]})
        log(f"epoch {ep}: train_loss {run_loss / max(seen, 1):.5f}, "
            f"val_auc {auc:.4f}, val_logloss {ll:.4f}, "
            f"lr {opt.param_groups[0]['lr']:.2e}, {(time.time() - t0) / 60:.1f} min")
        if auc > best_auc + 1e-5:
            best_auc, best_state, bad = auc, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        save_ckpt(ep, 0)
        if bad >= args.patience:
            log(f"验证集 {args.patience} 轮无提升, 提前停止")
            break
    model.load_state_dict(best_state)
    log(f"TCN 训练完成, best val_auc {best_auc:.4f}, 用时 {(time.time() - t0) / 60:.1f} min")
    return model, pd.DataFrame(hist), predict


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="随机抽取N只股票(冒烟测试)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--retrain-lgbm", action="store_true",
                    help="忽略已有模型文件, 重新训练 LGBM")
    ap.add_argument("--lgbm-model", default=LGBM_DEFAULT)
    ap.add_argument("--lgbm-rounds", type=int, default=3000)
    ap.add_argument("--leaves", type=int, default=31)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--lookback", type=int, default=20)
    ap.add_argument("--channels", default="64,64,64,64")
    ap.add_argument("--dilations", default="1,2,4,8")
    ap.add_argument("--kernel", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--tcn-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--train-stride", type=int, default=1,
                    help="训练样本抽稀步长, >1 加速训练")
    ap.add_argument("--max-train", type=int, default=0, help="训练样本上限")
    ap.add_argument("--throttle", type=float, default=0.0,
                    help="每步训练后休眠秒数, 降低GPU占空比/发热, 如 0.05")
    ap.add_argument("--tcn-weight", type=float, default=0.5)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--resume", action="store_true",
                    help="从 ensemble_output/tcn_ckpt.pt 断点续训")
    args = ap.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            log("CUDA 不可用, 回退 CPU")
            device = "cpu"
    if device == "cuda":
        cap = torch.cuda.get_device_capability()
        log(f"设备 cuda ({torch.cuda.get_device_name(0)}, sm_{cap[0]}{cap[1]},"
            f" cuda {torch.version.cuda})")
    else:
        torch.set_num_threads(os.cpu_count() or 4)
        log("设备 cpu")

    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    df = load_bars(args.limit or None, args.seed)
    X, next_ret, y, tradable = build_features(df)
    feat_order = list(X.columns)
    if set(feat_order) != set(FEATURES):
        log(f"警告: 特征列与 FEATURES 不一致 {feat_order}")

    year = df["date"].str.slice(0, 4).astype(int)
    base = ((year >= int(DATA_START[:4])) & next_ret.notna() & tradable
            & X["ret60"].notna())
    m_tr = (base & (year <= int(TRAIN_END[:4]))).values
    m_va = (base & ~(year <= int(TRAIN_END[:4]))).values
    tr_idx = np.flatnonzero(m_tr)
    va_idx = np.flatnonzero(m_va)
    log(f"样本: 训练 {len(tr_idx):,} / 验证 {len(va_idx):,}")

    y_all = y.to_numpy(dtype=np.float32)
    nr_all = next_ret.to_numpy(dtype=np.float32)
    code_cat = df["code"].cat.categories.to_numpy()
    codes = code_cat[df["code"].cat.codes.to_numpy()]
    dates = df["date"].values
    n_feat = X.shape[1]

    # ---- LGBM 预测 (复用或重训) ----
    reused_lgbm = False
    if not args.retrain_lgbm and os.path.exists(args.lgbm_model):
        model_path = args.lgbm_model
        log(f"加载已有 LGBM 模型: {model_path}")
        booster = lgb.Booster(model_file=model_path)
        if list(booster.feature_name()) != feat_order:
            log("警告: 模型特征与当前特征不一致, 改为重训")
            booster = None
        else:
            reused_lgbm = True

            def lgbm_pred(idx, chunk=2_000_000):
                out = np.empty(len(idx), dtype=np.float32)
                for i in range(0, len(idx), chunk):
                    out[i:i + chunk] = booster.predict(X.values[idx[i:i + chunk]])
                return out
    else:
        booster = None

    if booster is None:
        params = {
            "objective": "binary", "metric": ["auc", "binary_logloss"],
            "learning_rate": args.lr, "num_leaves": args.leaves,
            "min_data_in_leaf": 1000, "feature_fraction": 0.6,
            "bagging_fraction": 0.7, "bagging_freq": 1, "lambda_l2": 20.0,
            "max_bin": 127, "num_threads": os.cpu_count(), "verbosity": -1,
            "seed": args.seed, "feature_fraction_seed": args.seed,
            "bagging_seed": args.seed,
        }
        Xtr = X.iloc[tr_idx]
        Xva = X.iloc[va_idx]
        ytr = y_all[tr_idx]
        age = (pd.Timestamp(TRAIN_END) - pd.to_datetime(dates[tr_idx])).days.values
        wtr = (0.5 ** (age / 756.0)).astype(np.float32)
        log("训练 LightGBM ...")
        dtr = lgb.Dataset(Xtr, label=ytr, weight=wtr,
                          feature_name=list(Xtr.columns))
        dva = lgb.Dataset(Xva, label=y_all[va_idx], reference=dtr)
        model_lgb = lgb.train(
            params, dtr, num_boost_round=args.lgbm_rounds, valid_sets=[dva],
            callbacks=[lgb.early_stopping(150, first_metric_only=True,
                                          verbose=False),
                       lgb.log_evaluation(100)])
        model_lgb.save_model(os.path.join(OUT_DIR, "lgbm_model.txt"))
        pd.DataFrame({
            "feature": model_lgb.feature_name(),
            "gain": model_lgb.feature_importance("gain"),
            "split": model_lgb.feature_importance("split"),
        }).sort_values("gain", ascending=False).to_csv(
            os.path.join(OUT_DIR, "feature_importance.csv"), index=False)
        booster = model_lgb
        del Xtr, Xva, dtr, dva, model_lgb
        gc.collect()
        log("LGBM 训练完成")

        def lgbm_pred(idx, chunk=2_000_000):
            out = np.empty(len(idx), dtype=np.float32)
            for i in range(0, len(idx), chunk):
                out[i:i + chunk] = booster.predict(X.values[idx[i:i + chunk]])
            return out

    p_lgbm_tr = lgbm_pred(tr_idx)
    p_lgbm_va = lgbm_pred(va_idx)
    del lgbm_pred, booster
    gc.collect()

    # ---- 构建 float32 矩阵并释放 DataFrame ----
    del df
    gc.collect()
    vals = to_matrix(X)
    del X
    gc.collect()
    start = stock_start_idx(codes)
    log(f"特征矩阵: {vals.shape}, {vals.nbytes / 2**30:.2f} GiB")

    # ---- 训练集权重 (时间衰减, 半衰期3年) ----
    age = (pd.Timestamp(TRAIN_END) - pd.to_datetime(dates[tr_idx])).days.values
    w_tr = (0.5 ** (age / 756.0)).astype(np.float32)
    w_tr /= w_tr.mean()
    w_all = np.ones(len(vals), dtype=np.float32)
    w_all[tr_idx] = w_tr

    # ---- 标准化并训练 TCN ----
    mean, std = standardize(vals, tr_idx)
    pd.DataFrame({"feature": feat_order, "mean": mean, "std": std}).to_csv(
        os.path.join(OUT_DIR, "feature_scaler.csv"), index=False)
    log("开始训练 TCN ...")
    ckpt_path = os.path.join(OUT_DIR, "tcn_ckpt.pt")
    tcn, hist, predict = train_tcn(args, vals, start, tr_idx, va_idx,
                                   y_all, w_all, n_feat, device,
                                   ckpt_path=ckpt_path)
    ch_used, dl_used = pair_lists(parse_ints(args.channels),
                                  parse_ints(args.dilations))
    torch.save({"state_dict": tcn.state_dict(), "n_feat": n_feat,
                "channels": ch_used, "dilations": dl_used,
                "kernel": args.kernel, "dropout": args.dropout,
                "lookback": args.lookback, "features": feat_order},
               os.path.join(OUT_DIR, "tcn_model.pt"))
    hist.to_csv(os.path.join(OUT_DIR, "tcn_train_log.csv"), index=False)

    p_tcn_tr = predict(tr_idx)
    p_tcn_va = predict(va_idx)

    # ---- 融合与评估 ----
    w = args.tcn_weight
    p_fuse_tr = (1.0 - w) * p_lgbm_tr + w * p_tcn_tr
    p_fuse_va = (1.0 - w) * p_lgbm_va + w * p_tcn_va

    evals = []
    for name, ptr, pva in [("LGBM", p_lgbm_tr, p_lgbm_va),
                           ("TCN", p_tcn_tr, p_tcn_va),
                           ("融合", p_fuse_tr, p_fuse_va)]:
        for period, yy, pp in [("训练集", y_all[tr_idx], ptr),
                               ("验证集", y_all[va_idx], pva)]:
            evals.append({"model": name, "period": period,
                          "n": len(yy), "auc": roc_auc_score(yy, pp),
                          "logloss": log_loss(yy, pp),
                          "acc": accuracy_score(yy, pp > 0.5),
                          "base_rate": float(yy.mean())})
    ev = pd.DataFrame(evals)
    ev.to_csv(os.path.join(OUT_DIR, "model_eval.csv"), index=False)
    print("\n================ 模型评估 ================")
    print(ev.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n----------- 融合权重扫描 (验证集 AUC) -----------")
    for wi in np.arange(0, 1.0001, 0.1):
        auc = roc_auc_score(y_all[va_idx],
                            (1.0 - wi) * p_lgbm_va + wi * p_tcn_va)
        print(f"  w_tcn={wi:.1f}  val_auc={auc:.4f}")

    # ---- 回测: 单模型 + 概率融合 + 排名融合 ----
    def make_bt(idx, pl, pt):
        return pd.DataFrame({
            "date": dates[idx], "code": codes[idx],
            "prob_lgbm": pl, "prob_tcn": pt,
            "next_ret": nr_all[idx], "y": y_all[idx]})

    def rank_fuse(bt):
        r1 = bt.groupby("date")["prob_lgbm"].rank(pct=True)
        r2 = bt.groupby("date")["prob_tcn"].rank(pct=True)
        return (1.0 - w) * r1 + w * r2

    bt_tr = make_bt(tr_idx, p_lgbm_tr, p_tcn_tr)
    bt_va = make_bt(va_idx, p_lgbm_va, p_tcn_va)
    frame_pairs = []
    for period, bt, pl, pt in [("训练集", bt_tr, p_lgbm_tr, p_tcn_tr),
                               ("验证集", bt_va, p_lgbm_va, p_tcn_va)]:
        f = bt[["date", "code", "next_ret", "y"]].copy()
        f["prob"] = pl
        frame_pairs.append((f"{period} LGBM", f))
        f = f.copy()
        f["prob"] = pt
        frame_pairs.append((f"{period} TCN", f))
        f = f.copy()
        f["prob"] = (1.0 - w) * pl + w * pt
        frame_pairs.append((f"{period} 融合", f))
        f = f.copy()
        f["prob"] = rank_fuse(bt)
        frame_pairs.append((f"{period} 融合R", f))

    all_sum, all_daily, all_dec, all_auc, ic_rows = [], [], [], [], []
    for label, f in frame_pairs:
        log(f"回测 {label} ...")
        s, c, d, a, ic, icir = run_period(f, label)
        all_sum.append(s)
        all_daily.append(c)
        all_dec.append(d)
        all_auc.append(a)
        ic_rows.append({"period": label, "ic_mean": ic, "ic_ir": icir,
                        "t_stat": icir * np.sqrt(s.iloc[0]["days"])})

    summary = pd.concat(all_sum, ignore_index=True)
    print("\n========= 回测汇总 (t+1开盘买/收盘卖, 等权, 成本万15, 剔除涨停开盘) =========")
    cols = ["strategy", "days", "avg_hold", "ann_gross", "ann_net",
            "excess_ann", "sharpe", "mdd", "win_daily", "avg_turnover"]
    for p in summary["period"].unique():
        print(f"\n--- {p} ---")
        print(summary[summary["period"] == p][cols].to_string(
            index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n---------------- IC (RankIC) ----------------")
    print(pd.DataFrame(ic_rows).to_string(index=False,
          float_format=lambda v: f"{v:.4f}"))

    print("\n---------------- 分年度 AUC ----------------")
    auc_df = pd.concat(all_auc, ignore_index=True)
    print(auc_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---- 输出 ----
    summary.to_csv(os.path.join(OUT_DIR, "backtest_summary.csv"), index=False)
    pd.concat(all_daily, ignore_index=True).to_csv(
        os.path.join(OUT_DIR, "daily_equity.csv"), index=False)
    pd.concat(all_dec, ignore_index=True).to_csv(
        os.path.join(OUT_DIR, "prob_decile.csv"), index=False)
    auc_df.to_csv(os.path.join(OUT_DIR, "yearly_auc.csv"), index=False)

    pred_va = bt_va.copy()
    pred_va["prob_fuse"] = p_fuse_va
    pred_va["prob_fuse_rank"] = rank_fuse(bt_va).values
    pred_va.to_csv(os.path.join(OUT_DIR, "valid_predictions.csv.gz"),
                   index=False, compression="gzip")

    cfg = {"lgbm_model": args.lgbm_model if reused_lgbm else None,
           "tcn": {k: getattr(args, k) for k in
                   ["lookback", "channels", "dilations", "kernel", "dropout",
                    "tcn_lr", "weight_decay", "epochs", "batch_size",
                    "patience", "clip", "train_stride", "max_train"]},
           "tcn_weight": w, "device": device,
           "features": feat_order, "cost": COST, "min_hold": MIN_HOLD,
           "label": "close[t+1] > open[t+1] (t+1开盘买, t+1收盘卖)",
           "train": f"{DATA_START}~{TRAIN_END}", "valid": "2024-01-01~"}
    with open(os.path.join(OUT_DIR, "run_config.json"), "w",
              encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    try:
        os.remove(ckpt_path)
    except OSError:
        pass
    print(f"\n输出目录: {OUT_DIR}")
    log(f"全部完成, 总用时 {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
