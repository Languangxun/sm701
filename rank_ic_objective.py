#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LambdaRankIC 风格的自定义目标函数

参考: LambdaRankIC: Directly Optimizing Rank IC for Financial Prediction
      arXiv:2605.00501

修正点:
  1. 数据按 (code, date) 排列, 目标函数内部按日期重新排序/分组
  2. 收益先转截面 rank percentile, 再用 rank 差加权, 优化的是 Rank IC
  3. 每天只随机采样 max_pairs 对, 避免 O(n^2) 全矩阵
  4. 支持样本权重(如时间衰减权重)

使用方式:
  import lightgbm as lgb
  from rank_ic_objective import make_lambdarank_ic_objective, make_rank_ic_eval
  fobj = make_lambdarank_ic_objective(Rtr, dates_tr, weights=w_tr)
  feval = make_rank_ic_eval(Rva, dates_va)
  params = {"objective": fobj, "metric": "None"}
  model = lgb.train(params, dtr, num_boost_round=..., valid_sets=[dva],
                    feval=feval, ...)
"""
import warnings

import numpy as np


def _prepare(ret, dates, weights=None):
    """按日期排序, 返回排序后的 ret/dates/weights 及分组边界"""
    ret = np.asarray(ret, dtype=np.float64)
    dates = np.asarray(dates)
    order = np.argsort(dates, kind="stable")
    inv_order = np.empty(len(order), dtype=np.int64)
    inv_order[order] = np.arange(len(order))
    ret_s = ret[order]
    dates_s = dates[order]
    weights_s = (np.ones_like(ret_s) if weights is None
                 else np.asarray(weights, dtype=np.float64)[order])
    uniq, counts = np.unique(dates_s, return_counts=True)
    ends = np.cumsum(counts)
    starts = np.concatenate([[0], ends[:-1]])
    groups = list(zip(starts, ends))
    return ret_s, dates_s, weights_s, groups, inv_order


def _day_grad(p, r, w, max_pairs=64, seed=0):
    """单日的 pairwise 梯度 (按 rank 差加权)

    p: 预测分数, r: 真实收益, w: 样本权重
    返回对应当日样本的梯度/hess (按 p/r/w 的顺序)
    """
    n = len(p)
    if n < 2:
        return np.zeros(n), np.zeros(n)
    # 截面 rank percentile (1~n -> 0~1)
    rank = (np.argsort(np.argsort(r)) + 1.0) / n
    # 预测分也做 rank (用于计算梯度时更稳定)
    p_rank = (np.argsort(np.argsort(p)) + 1.0) / n

    rng = np.random.RandomState(seed)
    grad = np.zeros(n, dtype=np.float64)
    hess = np.zeros(n, dtype=np.float64)

    # 每天随机采样配对, 避免 O(n^2)
    n_pairs = min(max_pairs, n * (n - 1) // 2) if max_pairs > 0 else n * (n - 1) // 2
    if n_pairs <= 0:
        return grad, hess

    # 均匀随机采样 (i, j), i != j
    for _ in range(n_pairs):
        i, j = rng.choice(n, size=2, replace=False)
        dr = rank[i] - rank[j]          # rank 差, 范围 [-1,1]
        dp = p[i] - p[j]
        sgn = np.sign(dr)
        if sgn == 0:
            continue
        # 权重: |rank 差| * 两个样本权重的几何平均
        pair_w = abs(dr) * np.sqrt(w[i] * w[j])
        z = sgn * dp
        sig = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
        # dL/dp_i = pair_w * sgn * (sig - 1)
        g_i = pair_w * sgn * (sig - 1.0)
        g_j = -g_i
        # hess 近似
        h = pair_w * sig * (1.0 - sig)
        grad[i] += g_i
        grad[j] += g_j
        hess[i] += h
        hess[j] += h
    return grad, hess


def _lambdarank_ic_grad(preds, ret, dates, weights=None,
                        max_pairs_per_day=64, eps=1e-12, seed=42):
    """核心梯度计算"""
    ret_s, dates_s, weights_s, groups, inv_order = _prepare(
        ret, dates, weights)
    preds_s = np.asarray(preds, dtype=np.float64)[inv_order]
    grad_s = np.zeros_like(preds_s, dtype=np.float64)
    hess_s = np.zeros_like(preds_s, dtype=np.float64)

    for day_idx, (s, e) in enumerate(groups):
        if e - s < 2:
            continue
        g, h = _day_grad(
            preds_s[s:e], ret_s[s:e], weights_s[s:e],
            max_pairs=max_pairs_per_day,
            seed=seed + day_idx)
        grad_s[s:e] += g
        hess_s[s:e] += h

    grad = np.empty_like(preds_s)
    hess = np.empty_like(preds_s)
    grad[inv_order] = grad_s
    hess[inv_order] = hess_s

    # 数值稳定
    grad = np.clip(grad, -50, 50)
    hess = np.clip(hess, eps, 50)
    return grad, hess


def make_lambdarank_ic_objective(ret, dates, weights=None,
                                 max_pairs_per_day=64, seed=42):
    """返回 LightGBM 自定义目标函数"""
    def objective(preds, train_data):
        return _lambdarank_ic_grad(
            preds, ret, dates, weights=weights,
            max_pairs_per_day=max_pairs_per_day, seed=seed)
    return objective


def _rank_ic(preds, ret, dates):
    """逐日 Rank IC (Spearman)"""
    ret_s, dates_s, _, groups, inv_order = _prepare(ret, dates)
    preds_s = np.asarray(preds, dtype=np.float64)[inv_order]
    ic_vals = []
    for s, e in groups:
        if e - s < 50:
            continue
        r = ret_s[s:e]
        p = preds_s[s:e]
        if np.std(p) < 1e-12 or np.std(r) < 1e-12:
            continue
        rp = np.argsort(np.argsort(p)) + 1
        rr = np.argsort(np.argsort(r)) + 1
        ic = np.corrcoef(rp, rr)[0, 1]
        if np.isfinite(ic):
            ic_vals.append(ic)
    return float(np.mean(ic_vals)) if ic_vals else 0.0


def make_rank_ic_eval(ret, dates):
    """返回 LightGBM feval: 逐日 Rank IC"""
    def eval_fn(preds, train_data):
        return 'ic', _rank_ic(preds, ret, dates), True
    return eval_fn


def make_auc_eval(y):
    """返回 LightGBM feval: 二分类 AUC"""
    y = np.asarray(y)

    def eval_fn(preds, train_data):
        if len(np.unique(y)) < 2:
            return 'auc', 0.0, True
        from sklearn.metrics import roc_auc_score
        return 'auc', roc_auc_score(y, preds), True
    return eval_fn


def make_mse_eval(y):
    """返回 LightGBM feval: MSE (用于 cross_rank 回归标签)"""
    y = np.asarray(y)

    def eval_fn(preds, train_data):
        m = np.isfinite(y) & np.isfinite(preds)
        if m.sum() == 0:
            return 'mse', 0.0, False
        return 'mse', float(np.mean((y[m] - preds[m]) ** 2)), False
    return eval_fn
