#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LambdaRankIC 风格的自定义目标函数

参考: LambdaRankIC: Directly Optimizing Rank IC for Financial Prediction
      arXiv:2605.00501

由于 Rank IC 中的排序算子不可微, 这里采用 pairwise logistic 近似:
  - 每天内对样本对 (i, j), 按真实收益差加权
  - 损失 L = sum_{i,j} w_ij * log(1 + exp(-s_ij * (f_i - f_j)))
    s_ij = sign(ret_i - ret_j), w_ij = |ret_i - ret_j| 的秩次差近似
  - 该损失直接惩罚"收益高的股票预测分反而低"的成对排序错误

使用方式:
  import lightgbm as lgb
  from rank_ic_objective import make_lambdarank_ic_objective, make_rank_ic_eval
  fobj = make_lambdarank_ic_objective(Rtr, dates_tr)
  feval = make_rank_ic_eval(Rva, dates_va)
  model = lgb.train(
      {'objective': None, 'metric': 'None'},
      dtr, fobj=fobj, feval=feval, ...)
"""
import warnings

import numpy as np


def _group_indices(dates):
    """返回按日期分组的 (start, end) 列表, 假设 dates 已排序"""
    dates = np.asarray(dates)
    if len(dates) == 0:
        return []
    uniq, counts = np.unique(dates, return_counts=True)
    ends = np.cumsum(counts)
    starts = np.concatenate([[0], ends[:-1]])
    return list(zip(starts, ends))


def _lambdarank_ic_grad(preds, ret, dates, margin=1.0, eps=1e-12):
    """核心梯度计算, 不依赖 lightgbm.Dataset"""
    ret = np.asarray(ret, dtype=np.float64)
    preds = np.asarray(preds, dtype=np.float64)
    grad = np.zeros_like(preds, dtype=np.float64)
    hess = np.zeros_like(preds, dtype=np.float64)

    order = np.argsort(dates, kind='stable')
    inv_order = np.empty(len(order), dtype=np.int64)
    inv_order[order] = np.arange(len(order))
    ret_s = ret[order]
    pred_s = preds[order]
    groups = _group_indices(np.asarray(dates)[order])

    for s, e in groups:
        n = e - s
        if n < 2:
            continue
        r = ret_s[s:e]
        p = pred_s[s:e]
        dr = r[:, None] - r[None, :]
        dp = p[:, None] - p[None, :]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            w = np.abs(dr) / margin
            sgn = np.sign(dr)
            np.fill_diagonal(w, 0.0)
            np.fill_diagonal(sgn, 0.0)
            z = sgn * dp
            sig = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
            g = (w * sgn * (sig - 1.0)).sum(axis=1)
            h = (w * sig * (1.0 - sig)).sum(axis=1)
        grad_s = np.clip(g, -50, 50)
        hess_s = np.clip(h, eps, 50)
        grad[order[s:e]] = grad_s
        hess[order[s:e]] = hess_s

    return grad, hess


def make_lambdarank_ic_objective(ret, dates, margin=1.0):
    """返回一个 LightGBM 自定义目标函数

    参数
    ----
    ret : np.ndarray
        真实收益 (与训练集样本一一对应)
    dates : np.ndarray
        日期标识 (与训练集样本一一对应)
    margin : float
        收益差缩放, 默认 1.0; 收益波动大或标签为长周期收益时可调小
    """
    ret = np.asarray(ret, dtype=np.float64)
    dates = np.asarray(dates)

    def objective(preds, train_data):
        return _lambdarank_ic_grad(preds, ret, dates, margin=margin)

    return objective


def _rank_ic(preds, ret, dates):
    ret = np.asarray(ret, dtype=np.float64)
    preds = np.asarray(preds, dtype=np.float64)
    dates = np.asarray(dates)
    ic_vals = []
    for s, e in _group_indices(dates):
        if e - s < 50:
            continue
        r = ret[s:e]
        p = preds[s:e]
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
    ret = np.asarray(ret, dtype=np.float64)
    dates = np.asarray(dates)

    def eval_fn(preds, train_data):
        ic = _rank_ic(preds, ret, dates)
        return 'ic', ic, True

    return eval_fn


def make_rank_auc_eval(y):
    """返回 LightGBM feval: 二分类 AUC(用 y 标签)"""
    y = np.asarray(y)

    def eval_fn(preds, train_data):
        if len(np.unique(y)) < 2:
            return 'auc', 0.0, True
        from sklearn.metrics import roc_auc_score
        return 'auc', roc_auc_score(y, preds), True

    return eval_fn
