# 股市预测模型 · 技术文档

> A 股 T+1 日频上涨概率预测：LightGBM 横截面选股 + TCN 时序形态 + 多周期（T+1/T+5/T+10）+ 基本面/资金/事件因子。
> 数据全部 T+1 执行（t 收盘出信号、t+1 开盘成交），无日内换仓。

---

## 目录

1. [总体流程](#1-总体流程)
2. [数据层](#2-数据层)
3. [数据抓取脚本](#3-数据抓取脚本)
4. [数据清洗](#4-数据清洗)
5. [特征工程](#5-特征工程)
6. [标签与样本切分](#6-标签与样本切分)
7. [模型](#7-模型)
8. [训练脚本](#8-训练脚本)
9. [回测方法](#9-回测方法)
10. [评估指标](#10-评估指标)
11. [实验结论](#11-实验结论)
12. [概率校准](#12-概率校准)
13. [前视偏差审计](#13-前视偏差审计)
14. [运维工具](#14-运维工具)
15. [环境与运行命令](#15-环境与运行命令)
16. [输出文件说明](#16-输出文件说明)
17. [已知限制与下一步](#17-已知限制与下一步)

---

## 1. 总体流程

```
行情/基本面/资金数据抓取 (akshare -> stock_cache.db, SQLite)
        │
        ▼
数据清洗 clean_data.py（审计 + 保守修复 + 报告）
        │
        ▼
特征工程（9 组，全部 point-in-time 对齐，见 §5）
        │
        ▼
标签 R1/R5/R10 ＋ 训练集 2018–2023 / 验证集 2024+（按年硬切）
        │
        ├─► LightGBM 横截面二分类（特征组消融 / 多周期）
        ├─► TCN 时序形态（个股最近 L 天特征序列）
        └─► 概率融合 p = (1-w)·p_lgbm + w·p_tcn
        │
        ▼
评估（AUC/LogLoss/IC/Brier/ECE/PR）＋ 回测（t+1 开盘买/收盘卖或持有到期）
```

---

## 2. 数据层

数据库：`stock_cache.db`（SQLite，WAL 模式，约 7.7GB），全部表如下（2026-10-07 时点行数）：

### 2.1 行情与代码表

| 表 | 行数 | 内容 |
|---|---|---|
| `daily_bars` | 12,721,306 | 日K：code/date/open/high/low/close/vol，1993-04-30 ~ 2026-09-30，共 7629 个代码 |
| `stocks` | 7,594 | 代码/名称/行业/市值/PE/PB/换手快照 |
| `index_bars` | 44,956 | 10 个大盘指数日线（上证/深成/创业板/沪深300/中证500/1000/科创50/上证50/中证全指/北证50） |
| `industry_bars` | 143,561 | 申万 31 个一级行业指数日线（2000 年起，含 open/high/low/close/vol/amount） |
| `stock_industry` | 5,220 | 个股→申万一级行业映射（code/industry_code/industry_name/weight/start_date） |
| `adjust` / `adj_done` | 7,629 / 7,105 | 复权系数 K 与迁移记录（见 2.4） |
| `et_shares` | — | ETF 份额（预留） |
| `delisted` / `failed` | 321 / 94 | 退市标记 / 抓取失败记录 |
| `meta` | — | 运行期杂项 KV |

有效 A 股宇宙（训练用）：`sh60[0135]* / sh68[89]* / sz00[0123]* / sz30[01]* / bj*`，约 5,794 只。

### 2.2 基本面（历史全量，`fund_` 前缀）

| 表 | 行数 | 接口 | 历史起点 | 关键列 |
|---|---|---|---|---|
| `fund_valuation` | 9,534,123 | `stock_value_em` | 2018-01 | 每日 PE(TTM)/PE(静)/PB/PEG/PCF/PS/总市值/流通市值/股本 |
| `fund_valuation_baidu` | 23,214,613 | `stock_zh_valuation_baidu`（总市值/PE/PB/PCF×5 指标，长表） | 1999（周频-ish） | indicator/date/value |
| `fund_indicator_em` | 354,323 | `stock_financial_analysis_indicator_em`（141 列） | 上市起 | **NOTICE_DATE 公告日**＋ROE/毛利/负债率/同比增速等 |
| `fund_indicator` | 351,300 | `stock_financial_analysis_indicator`（新浪，86 列） | 上市起 | report_date（**训练未用**，无公告日） |
| `fund_abstract` | 351,914 | `stock_financial_abstract`（80 项关键指标） | 上市起 | report_date（**训练未用**，无公告日） |
| `fund_balance` | 317,215 | 资产负债表 by_report（221 列全字段） | 上市起 | NOTICE_DATE |
| `fund_profit` | 328,751 | 利润表 by_report（170 列） | 上市起 | NOTICE_DATE |
| `fund_cashflow` | 317,873 | 现金流量表 by_report（316 列） | 上市起 | NOTICE_DATE |
| `fund_fhps` | 56,888 | `stock_fhps_detail_em` | 上市起 | 分红送配（报告期） |
| `fund_gdhs` | 439,294 | `stock_zh_a_gdhs_detail_em` | 2013 起 | 股东户数/户均持股 |

> 训练实际只用 `fund_valuation`（每日估值）+ `fund_indicator_em`（按公告日对齐）。`fund_abstract` / `fund_indicator`（新浪）没有公告日字段，为避免前视偏差**不进训练**。

### 2.3 资金 / 事件 / 宏观

| 表 | 行数 | 接口 | 说明 |
|---|---|---|---|
| `margin_detail` | 6,681,783 | `stock_margin_detail_sse/szse`（逐交易日） | 2010-03 起；融资余额/买入/偿还、融券余量/卖出/偿还；ETF 代码按交易所修正前缀 |
| `lhb_detail` | 242,374 | `stock_lhb_detail_em`（逐月） | 2010 起；净买额/买入/卖出/换手率/流通市值/上榜原因/上榜后N日 |
| `yjyg` | 177,080 | `stock_yjyg_em`（逐报告期） | 业绩预告：预告类型/变动幅度/**公告日期** |
| `yjkb` | 34,908 | `stock_yjkb_em`（逐报告期） | 业绩快报：净利润同比/每股收益/**公告日期** |
| `stock_fund_flow` | 23,627 | `stock_individual_fund_flow`（逐股） | 主力/超大/大/中/小单净额+净占比；**仅最近约 120 个交易日**，覆盖 197 只（接口经代理不稳定，待补） |
| `qvix` | 5,646 | `index_option_50etf/300etf_qvix` | 期权波动率指数，2015 起 |
| `bond_yield` | 3,132 | `bond_zh_us_rate` | 中美 2/5/10/30 年国债收益率，2015 起（训练暂未用） |

### 2.4 复权口径

库内日K统一为**后复权 hfq（乘法口径）**：`daily_bars` 存 hfq 价，`adjust(code→K)` 存显示缩放（K=最新不复权价/后复权末价）。特征与标签只用**比率**（pct_change、open/close 比值），hfq 绝对值含未来分红因子但比率不受影响（已审计确认，见 §13）。

---

## 3. 数据抓取脚本

三个脚本相互独立，均支持断点续跑（进度表）+ 失败指数退避重试 + 多线程抓取/单线程入库 + 幂等覆盖写入：

| 脚本 | 数据集 | 进度表 | 任务量级 |
|---|---|---|---|
| `fetch_fundamentals.py` | 10 类基本面（valuation/baidu/abstract/indicator/indicator_em/balance/profit/cashflow/fhps/gdhs），默认 4 线程 | `fund_progress(code, dataset)` | ~57,940 任务 |
| `fetch_market_data.py` | 10 大盘指数（新浪 `stock_zh_index_daily`）+ 31 申万一级行业（`index_hist_sw` 行情 + `index_component_sw` 成分映射） | 无（幂等全量覆盖） | 一次性，约 2 分钟 |
| `fetch_extra_data.py` | qvix/bond/yjyg/yjkb（报告期）/lhb（逐月）/margin（逐交易日×两市）/flow（逐股），默认 3 线程 | `extra_progress(dataset, key)` | ~14,000 任务 |

```bash
python3 fetch_fundamentals.py --limit 3          # 冒烟
nohup python3 fetch_fundamentals.py --workers 4 > fundamental.log 2>&1 &
python3 fetch_market_data.py                     # 全量约2分钟
python3 fetch_extra_data.py --datasets qvix,bond # 指定数据集
```

注意：
- `stock_individual_fund_flow`（push2his）在代理下频繁 `ProxyError`，直连偶发 `RemoteDisconnected`；已用 `NO_PROXY=.eastmoney.com` 绕行，仍不稳定——flow 目前只覆盖 197 只。
- 深交所两融（szse.cn）曾因站点 302/SSL 失败 781 个任务，恢复后已补齐（margin 8,020 ok）。
- 两融标的含 ETF（51x/15x），入库时按交易所强制前缀（sse→sh、szse→sz），不用代码段推断。

---

## 4. 数据清洗

`clean_data.py`：离线审计 + 保守修复 + Markdown 报告（`reports/清洗报告_YYYYMMDD.md`）。

```bash
python3 clean_data.py                 # 只扫描
python3 clean_data.py --fix           # 修复
python3 clean_data.py --tables fund_valuation,daily_bars
python3 clean_data.py --fix --vacuum
```

检查项（逐表）：空表 / 非法·空日期 / 主键重复 / 非正价 / OHLC 结构错 / 全空行 / 公告日早于报告期 / 代码不在 stocks 表 / 行业代码无行情 / PE·PB=0 占位统计 / 抓取进度统计。

实际发现与处理：
- `daily_bars` 46 处低价基金/老股 OHLC 舍入瑕疵 → **最小修复**（high/low 撑到包住 open/close，不删 K 线），复查 0 问题。
- 东财 4 张报表 164 行 `NOTICE_DATE=1900-01-01` 占位符 → **置空**（训练按缺失跳过，防未来函数）。
- `fund_valuation_baidu` 193 万"非正 value"**不是脏数据**：亏损股 PE/PB/PCF 为负属正常，已调整规则（误报清零）。百度数值有 ±10000 截断痕迹，训练侧做 clip。
- `stock_industry` 70 只映射代码已不在 stocks 表（退市/转板）：只报告不处理。
- `lhb_detail` / `yjyg` 的 (code, date) 重复是**合法多行语义**（同一天多个上榜原因 / 同一报告期多个预测指标），全表 0 完全相同行；特征构建侧已按"求和/计数、均值"聚合消化。
- 其余表（index/industry/valuation/abstract/indicator/fhps/gdhs）0 问题。
- 日K复权口径清洗见 stock_predict 项目的 `data_clean.py`（本库 `daily_bars` 已是其迁移后的 hfq 数据，`adj_done` 7105 只，无需重跑）。

---

## 5. 特征工程

全部特征只用 t 日收盘及以前数据（point-in-time，规则见 5.10）。落盘为 `ablation_output/feat_<组>.npy`（float32），训练时按配置逐组读入，峰值内存 ≈ 单配置矩阵。

### 5.1 tech（40 列）：个股价量技术指标
ret1/5/10/20/60；ma5/10/20/60 偏离、ma5_20、ma20_60；vol20/vol60/volr；rsi6/rsi14；macd_dif/dea/hist；kdj_k/d/j；bollpos、hilo20、atr_n；vr5/vr20、logvol、amp、gap、cpos、updays20；横截面排名 r_ret5/r_ret20/r_ret60/r_ma20r/r_rsi14/r_vr20/r_vol20/r_logvol。
（原 44 列中的 mkt1/mkt5/mkt20/breadth5 划归 market 组。）

### 5.2 pattern（19 列）：K 线形态
`p_body/p_upper_sh/p_lower_sh`（实体/影线相对前收）、`p_body_pos`、doji、hammer、shooting、bull/bear_engulf、three_up/dn、up/dn_streak（连阳/连阴计数）、limit_up/dn20（20 日涨跌停计数）、yang_ratio20、gap_up20（向上跳空计数）、dist_high/low20（距20日高低点）。

### 5.3 market（28 列）：大盘
mkt1/mkt5/mkt20/breadth5（全市场等权收益+上涨家数占比）＋ 4 个指数（上证/深成/创业板/沪深300）×（ret1/5/20/60、ma20r、vol20）。
⚠️ **全部是"每日常数"特征**（同一天所有股票值相同）——混入横截面模型会导致早期分裂全被其占据、日内预测趋同、TopK 退化（见 §11.4）。选股模型应剔除，择时另做。

### 5.4 industry（10 列）：行业
申万一级行业指数：ind_ret1/5/20、ma20r、vol20/volr、amt_r（成交额相对20日均）、相对市场 ind_ret5/20_rel、行业动量横截面排名 ind_rank20。
行业映射用**当前快照**（`stock_industry` 的 start_date 是 SW2021 指数纳入日而非行业变更日——5220 只全 ≥2021-12，严格 asof 会清空 2018–2021 年特征，弊大于利；残余偏差是成分股历史变更）。

### 5.5 fund（29 列）：基本面
- 估值 7 列（`fund_valuation` 按日精确对齐、缺失向前填）：ep=1/pe_ttm（clip±1）、bp=1/pb（0–5）、sp=1/ps、cfp=1/pcf、peg、log_mv、float_ratio。
- 财报 22 列（`fund_indicator_em` **按 NOTICE_DATE−1 天 asof**，只用公告日严格早于 t 的报告）：roe、roe_kcfj、毛利率/净利率/ROA、营收/净利/扣非同比、eps、bps、ocf_ps、资产负债率、流动/速动比率、现金流占比、经营现金流/营收、应收/营收、周转天数（总资产/存货/应收）、roic、total_roi。极端值截断 ±1000，缺失保留 NaN 给 LightGBM。
- 百度长历史估值、新浪 86 指标、财务摘要：**训练未用**（周频/无公告日）。

### 5.6 flow（7 列）：资金流
main/xl/l/d/s 净流入占比 + main_5/20、xl_5 均值；**组内错后 1 个交易日**（盘后披露）。
⚠️ 仅覆盖 2026-04 后约 120 个交易日、197 只股票——当前基本无信息量。

### 5.7 margin（6 列）：两融
融资余额 1/5/20 日变化、融资买入偿还比、融券余额 5 日变化、log(融资余额)；**T+1 披露，错后 1 个交易日**；2010-03 起全量（SSE 4010 天 / SZSE 3772 天）。

### 5.8 lhb（12 列）：龙虎榜
28/84 天窗口：上榜次数、净买额（亿）、平均换手率、平均净买占比、上榜原因计数（涨幅偏离/跌幅偏离/连续三日涨/连续三日跌/换手率/无涨跌幅限制）。**只用严格早于 t 的上榜记录**（`searchsorted side='left'`），窗口按自然日（28/84 天 ≈ 20/60 交易日）。

### 5.9 events（4 列）：业绩事件
`yjyg`：预告类型打分（预增/扭亏等 +1，预减/首亏等 −1）+ 变动幅度（clip±100%）；`yjkb`：净利润同比（clip±500）+ 每股收益。**公告日 +1 天生效**，事件只保留 150 天。

### 5.10 PIT 对齐总表

| 数据 | 对齐方式 |
|---|---|
| 技术/形态/指数/行业 | 仅用 t 及以前；横截面排名限当日 |
| 每日估值 | 按日精确对齐（t 收盘可算） |
| 财报指标 | 公告日 −1 天 asof |
| 资金流/两融 | 组内 shift(1) 后 asof |
| 龙虎榜 | 事件日严格 < t |
| 业绩预告/快报 | 公告日 +1 天，150 天过期 |

---

## 6. 标签与样本切分

- **标签**：`R_h = close[t+h]/open[t+1] − 1`（t+1 开盘买入、持有 h 天），`y = R_h > 0`。h=1/5/10。
- **可交易门槛**（与 t+1 规则一致，沿用原脚本）：次日有行情、停牌间隔 ≤5 天、t+1 开盘非涨停封板（`open1 < close·(1+lim−0.005)`）。
  - 涨跌停幅度：主板 10%、创业板/科创板 20%、北交所 30%、ST 5%；**创业板 20% 仅 2020-08-24 起**（之前 10%，69 万根已按日期分段修正）；ST 按当前名称回溯（含历史误差，仅影响样本筛选）。
  - h>1 额外要求：持有窗内无 >5 天停牌、i+h 仍是同只股票。
- **切分**（全部脚本统一）：训练 **2018-01-01 ~ 2023-12-31**，验证 **2024-01-01 ~**，按年硬切（断言 0 重叠；训练最大 2023-12-29 / 验证最小 2024-01-02）。
- 训练样本时间衰减权重（半衰期 3 年）；验证集默认全量（`--train-sample/--valid-sample` 为 0 即全量，约 594 万/347 万行）。

---

## 7. 模型

### 7.1 LightGBM（二分类上涨概率）
`objective=binary, metric=[auc, binary_logloss], lr=0.03, leaves=31, min_data_in_leaf=1000, feature_fraction=0.6, bagging 0.7, lambda_l2=20, max_bin=127`，验证集 early stopping 150 轮。全量单配置约 5–15 分钟（8 核）。

### 7.2 TCN（时序形态）
膨胀因果卷积 + 残差块（`Chomp1d` 保证因果性），默认 `channels=64×4, dilations=1,2,4,8, kernel=3, lookback=20, dropout=0.1`，AdamW(lr=1e-3, wd=1e-4)，BCEWithLogits + 样本权重，验证 AUC 选最优。输入为个股最近 L 天特征序列（按股票起始位置截断，不跨股）。MX350 2G /a100 均可跑，`--device cuda`。

### 7.3 融合
`p = (1−w)·p_lgbm + w·p_tcn`（概率融合）＋横截面排名融合，与单模型分别回测对比。

---

## 8. 训练脚本

| 脚本 | 用途 | 关键参数 |
|---|---|---|
| `lgbm_train_backtest.py` | LGBM 基线（44 技术特征）+ 训练/验证分段回测 | `--limit/--rounds/--leaves/--lr` |
| `lgbm_tcn_ensemble.py` | LGBM 复用 + TCN 训练 + 三路回测对比 | `--lookback/--channels/--dilations/--tcn-weight/--device/--resume` |
| `train_ablation.py` | **主脚本**：9 特征组消融 + 多周期 | `--mode full/single/cum/loo/horizon`，`--groups`，`--horizons 1,5,10`，`--per-group`，`--train-sample/--valid-sample`（0=全量），`--threads/--throttle/--temp-limit/--temp-resume` |
| `train_tcn_pattern.py` | TCN（序列）vs LGBM（标量）对照：pattern/tech/tech+pattern 同采样对比 | `--configs/--lookback/--epochs/--batch-size/--skip-lgbm/--no-parallel` |
| `analyze_lhb.py` | 龙虎榜深挖：分年度/分月/股票池对照/IC | 直接运行 |
| `features_extra.py` | flow/margin/lhb/events 四组构建器（被消融脚本调用） | — |
| `eval_calibration.py` | 校准评估：校准曲线/ECE/Brier/PR-AUC/阈值精度 | 直接运行（只读） |
| `eval_calibrate.py` | Platt/Isotonic 实验：2024 拟合 → 2025–2026 评估 | 直接运行（只读） |

消融模式：`single`（单组）/ `cum`（逐组累加）/ `loo`（留一）/ `full`（三者全）；`horizon` 模式每个周期独立训练（特征=所选全部组），可选 `--per-group` 让每组每周期都训（9×3=27 模型找每周期最强）。

---

## 9. 回测方法

**全部 T+1 执行，无日内换仓**：t 收盘出信号 → t+1 开盘买 → h=1 当天收盘卖 / h>1 持有到期。成本万 15（单程各半）。

- `run_period`（h=1）：等权 TopK（5/10/20/50）+ 概率阈值（0.50/0.55/0.60，入选不足 5 只则空仓）+ 市场等权基准；输出年化（毛/净/超额）、Sharpe、MDD、日胜率、换手率、概率分档、年度 AUC、RankIC。
- `run_hold_backtest`（h=5/10）：**重叠持仓**——每日信号选 TopK（10/20/50），持有 h 天；每日组合收益 = 在持 vintage 当日收益等权平均；成本按进/出分摊（与单程口径一致）。仍是每日开盘调仓，无日内。
- tradable 过滤用 t+1 开盘价判断涨停买不到——开盘集合竞价可见，**可执行，不算前视**；回测曲线按信号日 t 记账只是展示惯例。

---

## 10. 评估指标

AUC / LogLoss / accuracy / base_rate / RankIC（逐日横截面 rank 相关，对比 T 日实际收益）/ ICIR / Brier / ECE（10 等宽档）/ PR-AUC / 阈值精度（P≥t 的实际胜率与样本数）。

---

## 11. 实验结论

验证集 2024-01 ~ 2026-09，市场等权基准年化 **33.4%**（大牛市，结论需结合 regime 看）。

### 11.1 特征组消融（LGBM 全量，25 配置）
| 配置 | 特征 | AUC | IC | Top20年化净/Sharpe |
|---|---|---|---|---|
| **tech** | 40 | **0.5440** | 0.0139 | 31.1% / 1.07 |
| tech+pattern | 59 | 0.5438 | 0.0166 | 20.4% / 0.78 |
| 全 9 组 | 147 | 0.5331 | 0.0119 | 9.4% / 0.44 |
| pattern | 19 | 0.5280 | 0.0054 | **44.5% / 1.26** |
| industry | 10 | 0.5332 | -0.0125 | −6.8% |
| market | 28 | 0.5241 | NaN（退化） | 退化 |
| fund | 29 | 0.5117 | 0.0128 | 12.4% / 0.50 |
| lhb（单组，旧 4 列版本） | 4 | 0.5022 | -0.0047 | Top20 **169.0%** / 2.75（股票池 beta，见 §11.4；深挖后已扩展为 12 列） |
| margin/events/flow | 4–7 | 0.500–0.507 | ≈0 | — |

- 加特征普遍稀释 T+1 效果；`market` 组（每日常数）会占据早期分裂、日内预测趋同、TopK 退化——选股模型应剔除，择时另做。

### 11.2 多周期（每周期独立模型，全验证集 IC，重叠持仓回测）

预测质量（各周期内按 AUC 排前几）：

| 周期 | 最强组 | AUC | IC | 次强 |
|---|---|---|---|---|
| T+1 | **tech** 0.5440 / 0.0145 | all 0.5404 / 0.0087 | industry 0.5339 / −0.0116 |
| T+5 | **tech** 0.5560 / 0.0461 | all 0.5492 / −0.0056 | market 0.5480 / 0（退化） |
| T+10 | tech 0.5645 / 0.0665（真实最强） | all 0.6007 / 0.0586（指数择时虚高） | industry 0.5471 / −0.0046 |

重叠持仓回测 Top20（年化净 / Sharpe / 回撤 / 日胜率，全验证集 665 天；T+10 全组已验证，T+1/T+5 待当前后台任务跑完后补）：

| T+10 配置 | 年化净 | Sharpe | 回撤 | 日胜率 |
|---|---|---|---|---|
| lhb | **115.3%** | 1.86 | −0.42 | 53.1% |
| fund | **110.0%** | 1.60 | −0.46 | 52.0% |
| market（=flow，逐位相同） | 91.9% | 1.60 | −0.43 | 52.9% |
| tech | 20.0% | 0.81 | −0.36 | 55.3% |
| margin | 15.8% | 0.60 | −0.45 | 55.0% |
| all（155 特征） | 14.8% | 0.61 | −0.42 | 55.5% |
| pattern | 8.0% | 0.40 | −0.35 | 54.6% |
| industry | 3.1% | 0.25 | −0.26 | 54.3% |

判读：
- 排序最准的是 tech（IC 0.0665），但持有收益只有 20%——它挑的股票价差小；lhb/fund 挑的是高 beta 妖股/困境反转，牛市里涨幅巨大。**IC 看方向，收益看幅度**，两者脱节是正常的。
- market=flow 两行逐位相同：恒定预测导致并列按行序取前 N 只，退化为固定篮子——再次证明 market 组禁入选股模型（框架自检时发现）。
- 高收益全是高波动换的（回撤 −0.36~−0.46），且全部处于 2024–2026 牛市样本内。

### 11.3 TCN vs LGBM（同 100 万采样行 / 30 万验证行）
| 配置 | TCN AUC | LGBM AUC |
|---|---|---|
| pattern（序列 vs 标量） | **0.5319** | 0.5280 |
| tech | 0.5372 | **0.5440** |
| tech+pattern | 0.5417 | **0.5438** |
TCN 只在 pattern 序列（形态组合）上略胜；TCN 的 P≥0.55 回测 47.4%（采样口径）。CPU+GPU 并行：TCN 跑 GPU 时 LGBM 基线在 CPU 线程同时跑（`--no-parallel` 可关，`--skip-lgbm` 只跑 TCN）。

### 11.4 龙虎榜深挖（179% 真伪）
分年度（模型 Top20）：2024 **+139.2%** / 2025 **+281.9%** / 2026 **+60.4%**；
对照（近28天上过龙虎榜的股票等权，完全不看模型）：**+100.0% / +132.2% / +44.7%**（基准 +15.3%/+72.6%/+14.5%）；
模型 RankIC **−0.0044**（分年 +0.005/−0.010/−0.009）；月胜率 79%，最差月 −21%，最好 10 天占总收益 26.4%。
**结论：收益大头是股票池 beta（2024–2026 游资/妖股牛市），不是排序 alpha**；P≥0.55 阈值策略为负（−1.3%）。当 regime 指标用，别当独立 alpha。

### 11.5 融合
LGBM+TCN 概率融合与排名融合均跑过（见 `ensemble_output/`）；受 TCN 未校准拖累（见 §12），提升有限。

---

## 12. 概率校准

验证集 347 万样本实测（`reports/校准评估_*.md`）：

| 模型 | AUC | Brier | ECE | P≥0.55 实际胜率 |
|---|---|---|---|---|
| LGBM | 0.5422 | 0.2496 | 0.0221 | 55.0%（0.6→59.2%，单调可靠） |
| TCN | 0.5242 | 0.2728 | **0.1204** | 51.8%（预测 0.08–0.35 的档实际全是 ~0.47） |
| 融合 | 0.5320 | 0.2542 | 0.0584 | 被 TCN 带偏 |

TCN 的 LogLoss 0.755 比瞎猜基准（0.693）还差——典型深度模型过度自信。
Platt/Isotonic 实验（2024 拟合 → 2025–2026 评估，223 万样本，`reports/校准对比_*.md`）：
**Platt 全面胜出**，Isotonic 反而略差。TCN 经 Platt 后 ECE 0.118→**0.014**、LogLoss 恢复正常（代价是高分区被压平：P≥0.6 几乎无样本——它的"高分"原本就是幻觉）；LGBM P≥0.55 胜率 53.4%→**56.9%**（样本 34.5 万→17 万，更挑剔）；融合 P≥0.55 51.7%→55.6%、P≥0.6 达 66.1%。
落地建议：训练 2018–2022 / 2023 拟合 Platt / 2024+ 评估回测，并对校准概率重搜阈值（阈值含义会变）。

---

## 13. 前视偏差审计

方法：代码通读 + 库内抽查 + 程序化断言（训练/验证 0 重叠、特征因果现货核对）+ 300 只冒烟回归。

**通过项**：切分按年硬切互斥；44 技术指标全 t 及以前 rolling（现货核对 `ret1[t]=close[t]/close[t-1]−1` 精确一致）；标签只用 t+1；训练只用 `fund_valuation` + `fund_indicator_em`（财务摘要/新浪 86 指标虽在库但**训练未用**）；市场/行业/资金组 PIT 对齐（§5.10）；TCN 序列按股票起点截断、因果回看；复权只用比率；标准化只用训练集统计；`fund_valuation` 同日对齐属标准 PIT。300 只冒烟全通。

**修了 2 个真问题**：
1. 创业板 20% 被用到 2020-08-24 之前（69 万根）→ 已按日期分段（单元测试 `[0.1, 0.2, 0.1, 0.2]` 通过）。影响：tradable 筛选变严，重跑历史回测数会有轻微变化。
2. 财报 asof 含"同日公告"（`side='right'`）→ 已改为公告日 −1 天，严格 PIT。

**保留现状（已在代码注记）**：
- 行业映射用当前快照：`start_date` 全 ≥2021-12（那是 SW2021 指数纳入日而非行业变更日），严格 asof 会清空 2018–2021 年行业特征，弊大于利。
- ST 5% 按当前名称回溯（需历史改名数据才能精确），仅影响样本筛选。
- 早停/选型都在验证集上：常规做法，最终定稿建议加 holdout 或 walk-forward。
- lhb/yjyg 的 (代码,日期) 重复是合法多行语义（不同上榜原因/预测指标，全表 0 完全相同行），构建器已聚合消化。

---

## 14. 运维工具

- **温度看门狗**（`lgbm_train_backtest.thermal_wait/ThermalCallback`，读 coretemp hwmon）：`--temp-limit/--temp-resume` 超温自动挂起降温（`--throttle` 每轮/每批休眠，`--threads` 限线程）。MX350（Pascal, sm_61）可用 CUDA；注意整机散热——本机待机 76–80°C，曾因 3.2GHz 全核 + 训练触发过热强关。
- **进度**：`python3 progress_ablation.py [ablation.log|ablation2.log]`（百分比/ETA）；`grep -aE "h=|完成" horizon.log | tail -5`；`grep -a 进度 extra.log|fundamental.log | tail -1`。
- **监控脚本**：`bash watch_progress.sh [秒]`（下载/训练/GPU/温度同屏；注意 watch 外层用单引号）。
- **swap**：`sudo bash setup_swap.sh`（新增 28G swap 文件凑满 32G，训练中可安全执行；vm.swappiness=20）。

---

## 15. 环境与运行命令

- Python 3.14，lightgbm 4.7.0，torch 2.14.1+cu126，akshare 1.19.1，pandas 3.0.6，numpy 2.5.3，scikit-learn 1.9.1，tabulate 0.10.0。机器：14G 内存 / 8 核 / MX350 2G。
- 全量消融：`python3 train_ablation.py --mode full`（13 配置，~1.5h；9 组 25 配置约 2–3h）
- 多周期：`python3 train_ablation.py --mode horizon --groups tech,pattern,industry,fund,flow,margin,lhb,events`（T+1/5/10，~1h）；加 `--per-group` 找每周期最强组（30 模型，~3–4h）
- TCN 对比：`python3 train_tcn_pattern.py --configs pattern,tech,tech+pattern --epochs 5`（GPU）
- 龙虎榜深挖：`python3 analyze_lhb.py` → `ablation_output/lhb_deepdive.md`
- 校准：`python3 eval_calibration.py` / `python3 eval_calibrate.py` → `reports/`
- 清洗：`python3 clean_data.py --fix`

---

## 16. 输出文件说明

| 目录 | 内容 |
|---|---|
| `lgbm_output/` | LGBM 基线：模型/重要度/回测汇总/逐日净值/分档/年AUC/**逐样本验证预测**（校准用） |
| `ensemble_output/` | LGBM+TCN：三路评估与回测、融合权重扫描、逐样本预测 |
| `ablation_output/` | 消融：`ablation_results.csv`（排序对比）、`ablation_importance.csv`、`horizon_results.csv`（+`config`列）、`holdbacktest_summary/daily.csv`、`lgbm_all.txt`、`lgbm_h{h}[_all].txt`、`lhb_deepdive.md`、`lhb_monthly.csv`、`run_config.json` |
| `tcn_output/` | `tcn_results.csv`、`backtest_report.md`、`backtest_summary.csv` |
| `reports/` | 清洗报告 / 校准评估 / 校准对比（日期命名） |

---

## 17. 已知限制与下一步

1. `flow`（资金流）只覆盖 197 只：push2his 经代理不稳定（`ProxyError`），直连偶发 `RemoteDisconnected`；网络好时用 `NO_PROXY=.eastmoney.com` 重跑 `fetch_extra_data.py --datasets flow`。
2. T+1 最强仍是 tech 40 特征（0.5440）；T+5/T+10 真实技能在 tech（IC 0.046/0.067，fund 的 IC 接近可供参考），industry 的 IC 为负、market 组禁入选股模型。
3. 基本面缺横截面标准化/行业中性化；`fund` best_iter=1 的单树高 IC 需分年度验证。
4. 最终定稿建议：walk-forward 多窗口 + holdout；Platt 接入训练流程（2018–2022 训练 / 2023 校准 / 2024+ 评估）并重搜阈值；lhb/fund 高 beta 组合配 regime 过滤与回撤控制。
5. T+5/T+10 重叠持仓回测已实现（`holdbacktest_*.csv`，Top10/20/50），但阈值类持有策略未做。
6. 硬件：散热是瓶颈（待机 76–80°C 偏高），长跑任务建议开限温 `--temp-limit 92 --temp-resume 85` + 换硅脂/清灰。
