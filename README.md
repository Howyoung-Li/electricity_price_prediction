# 小时级电力价格预测项目

德国/Luxembourg bidding zone 日前电价小时级预测项目。项目目标是跑通一个完整的电价预测流程，并客观评估 ERA5 气象再分析数据对电价预测的边际贡献。

这里的“日前电价预测”指的是：在今天市场出清前，预测明天 24 个小时分别对应的电价。模型表按小时展开，因此测试集中每一行对应 2023 年某个小时的 day-ahead price。

## TL;DR

- 本项目完成了一个小时级日前电价预测流程：用 2019-2022 年训练/验证，预测 2023 年逐小时 day-ahead price。
- 全年平均 MAE 上，`with ERA5` 与 `without ERA5` 的最优模型几乎打平：`19.1792` vs `19.1754`。这说明直接加入 ERA5 reanalysis 并不会稳定提升全年点预测误差。
- 但气象数据并非没有价值。分模型看，ERA5 改善了 XGBoost、LightGBM、ElasticNet 和 LSTM；在事件指标上，with ERA5 的 XGBoost 对负电价召回率更高，能抓住约 `75.7%` 的真实负电价小时，而 without ERA5 的 CatBoost 为 `67.0%`。
- SHAP 显示，电价预测的主导变量仍是残余负荷、历史电价滞后、滚动价格 regime、负荷预测和新能源占比。气象数据更可能通过“改进新能源功率预测和 residual load 预测”间接提升电价预测，而不是简单作为一组额外表格变量直接加入。
- 对中国市场迁移时，重点不只是复用模型，而是适配中国正在推进的中长期、现货、辅助服务、新能源入市和价格结算机制。随着新能源更多进入市场，负电价、低价时段、尖峰风险、偏差结算和场景化风险预测会比单一 MAE 更重要。
- 实时或准实时卫星气象数据的潜在价值在于：更接近交易时点可用信息，能更好刻画云、辐照、风场和短时天气变化；这类信息尤其可能提升负电价、高光伏午间低价、高风出力、尖峰前后等场景的识别能力。

## 项目结论概览

本项目比较了两组特征：

- `with ERA5`：历史电价、日历、负荷/新能源预测、PVGIS-SARAH3、ERA5 气象再分析特征。
- `without ERA5`：去掉 ERA5，保留其他默认特征。

正式实验设置：

- 训练/验证：2019-2022
- 测试：2023
- 粒度：1 小时
- 调参：每个主要模型随机搜索 `max_trials=10`
- 模型：ElasticNet、XGBoost、LightGBM、CatBoost、LSTM、RNN、OOF stacking
- 主要指标：MAE、RMSE、SMAPE、R2、方向准确率、负电价 precision/recall、高价前 10% MAE

全局最优结果几乎打平：

| 特征组 | 最优模型 | MAE | RMSE | 方向准确率，lag24 | 负电价召回率 | 高价前10% MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| with ERA5 | XGBoost | 19.1792 | 26.4897 | 0.7450 | 0.7567 | 97.2037 |
| without ERA5 | CatBoost | 19.1754 | 26.1358 | 0.7553 | 0.6700 | 98.8439 |

`with ERA5 - without ERA5` 的最优 MAE 差值约为 `+0.0038`，可以认为在全年平均 MAE 上没有显著差异。

但分模型看，ERA5 并非没有价值：

| 模型 | MAE with ERA5 | MAE without ERA5 | delta | 说明 |
| --- | ---: | ---: | ---: | --- |
| XGBoost | 19.1792 | 19.3120 | -0.1328 | ERA5 小幅提升 |
| LightGBM | 19.7003 | 20.4938 | -0.7934 | ERA5 提升明显 |
| ElasticNet | 20.1027 | 20.4528 | -0.3501 | ERA5 小幅提升 |
| LSTM | 20.2677 | 22.3757 | -2.1080 | ERA5 对 LSTM 提升最大 |
| CatBoost | 20.9432 | 19.1754 | +1.7679 | ERA5 反而变差 |
| RNN | 23.1357 | 21.7342 | +1.4015 | ERA5 反而变差 |

负数表示加入 ERA5 后 MAE 下降。结论是：ERA5 对部分模型有效，但不是“加入气象变量就稳定提升”。在已有负荷预测、风电预测、光伏预测等市场侧特征时，ERA5 的边际贡献会被这些特征吸收一部分。

## 核心可视化

完整图集见：

[reports/figures/era5_ablation_max10](reports/figures/era5_ablation_max10)

详细可视化报告见：

[reports/modeling/era5_ablation_max10_visual_report.md](reports/modeling/era5_ablation_max10_visual_report.md)

### 2023 全年小时级趋势

全年小时级曲线非常密集，适合观察整体 regime 和尖峰，但不适合看局部拟合细节。

![2023 hourly trend](reports/figures/era5_ablation_max10/best_models_2023_hourly_trend.png)

### 最后 240 小时短窗口

短窗口更适合检查模型是否跟得上局部趋势、日内形状和尖峰。

![last 240 hours](reports/figures/era5_ablation_max10/best_models_2023_last_240h_trend.png)

### 7 天滚动趋势

滚动趋势用于观察模型是否有系统性偏高或偏低。

![7-day rolling trend](reports/figures/era5_ablation_max10/best_models_2023_rolling_7d_trend.png)

### 残差趋势

![rolling residuals](reports/figures/era5_ablation_max10/best_models_2023_rolling_residuals.png)

### 月度 MAE

![monthly mae](reports/figures/era5_ablation_max10/monthly_mae_best_models.png)

### SHAP Summary

红色表示该特征取值较大，蓝色表示该特征取值较小。横轴是 SHAP value，表示该特征对预测电价的正向或负向贡献。

with ERA5 最优模型 XGBoost：

![shap with era5](reports/figures/era5_ablation_max10/shap_summary_with_era5_xgboost.png)

without ERA5 最优模型 CatBoost：

![shap without era5](reports/figures/era5_ablation_max10/shap_summary_without_era5_catboost.png)

两个模型的 SHAP 都显示，最重要的变量主要是：

- `residual_load_forecast`：残余负荷预测，是电价最核心的供需压力变量。
- `price_lag_24h`、`price_lag_168h`：昨天同小时、上周同小时价格。
- `price_rolling_*`：历史电价均值、波动率、最大值，捕捉价格 regime。
- `renewable_share_forecast`、`wind_share_forecast`：新能源占比相关变量。
- `load_forecast`：负荷预测。

这说明电价预测里，市场供需变量和历史价格 regime 仍然是主导信息。气象变量的价值更可能体现在特定场景，例如高风、高光伏、低残余负荷、负电价、价格尖峰和天气突变。

## 气象数据分析：ERA5 的价值与局限

本项目使用 ERA5 的风、热力、云、太阳辐射相关变量，并把德国区域内网格气象数据聚合成小时级特征。实验结果说明：

1. ERA5 对某些模型有帮助，尤其是 LightGBM、XGBoost、ElasticNet、LSTM。
2. ERA5 对 CatBoost 和 RNN 在本次设置下反而造成性能下降。
3. 全年平均 MAE 上，最优 with/without ERA5 基本打平。

这不是“气象数据无用”，而是说明当前 ERA5 使用方式有局限：

- ERA5 是 reanalysis，再分析数据，偏向事后重建的真实大气状态，不是交易时点真实可获得的预报数据。
- 日前电价预测应使用“今天能拿到的明天天气预报”，而不是事后再分析天气。
- 当前 ERA5 是区域统计聚合，没有按风电、光伏装机空间分布加权，可能稀释局部有效信号。
- 数据里已有负荷预测、风电预测、光伏预测，这些变量本身已经吸收了大量气象信息。
- ERA5 变量多且相关性强，可能给高容量模型带来噪声、冗余和过拟合。
- 2023 年电价 regime 与 2019-2022 差异较大，历史价格滚动特征存在显著漂移。

### 实时卫星气象数据的潜在改进方向

与 ERA5 reanalysis 相比，实时或准实时卫星气象数据更接近真实预测业务中的信息条件，潜在改进方向包括：

- **信息时点更真实**：使用交易前可获得的卫星观测和临近预报，避免 reanalysis 的事后信息问题。
- **云与辐照更直接**：光伏预测对云场、云移动、云光学厚度、辐照变化高度敏感，卫星云图可以提供更高频、更直接的观测。
- **空间分辨率更高**：如果结合光伏/风电装机分布加权，卫星数据可能比区域平均 ERA5 更有效。
- **短时突变更敏感**：卫星云图序列可用于 nowcasting，提升未来数小时光伏和局地天气变化预测。
- **改善功率预测再传导到电价**：卫星数据未必直接进入电价模型效果最好，更合理的链路是先改进风光功率预测，再通过 residual load、renewable share 等变量传导到电价预测。
- **场景价值更明显**：实时卫星数据可能主要改善负电价、高光伏午间低价、高风出力、尖峰前后、天气快速变化等特定场景，而不是平均 MAE。

因此，本项目对气象数据的结论是：直接加入 ERA5 reanalysis 特征并不会稳定提升全年电价 MAE，但这并不否定气象数据的价值。更合理的技术路线是构建交易时点可用的高分辨率气象特征，优先改善新能源功率预测和 residual load 预测，再评估其对电价尖峰、负电价和高波动时段的增益。

## 特征漂移

![feature drift](reports/figures/era5_ablation_max10/feature_drift_top30_psi.png)

漂移最强的是历史电价滚动统计，尤其是 rolling volatility、rolling mean、rolling max。这说明 2023 年价格分布与 2019-2022 训练/验证期差异明显。对于电价预测，这是一个很重要的业务风险：模型可能学到历史价格 regime，但遇到新的能源价格、政策、供需结构时泛化能力下降。

## 中国市场迁移与政策语境

本项目使用德国/Luxembourg bidding zone 数据，是因为欧洲公开数据较完整，适合快速验证“气象数据是否提升电价预测”。迁移到中国市场时，不能只替换数据源，还需要适配中国电力市场的交易结构、价格机制和新能源消纳逻辑。

### 市场机制差异

中国电力市场正在从“中长期交易为主、现货交易逐步建设”向更完整的多层次市场体系演进。国家发展改革委、国家能源局发布的《电力现货市场基本规则（试行）》将电力现货市场定义为在日前及更短时间内开展的次日、日内至实时调度前的电力交易活动，并强调现货市场与中长期、辅助服务、容量补偿、代理购电等机制的衔接。

这意味着中国场景下的电价预测通常至少要区分：

- **中长期合约价格与分时曲线**：影响发用双方的基准收益和风险敞口。
- **日前现货价格**：反映次日分时供需、机组组合、联络线约束和新能源预期。
- **实时价格/偏差结算价格**：更受短时预测误差、机组故障、负荷波动和新能源偏差影响。
- **辅助服务与容量机制**：影响灵活性资源、储能、虚拟电厂等主体收益。

因此，迁移到中国市场时，本项目的小时级日前预测框架可以作为基础，但需要进一步扩展为“中长期-日前-实时-辅助服务”的联合风险预测框架。

### 新能源入市后的预测重点

2025 年《关于深化新能源上网电价市场化改革 促进新能源高质量发展的通知》提出，推动新能源上网电量进入电力市场、上网电价通过市场交易形成，并同步建立支持新能源可持续发展的价格结算机制。这一方向会提高新能源发电侧对市场价格、偏差风险和出力预测质量的敏感度。

在这种语境下，电价预测的目标不应只看全年平均 MAE，还应加入更贴近业务的场景指标：

- **负电价召回率**：真实负电价小时中，有多少被模型预测为负电价。它衡量模型是否能抓住新能源大发、低负荷、系统调节不足等低价风险。
- **低价/零价时段识别**：服务新能源报价、储能充电、可调负荷响应。
- **尖峰价格召回率**：服务火电、储能、虚拟电厂和售电公司风险控制。
- **偏差方向准确率**：判断价格相对昨日同小时或合约基准是上行还是下行。
- **分省/分区/节点迁移能力**：中国市场省级规则差异较大，模型需要适配本地市场规则和网架约束。

本项目中，with ERA5 的最优模型虽然全年 MAE 没有显著领先，但负电价召回率更高。这类结果在中国新能源入市后具有实际意义：如果模型能更早识别低价或负价风险，即使平均 MAE 提升有限，也可能改善新能源报价、储能调度和偏差风险管理。

### 实时卫星气象数据的作用路径

ERA5 是 reanalysis，不是交易时点真实可获得的预报数据。迁移到中国市场时，更有价值的数据形态是实时/准实时卫星观测、卫星反演产品、NWP 预报和地面站/场站数据的融合特征。

潜在技术路径包括：

1. **先改进新能源功率预测**  
   卫星云图、云光学厚度、云顶温度、辐照反演、风场和大气温湿压廓线可以改善光伏和风电预测。更准确的风光预测会进一步改善 residual load 预测，而 residual load 是本项目 SHAP 中最重要的电价驱动变量。

2. **提高极端场景识别能力**  
   负电价、低价午间、高风出力、天气突变和价格尖峰往往由气象驱动的供需快速变化触发。卫星数据的价值可能主要体现在这些场景指标上，而不是全年平均 MAE。

3. **做空间加权而不是简单区域平均**  
   中国新能源资源分布和负荷中心空间错配明显。卫星气象特征应按风电/光伏装机分布、送出通道、省间交易断面、负荷中心进行加权，而不是简单行政区平均。

4. **支持省级市场差异化建模**  
   不同省份新能源占比、火电灵活性、水电调节能力、外送通道、现货规则和价格限制不同。气象特征需要与本地市场规则、机组结构、联络线约束共同建模。

5. **从点预测扩展到风险预测**  
   对新能源企业、售电公司、储能和虚拟电厂而言，识别“价格会不会跌到低价/负价区间”或“是否会出现尖峰”往往比只降低 0.1-0.2 的 MAE 更有价值。

### 迁移后的数据需求

如果将本项目迁移到中国市场，建议的数据表按小时或 15 分钟构建，至少包括：

- 价格：日前、实时、中长期分时合约、偏差结算价格。
- 供需：系统负荷预测/实际、风光预测/实际、火电/水电/核电可用容量、检修计划。
- 网架：省间联络线计划、断面约束、外送/受入能力。
- 市场：中长期合约电量比例、辅助服务价格、限价规则、市场干预标识。
- 气象：NWP 预报、卫星云图/辐照/风场/温湿压反演、地面站、场站 SCADA。
- 政策与日历：节假日、迎峰度夏/冬、需求响应、限电或保供事件。

## 结论

本项目的主要结论不是“ERA5 一定提升电价预测”，而是：

1. 电价预测首先由市场供需、残余负荷、历史价格 regime 和新能源占比驱动。
2. 直接加入 reanalysis 气象变量的全年 MAE 提升有限，甚至可能因噪声、冗余和信息时点不一致导致部分模型变差。
3. 气象数据的价值更适合通过实时化、空间加权、场景化指标和新能源功率预测链路体现。
4. 对中国市场而言，随着电力现货市场建设、新能源入市和价格市场化推进，负电价召回率、低价时段识别、尖峰风险识别、偏差方向准确率等指标会越来越重要。
5. 卫星气象数据如果能在交易时点提供高频、高覆盖、可追溯的天气观测和临近预报，最可能在新能源出力预测、residual load 预测和低价/负价/尖峰场景识别中形成增量价值。

相关政策背景：

- [《电力现货市场基本规则（试行）》](https://www.ndrc.gov.cn/xxgk/zcfb/ghxwj/202309/t20230915_1360625_ext.html)
- [关于进一步加快电力现货市场建设工作的通知](https://www.ndrc.gov.cn/xxgk/zcfb/tz/202311/t20231101_1361704.html)
- [关于深化新能源上网电价市场化改革 促进新能源高质量发展的通知](https://www.gov.cn/zhengce/zhengceku/202502/content_7002959.htm)

## 项目结构

```text
configs/
  modeling_config.json
scripts/
  train_tabular_models.py
  train_lstm_full.py
  visualize_era5_ablation.py
reports/
  modeling/
    era5_ablation_max10_summary.csv
    era5_ablation_max10_model_comparison.csv
    era5_ablation_max10_visual_report.md
    era5_ablation_max10_shap_importance_*.csv
    era5_ablation_max10_feature_drift_psi.csv
  figures/
    era5_ablation_max10/
data/
  predictions/
    era5_ablation_*_test_predictions_2023.csv
```

原始数据和中间处理数据体积较大，且 ERA5 下载依赖 CDS API 授权，本仓库默认不上传 `data/raw/` 与 `data/processed/`。复现实验时需要按 `DATA_CATALOG.md` 和脚本重新下载或准备数据。

## 运行方式

环境依赖包括：

- Python 3.9+
- pandas、numpy、scikit-learn
- xgboost、lightgbm、catboost
- torch
- shap
- matplotlib

训练正式消融实验：

```bash
bash scripts/run_era5_ablation_max10.sh
```

生成可视化与 SHAP：

```bash
python3 scripts/visualize_era5_ablation.py
```

## 相关说明文档

- [DATA_CATALOG.md](DATA_CATALOG.md)
- [PROJECT_PLAN.md](PROJECT_PLAN.md)
- [ENVIRONMENT_CHECK.md](ENVIRONMENT_CHECK.md)
