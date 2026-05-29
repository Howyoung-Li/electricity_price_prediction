# electricity_price_prediction

德国/Luxembourg bidding zone 日前电价小时级预测项目。项目目标是跑通一个完整的电价预测流程，并客观评估 ERA5 气象再分析数据对电价预测的边际贡献。

这里的“日前电价预测”指的是：在今天市场出清前，预测明天 24 个小时分别对应的电价。模型表按小时展开，因此测试集中每一行对应 2023 年某个小时的 day-ahead price。

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
