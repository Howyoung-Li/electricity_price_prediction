# 电价预测项目后续计划

目标：跑通德国 `DE-LU` 日前电价预测，并量化 ERA5 / PVGIS-SARAH3 卫星气象数据对预测的贡献。

当前状态：

- SMARD 电力市场数据已下载。
- PVGIS-SARAH3 卫星辐照数据已下载。
- ERA5 GRIB 已整理成小时级区域特征。
- Open-Meteo 天气数据已下载，可作为 ERA5 对照或备用。
- 统一建模主表 `data/processed/de_lu_hourly_2019_2023.csv` 已生成。
- 第一轮表格模型训练已跑通，结果见 `reports/modeling/tabular_model_results.csv`。

## 1. 数据资产整理

已完成：

- `data/raw/smard/`：日前电价、负荷、风电、光伏、日前预测。
- `data/raw/pvgis/`：五个德国代表城市的 PVGIS-SARAH3 辐照数据。
- `data/raw/era5/`：ERA5 GRIB 文件已重命名。
- `data/processed/era5/era5_de_lu_hourly_region_features_2019_2023.csv`：ERA5 区域小时特征。

已完成：

- 主键为 `timestamp_utc`。
- 所有时间统一 UTC，同时派生德国本地时间特征。

下一步：

- 根据消融实验重新生成不同 feature set 的训练结果。
- 加强 stacking 的 OOF 生成方式，避免当前简化 meta-training 带来的偏差。

## 2. 构建统一建模主表

输入：

- SMARD 电力变量。
- ERA5 区域气象变量。
- PVGIS-SARAH3 城市级卫星辐照变量。
- 德国日历、周末、节假日特征。

主表字段建议：

- 目标：`day_ahead_price`
- 电力基本面：`load_forecast`, `load_actual`, `solar_forecast_day_ahead`, `wind_onshore_forecast_day_ahead`, `wind_offshore_forecast_day_ahead`
- 实际出力：`solar_actual`, `wind_onshore_actual`, `wind_offshore_actual`
- ERA5：温度、露点、气压、降水、云量、10m/100m 风、阵风、辐射
- PVGIS：五城市 `Gb(i)`, `Gd(i)`, `Gr(i)`, `H_sun`, `T2m`, `WS10m`
- 时间：小时、星期、月份、周末、节假日、是否工作日

注意：

- 第一版可以同时保留 `actual` 和 `forecast`，但建模时要分清实验目的。
- 真实日前预测场景应优先使用日前可获得变量，如负荷预测、风光日前预测、天气预报或再分析代理。
- 使用实际发电和实际天气时，应把实验解释为“气象/基本面解释力上限”。

## 3. 特征工程

价格滞后特征：

- `price_lag_24h`
- `price_lag_48h`
- `price_lag_168h`
- `price_rolling_24h_mean/std`
- `price_rolling_168h_mean/std`

负荷和新能源特征：

- `residual_load_forecast = load_forecast - wind_forecast - solar_forecast`
- `renewable_forecast = wind_forecast + solar_forecast`
- `renewable_share_forecast = renewable_forecast / load_forecast`
- `wind_total_forecast = wind_onshore_forecast + wind_offshore_forecast`
- `wind_total_actual = wind_onshore_actual + wind_offshore_actual`
- `solar_forecast_error_proxy = solar_actual - solar_forecast_day_ahead`，仅用于误差分析或事后解释

气象特征：

- `temperature_2m_mean`
- `wind_speed_100m_from_mean_uv`
- `wind_gust_10m_mean/max`
- `total_cloud_cover_mean`
- `surface_solar_radiation_downwards_mean`
- `total_sky_direct_solar_radiation_at_surface_mean`
- `total_precipitation_mean`
- `air_density_approx_mean`

时间特征：

- `hour`
- `weekday`
- `month`
- `is_weekend`
- `is_holiday_de`
- `sin_hour`, `cos_hour`
- `sin_dayofyear`, `cos_dayofyear`

## 4. 实验设计

核心是消融实验，而不是只追求一个最低误差。

实验 A：历史价格基线

- 特征：日历 + 历史电价滞后 + rolling 统计。
- 目的：建立强时间序列 baseline。

实验 B：加入电力基本面

- 特征：A + 负荷预测 + 风光日前预测 + residual load。
- 目的：验证电力供需变量带来的增益。

实验 C：加入 ERA5 气象

- 特征：B + ERA5 风、温度、云、辐射、降水。
- 目的：验证再分析气象变量是否进一步改善电价预测。

实验 D：加入 PVGIS-SARAH3 卫星辐照

- 特征：B + PVGIS-SARAH3 五城市卫星辐照特征。
- 目的：评估卫星辐照对光伏相关电价时段的贡献。

实验 E：完整模型

- 特征：B + ERA5 + PVGIS-SARAH3。
- 目的：构建最终展示模型。

## 5. 数据切分

推荐切分：

- 训练集：2019-01-01 到 2021-12-31
- 验证集：2022-01-01 到 2022-12-31
- 测试集：2023-01-01 到 2023-12-31

原因：

- 2022 年能源危机波动强，适合作为验证模型稳健性的 regime shift 年份。
- 2023 年作为最终测试集，能展示新能源和价格波动下的泛化能力。

可选滚动验证：

- train 2019, validate 2020
- train 2019-2020, validate 2021
- train 2019-2021, validate 2022
- train 2019-2022, test 2023

## 6. 模型路线

模型配置统一保存在 `configs/modeling_config.json`。原则是：除 naive baseline 外，不事先固定单一超参数；每个模型必须用时间序列验证做超参数搜索。对 ElasticNet、LSTM、RNN 等需要尺度敏感的模型，训练前必须只在训练集上拟合标准化器。

基线模型：

- 历史同小时均值。
- 前一天同小时价格 `lag_24h`。
- 前一周同小时价格 `lag_168h`。

主力模型：

- LightGBM 或 XGBoost。
- ElasticNet 作为可解释线性基线。
- CatBoost 作为另一个 GBM 对照。

进阶模型：

- 分位数回归 LightGBM，输出 P10/P50/P90。
- CatBoost，对类别和非线性有时更稳。
- TCN / Transformer，用作时序深度学习对照，但不作为第一优先级。
- Stacking 1：用 GBM 作为 meta learner，融合各 tuned base model 的 out-of-fold 预测。
- Stacking 2：用非负权重线性融合 out-of-fold 预测，权重不要求和为 1，可带截距和 L2 正则。

## 7. 评价指标

整体点预测：

- `MAE`
- `RMSE`
- `sMAPE`

业务切片：

- 高价 top 10% 时段 MAE。
- 低价/负价时段识别准确率和召回率。
- 午间 10:00-15:00 光伏高发时段误差。
- 傍晚 17:00-21:00 爬坡时段误差。
- 冬季和夏季分季节误差。

解释性：

- permutation importance。
- SHAP。
- 分组特征重要性：历史价格、电力基本面、ERA5、PVGIS。

## 8. 卫星气象贡献分析

重点问题：

- PVGIS-SARAH3 是否改善午间低价/负价预测？
- ERA5 云量和辐射是否改善光伏高发时段价格？
- 100m 风速和阵风是否改善高风电时段价格？
- 气象变量对极端价格是否比对普通小时更有帮助？

推荐图表：

- 各实验 MAE/RMSE 对比柱状图。
- 按小时分组的误差曲线。
- 按新能源出力分位数分组的误差曲线。
- SHAP summary plot。
- 2023 年典型周价格预测曲线。
- 负价/尖峰事件案例分析。

## 9. 面试展示结构

建议按这个故事线讲：

1. 业务问题：新能源提高了电价对气象的敏感性。
2. 数据建设：用 SMARD + ERA5 + PVGIS-SARAH3 构建小时级主表。
3. 模型设计：从历史价格基线到电力基本面，再到气象/卫星消融。
4. 结果分析：不只看整体误差，还看负价、高价、午间光伏和风电高发切片。
5. 岗位关联：云遥宇航的卫星气象数据可通过改善风光预测和天气特征，进一步改善电价预测。

## 10. 下一次具体任务

已完成：

1. 写 `scripts/build_model_table.py`，合并 SMARD、ERA5、PVGIS。
2. 生成 `data/processed/de_lu_hourly_2019_2023.csv`。
3. 做主表质量检查：行数、缺失、时间连续性、变量分布。
4. 写并运行第一版 `scripts/train_tabular_models.py`。

建议下一步直接做：

1. 改进训练脚本，支持按 `configs/modeling_config.json` 中的 feature set 做 A/B/C/D/E 消融。
2. 将 stacking 改成真正的 time-series OOF stacking，而不是当前快速版 2022 meta-training。
3. 加入业务切片评估：负价、高价 top 10%、午间光伏、傍晚爬坡。
4. 写 PyTorch LSTM/RNN 训练脚本，并把结果纳入 stacking base model。
