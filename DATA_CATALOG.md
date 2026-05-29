# Data Catalog

本项目第一版数据已下载到 `data/raw/`，统一目标范围为：

- 市场区域：德国/卢森堡 `DE-LU`
- 时间范围：2019-01-01 00:00 UTC 到 2023-12-31 23:00 UTC
- 时间粒度：小时级
- 理论小时数：43,824

## 1. SMARD 电力市场数据

目录：`data/raw/smard/`

来源：Bundesnetzagentur SMARD public chart data endpoint

区域：`DE-LU`

粒度：`hour`

每个文件行数：43,824

| 文件 | 字段 | 含义 |
|---|---|---|
| `day_ahead_price_DE-LU_2019-01-01_2023-12-31_hour.csv` | `day_ahead_price` | 日前电价，EUR/MWh |
| `load_actual_DE-LU_2019-01-01_2023-12-31_hour.csv` | `load_actual` | 实际电网负荷，MWh |
| `load_forecast_DE-LU_2019-01-01_2023-12-31_hour.csv` | `load_forecast` | 负荷预测，MWh |
| `solar_actual_DE-LU_2019-01-01_2023-12-31_hour.csv` | `solar_actual` | 光伏实际发电，MWh |
| `wind_onshore_actual_DE-LU_2019-01-01_2023-12-31_hour.csv` | `wind_onshore_actual` | 陆上风电实际发电，MWh |
| `wind_offshore_actual_DE-LU_2019-01-01_2023-12-31_hour.csv` | `wind_offshore_actual` | 海上风电实际发电，MWh |
| `solar_forecast_day_ahead_DE-LU_2019-01-01_2023-12-31_hour.csv` | `solar_forecast_day_ahead` | 光伏日前预测，MWh |
| `wind_onshore_forecast_day_ahead_DE-LU_2019-01-01_2023-12-31_hour.csv` | `wind_onshore_forecast_day_ahead` | 陆上风电日前预测，MWh |
| `wind_offshore_forecast_day_ahead_DE-LU_2019-01-01_2023-12-31_hour.csv` | `wind_offshore_forecast_day_ahead` | 海上风电日前预测，MWh |
| `wind_solar_forecast_day_ahead_DE-LU_2019-01-01_2023-12-31_hour.csv` | `wind_solar_forecast_day_ahead` | 风光合计日前预测，MWh |

## 2. PVGIS-SARAH3 卫星辐照数据

目录：`data/raw/pvgis/`

来源：European Commission JRC PVGIS API，`PVGIS-SARAH3`

代表点：

- Berlin: 52.5200, 13.4050
- Hamburg: 53.5511, 9.9937
- Munich: 48.1351, 11.5820
- Frankfurt: 50.1109, 8.6821
- Cologne: 50.9375, 6.9603

每个城市文件行数：43,824

字段：

- `time`：PVGIS 时间字段，格式如 `20190101:0011`。后续合并时需转换为小时索引。
- `Gb(i)`：beam/direct irradiance on inclined plane。
- `Gd(i)`：diffuse irradiance on inclined plane。
- `Gr(i)`：reflected irradiance on inclined plane。
- `H_sun`：太阳高度。
- `T2m`：2m 气温。
- `WS10m`：10m 风速。
- `Int`：PVGIS 插值/数据标识字段。

说明：PVGIS-SARAH3 可作为本项目“卫星辐照数据贡献”的主要公开代理数据源。

## 3. Open-Meteo 历史天气数据

目录：`data/raw/open_meteo/`

来源：Open-Meteo Archive API

代表点：与 PVGIS 相同的五个德国城市。

每个城市文件行数：43,824

字段：

- `temperature_2m`
- `wind_speed_10m`
- `wind_speed_100m`
- `cloud_cover`
- `precipitation`
- `shortwave_radiation`

说明：这是第一版无需账号的气象补充数据。官方 Copernicus ERA5 下载需要本机配置 `~/.cdsapirc` 和安装 `cdsapi`，当前环境尚未配置。

## 4. ERA5 再分析数据

原始目录：`data/raw/era5/`

处理后目录：`data/processed/era5/`

来源：Copernicus ERA5 single levels，GRIB 格式

区域：德国主体 bbox `[55.2, 5.5, 47.0, 15.5]`，实际 0.25° 网格为：

- 纬度：55.0 到 47.0
- 经度：5.5 到 15.5
- 网格：33 × 41

原始文件：

| 文件 | 变量族 | ERA5 shortName |
|---|---|---|
| `era5_de_lu_2019_2023_cloud_jan_jun.grib` | 云量，1-6月 | `tcc`, `lcc`, `mcc`, `hcc` |
| `era5_de_lu_2019_2023_cloud_jul_dec.grib` | 云量，7-12月 | `tcc`, `lcc`, `mcc`, `hcc` |
| `era5_de_lu_2019_2023_thermal_jan_jun.grib` | 温压降水，1-6月 | `2t`, `2d`, `sp`, `msl`, `tp` |
| `era5_de_lu_2019_2023_thermal_jul_dec.grib` | 温压降水，7-12月 | `2t`, `2d`, `sp`, `msl`, `tp` |
| `era5_de_lu_2019_2023_wind_jan_jun.grib` | 风，1-6月 | `10u`, `10v`, `100u`, `100v`, `10fg` |
| `era5_de_lu_2019_2023_wind_jul_dec.grib` | 风，7-12月 | `10u`, `10v`, `100u`, `100v`, `10fg` |
| `era5_de_lu_2019_2023_solar_full_year.grib` | 太阳辐射，全年 | `ssrd`, `fdir` |

处理后文件：

- `data/processed/era5/era5_de_lu_hourly_region_features_2019_2023.csv`
- `data/processed/era5/era5_de_lu_hourly_region_features_2019_2023_metadata.json`

处理后数据范围：

- 时间：2019-01-01 00:00 UTC 到 2023-12-31 23:00 UTC
- 行数：43,824
- 特征数：67
- 缺失：无缺失小时、无缺失特征单元

处理逻辑：

- 使用 GRIB 的 `validityDate` / `validityTime` 对齐到真实有效小时。
- 对每个变量计算空间加权 `mean`, `min`, `max`, `std`。
- 温度从 K 转为 °C。
- 降水 `tp` 从 m 转为 mm。
- 辐射 `ssrd` / `fdir` 从 J/m² 转为近似 W/m²，即除以 3600。
- 额外生成 `wind_speed_10m_from_mean_uv`、`wind_speed_100m_from_mean_uv` 和 `air_density_approx_mean`。

注意：

- `tp`、`ssrd`、`fdir` 是累积量，不能简单按 `dataDate` 判断月份，应使用有效时间。
- `10fg` 是 10m wind gust，可作为风电和极端天气补充特征。

## 5. 下载与处理脚本

脚本目录：`scripts/`

- `download_smard.py`
- `download_pvgis.py`
- `download_open_meteo.py`
- `scan_and_rename_era5.py`
- `build_era5_features.py`

重新下载：

```bash
python3 scripts/download_smard.py
python3 scripts/download_pvgis.py
python3 scripts/download_open_meteo.py
```

重新扫描和处理 ERA5：

```bash
python3 scripts/scan_and_rename_era5.py
python3 scripts/build_era5_features.py
```

## 6. 下一步处理

建议下一步生成 `data/processed/de_lu_hourly_2019_2023.csv`：

1. 以 SMARD `day_ahead_price` 的 `timestamp_utc` 为主键。
2. 横向合并所有 SMARD 电力变量。
3. 将 Open-Meteo 五个点转为全国均值、最大值、最小值或保留城市级特征。
4. 将 PVGIS `time` 解析并对齐到小时级时间戳，构造城市级与全国平均卫星辐照特征。
5. 添加日历特征、滞后特征和滚动统计特征。
6. 做数据泄露检查，确保训练真实预测模型时只使用预测时刻可获得的信息。
