#!/usr/bin/env python3
"""Build the hourly DE-LU model table from SMARD, ERA5, and PVGIS data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


START = pd.Timestamp("2019-01-01T00:00:00Z")
END = pd.Timestamp("2023-12-31T23:00:00Z")

SMARD_DIR = Path("data/raw/smard")
PVGIS_DIR = Path("data/raw/pvgis")
ERA5_PATH = Path("data/processed/era5/era5_de_lu_hourly_region_features_2019_2023.csv")
OUT_PATH = Path("data/processed/de_lu_hourly_2019_2023.csv")
METADATA_PATH = Path("data/processed/de_lu_hourly_2019_2023_metadata.json")


def read_smard() -> pd.DataFrame:
    files = sorted(SMARD_DIR.glob("*_DE-LU_2019-01-01_2023-12-31_hour.csv"))
    if not files:
        raise FileNotFoundError(f"no SMARD files found under {SMARD_DIR}")

    merged: pd.DataFrame | None = None
    for path in files:
        df = pd.read_csv(path)
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        value_cols = [col for col in df.columns if col not in {"timestamp_utc", "timestamp_ms"}]
        if len(value_cols) != 1:
            raise ValueError(f"{path} expected exactly one value column, got {value_cols}")
        df = df[["timestamp_utc", value_cols[0]]]
        merged = df if merged is None else merged.merge(df, on="timestamp_utc", how="outer")

    assert merged is not None
    return merged.sort_values("timestamp_utc")


def parse_pvgis_time(series: pd.Series) -> pd.Series:
    # PVGIS returns strings such as 20190101:0011. For this hourly project,
    # keep the YYYYMMDD:HH part and align it to the target UTC hour key.
    hour_text = series.astype(str).str.slice(0, 11)
    return pd.to_datetime(hour_text, format="%Y%m%d:%H", utc=True)


def read_pvgis() -> pd.DataFrame:
    city_frames = []
    for path in sorted(PVGIS_DIR.glob("*_pvgis_sarah3_2019_2023.csv")):
        city = path.name.split("_pvgis_sarah3_")[0]
        df = pd.read_csv(path)
        df["timestamp_utc"] = parse_pvgis_time(df["time"])
        rename = {
            "Gb(i)": f"pvgis_{city}_beam_irradiance",
            "Gd(i)": f"pvgis_{city}_diffuse_irradiance",
            "Gr(i)": f"pvgis_{city}_reflected_irradiance",
            "H_sun": f"pvgis_{city}_sun_height",
            "T2m": f"pvgis_{city}_temperature_2m",
            "WS10m": f"pvgis_{city}_wind_speed_10m",
            "Int": f"pvgis_{city}_interpolated_flag",
        }
        keep = ["timestamp_utc", *rename.values()]
        df = df.rename(columns=rename)[keep]
        city_frames.append(df)

    if not city_frames:
        raise FileNotFoundError(f"no PVGIS files found under {PVGIS_DIR}")

    merged = city_frames[0]
    for df in city_frames[1:]:
        merged = merged.merge(df, on="timestamp_utc", how="outer")

    cities = sorted({col.split("_")[1] for col in merged.columns if col.startswith("pvgis_")})
    for variable in [
        "beam_irradiance",
        "diffuse_irradiance",
        "reflected_irradiance",
        "sun_height",
        "temperature_2m",
        "wind_speed_10m",
        "interpolated_flag",
    ]:
        cols = [f"pvgis_{city}_{variable}" for city in cities if f"pvgis_{city}_{variable}" in merged.columns]
        merged[f"pvgis_{variable}_mean"] = merged[cols].mean(axis=1)
        merged[f"pvgis_{variable}_min"] = merged[cols].min(axis=1)
        merged[f"pvgis_{variable}_max"] = merged[cols].max(axis=1)
        merged[f"pvgis_{variable}_std"] = merged[cols].std(axis=1)

    return merged.sort_values("timestamp_utc")


def read_era5() -> pd.DataFrame:
    df = pd.read_csv(ERA5_PATH)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df.sort_values("timestamp_utc")


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    local_time = df["timestamp_utc"].dt.tz_convert("Europe/Berlin")
    df["hour"] = local_time.dt.hour
    df["weekday"] = local_time.dt.weekday
    df["month"] = local_time.dt.month
    df["dayofyear"] = local_time.dt.dayofyear
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["is_summer_time"] = local_time.apply(lambda x: int(x.dst().total_seconds() != 0))
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_dayofyear"] = np.sin(2 * np.pi * df["dayofyear"] / 366)
    df["cos_dayofyear"] = np.cos(2 * np.pi * df["dayofyear"] / 366)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    target = "day_ahead_price"
    for lag in [24, 48, 72, 168, 336]:
        df[f"price_lag_{lag}h"] = df[target].shift(lag)
    for window in [24, 168, 336]:
        shifted = df[target].shift(24)
        df[f"price_rolling_{window}h_mean"] = shifted.rolling(window, min_periods=max(6, window // 4)).mean()
        df[f"price_rolling_{window}h_std"] = shifted.rolling(window, min_periods=max(6, window // 4)).std()
        df[f"price_rolling_{window}h_min"] = shifted.rolling(window, min_periods=max(6, window // 4)).min()
        df[f"price_rolling_{window}h_max"] = shifted.rolling(window, min_periods=max(6, window // 4)).max()
    return df


def add_power_features(df: pd.DataFrame) -> pd.DataFrame:
    df["wind_total_actual"] = df["wind_onshore_actual"] + df["wind_offshore_actual"]
    df["wind_total_forecast_day_ahead"] = (
        df["wind_onshore_forecast_day_ahead"] + df["wind_offshore_forecast_day_ahead"]
    )
    df["renewable_actual"] = df["wind_total_actual"] + df["solar_actual"]
    df["renewable_forecast_day_ahead"] = df["wind_total_forecast_day_ahead"] + df["solar_forecast_day_ahead"]
    df["residual_load_forecast"] = df["load_forecast"] - df["renewable_forecast_day_ahead"]
    df["residual_load_actual"] = df["load_actual"] - df["renewable_actual"]
    df["renewable_share_forecast"] = df["renewable_forecast_day_ahead"] / df["load_forecast"].replace(0, np.nan)
    df["solar_share_forecast"] = df["solar_forecast_day_ahead"] / df["load_forecast"].replace(0, np.nan)
    df["wind_share_forecast"] = df["wind_total_forecast_day_ahead"] / df["load_forecast"].replace(0, np.nan)
    return df


def build_table() -> pd.DataFrame:
    hourly = pd.DataFrame({"timestamp_utc": pd.date_range(START, END, freq="h")})
    df = hourly.merge(read_smard(), on="timestamp_utc", how="left")
    df = df.merge(read_era5(), on="timestamp_utc", how="left")
    df = df.merge(read_pvgis(), on="timestamp_utc", how="left")
    df = add_calendar_features(df)
    df = add_power_features(df)
    df = add_lag_features(df)
    return df


def write_metadata(df: pd.DataFrame) -> None:
    missing = df.isna().sum().sort_values(ascending=False)
    metadata = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "start": df["timestamp_utc"].min().isoformat(),
        "end": df["timestamp_utc"].max().isoformat(),
        "duplicate_timestamps": int(df["timestamp_utc"].duplicated().sum()),
        "missing_cells_by_column": {k: int(v) for k, v in missing.items() if int(v) > 0},
        "column_groups": {
            "target": ["day_ahead_price"],
            "smard": [c for c in df.columns if c in {
                "load_actual", "load_forecast", "solar_actual", "wind_onshore_actual",
                "wind_offshore_actual", "solar_forecast_day_ahead",
                "wind_onshore_forecast_day_ahead", "wind_offshore_forecast_day_ahead",
                "wind_solar_forecast_day_ahead"
            }],
            "era5": [c for c in df.columns if c not in {"timestamp_utc"} and not c.startswith("pvgis_") and c not in {
                "day_ahead_price", "load_actual", "load_forecast", "solar_actual",
                "wind_onshore_actual", "wind_offshore_actual", "solar_forecast_day_ahead",
                "wind_onshore_forecast_day_ahead", "wind_offshore_forecast_day_ahead",
                "wind_solar_forecast_day_ahead", "hour", "weekday", "month", "dayofyear",
                "is_weekend", "is_summer_time", "sin_hour", "cos_hour", "sin_dayofyear",
                "cos_dayofyear"
            } and not c.startswith("price_")],
            "pvgis_sarah3": [c for c in df.columns if c.startswith("pvgis_")],
            "calendar": [
                "hour", "weekday", "month", "dayofyear", "is_weekend", "is_summer_time",
                "sin_hour", "cos_hour", "sin_dayofyear", "cos_dayofyear"
            ],
            "price_lags_rollings": [c for c in df.columns if c.startswith("price_")]
        }
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = build_table()
    df.to_csv(OUT_PATH, index=False)
    write_metadata(df)
    print(f"wrote {len(df)} rows x {df.shape[1]} cols -> {OUT_PATH}")
    print(f"wrote metadata -> {METADATA_PATH}")


if __name__ == "__main__":
    main()
