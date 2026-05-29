#!/usr/bin/env python3
"""Build hourly region-level ERA5 features from GRIB files.

The output is intentionally compact: one row per UTC hour, with spatially
weighted mean/min/max/std features for each ERA5 variable.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
from collections import defaultdict
from pathlib import Path

import eccodes
import numpy as np


ERA5_DIR = Path("data/raw/era5")
OUT_DIR = Path("data/processed/era5")
OUT_DIR.mkdir(parents=True, exist_ok=True)

START = dt.datetime(2019, 1, 1, tzinfo=dt.timezone.utc)
END_EXCLUSIVE = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)

FILES = sorted(ERA5_DIR.glob("era5_de_lu_2019_2023_*.grib"))

SHORT_NAME_TO_FEATURE = {
    "2t": "temperature_2m",
    "2d": "dewpoint_2m",
    "sp": "surface_pressure",
    "msl": "mean_sea_level_pressure",
    "tp": "total_precipitation",
    "10u": "u_wind_10m",
    "10v": "v_wind_10m",
    "100u": "u_wind_100m",
    "100v": "v_wind_100m",
    "10fg": "wind_gust_10m",
    "tcc": "total_cloud_cover",
    "lcc": "low_cloud_cover",
    "mcc": "medium_cloud_cover",
    "hcc": "high_cloud_cover",
    "ssrd": "surface_solar_radiation_downwards",
    "fdir": "total_sky_direct_solar_radiation_at_surface",
}

KELVIN_VARS = {"2t", "2d"}
PRECIP_M_VARS = {"tp"}
RADIATION_J_M2_VARS = {"ssrd", "fdir"}


def get_key(gid: int, key: str):
    try:
        return eccodes.codes_get(gid, key)
    except Exception:
        return None


def valid_datetime(gid: int) -> dt.datetime | None:
    date_value = get_key(gid, "validityDate")
    time_value = get_key(gid, "validityTime")
    if date_value is None or time_value is None:
        date_value = get_key(gid, "dataDate")
        time_value = get_key(gid, "dataTime")
    if date_value is None or time_value is None:
        return None

    date_text = f"{int(date_value):08d}"
    time_text = f"{int(time_value):04d}"
    return dt.datetime(
        int(date_text[:4]),
        int(date_text[4:6]),
        int(date_text[6:8]),
        int(time_text[:2]),
        int(time_text[2:]),
        tzinfo=dt.timezone.utc,
    )


def latitude_weights(gid: int) -> np.ndarray:
    lat_first = float(eccodes.codes_get(gid, "latitudeOfFirstGridPointInDegrees"))
    lat_last = float(eccodes.codes_get(gid, "latitudeOfLastGridPointInDegrees"))
    ni = int(eccodes.codes_get(gid, "Ni"))
    nj = int(eccodes.codes_get(gid, "Nj"))
    lats = np.linspace(lat_first, lat_last, nj)
    weights_by_lat = np.cos(np.deg2rad(lats))
    return np.repeat(weights_by_lat, ni)


def convert_values(short_name: str, values: np.ndarray) -> np.ndarray:
    converted = values.astype("float64", copy=False)
    if short_name in KELVIN_VARS:
        converted = converted - 273.15
    elif short_name in PRECIP_M_VARS:
        converted = converted * 1000.0
    elif short_name in RADIATION_J_M2_VARS:
        converted = converted / 3600.0
    return converted


def stats(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(values)
    vals = values[mask]
    w = weights[mask]
    mean = float(np.average(vals, weights=w))
    variance = float(np.average((vals - mean) ** 2, weights=w))
    return {
        "mean": mean,
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "std": math.sqrt(variance),
    }


def scan_file(path: Path, rows: dict[str, dict[str, float]], metadata: list[dict]) -> None:
    message_count = 0
    used_count = 0
    skipped_count = 0
    variables = set()
    weights_cache: np.ndarray | None = None

    with path.open("rb") as f:
        while True:
            gid = eccodes.codes_grib_new_from_file(f)
            if gid is None:
                break
            message_count += 1
            try:
                short_name = str(eccodes.codes_get(gid, "shortName"))
                feature = SHORT_NAME_TO_FEATURE.get(short_name)
                when = valid_datetime(gid)
                if feature is None or when is None or not (START <= when < END_EXCLUSIVE):
                    skipped_count += 1
                    continue

                if weights_cache is None:
                    weights_cache = latitude_weights(gid)

                values = eccodes.codes_get_values(gid)
                converted = convert_values(short_name, values)
                stat_values = stats(converted, weights_cache)
                timestamp = when.isoformat().replace("+00:00", "Z")
                row = rows[timestamp]
                for stat_name, value in stat_values.items():
                    row[f"{feature}_{stat_name}"] = value
                variables.add(short_name)
                used_count += 1
            finally:
                eccodes.codes_release(gid)

    metadata.append(
        {
            "file": str(path),
            "message_count": message_count,
            "used_count": used_count,
            "skipped_count": skipped_count,
            "short_names": sorted(variables),
        }
    )
    print(f"{path.name}: used {used_count}/{message_count} messages", flush=True)


def add_derived_features(row: dict[str, float]) -> None:
    if "u_wind_10m_mean" in row and "v_wind_10m_mean" in row:
        row["wind_speed_10m_from_mean_uv"] = math.hypot(row["u_wind_10m_mean"], row["v_wind_10m_mean"])
    if "u_wind_100m_mean" in row and "v_wind_100m_mean" in row:
        row["wind_speed_100m_from_mean_uv"] = math.hypot(row["u_wind_100m_mean"], row["v_wind_100m_mean"])
    if "temperature_2m_mean" in row and "surface_pressure_mean" in row:
        temperature_k = row["temperature_2m_mean"] + 273.15
        row["air_density_approx_mean"] = row["surface_pressure_mean"] / (287.05 * temperature_k)


def main() -> None:
    if not FILES:
        raise FileNotFoundError(f"no ERA5 GRIB files found under {ERA5_DIR}")

    rows: dict[str, dict[str, float]] = defaultdict(dict)
    metadata: list[dict] = []
    for path in FILES:
        scan_file(path, rows, metadata)

    expected = []
    current = START
    while current < END_EXCLUSIVE:
        expected.append(current.isoformat().replace("+00:00", "Z"))
        current += dt.timedelta(hours=1)

    for timestamp in expected:
        add_derived_features(rows[timestamp])

    feature_names = sorted({key for row in rows.values() for key in row})
    out_path = OUT_DIR / "era5_de_lu_hourly_region_features_2019_2023.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp_utc", *feature_names])
        writer.writeheader()
        for timestamp in expected:
            writer.writerow({"timestamp_utc": timestamp, **rows[timestamp]})

    completeness = {
        "rows": len(expected),
        "features": len(feature_names),
        "missing_cells": {
            feature: sum(1 for timestamp in expected if feature not in rows[timestamp])
            for feature in feature_names
        },
        "files": metadata,
    }
    metadata_path = OUT_DIR / "era5_de_lu_hourly_region_features_2019_2023_metadata.json"
    metadata_path.write_text(json.dumps(completeness, indent=2), encoding="utf-8")

    print(f"wrote {len(expected)} rows and {len(feature_names)} features -> {out_path}")
    print(f"wrote metadata -> {metadata_path}")


if __name__ == "__main__":
    main()
