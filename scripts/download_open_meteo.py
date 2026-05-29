#!/usr/bin/env python3
"""Download account-free hourly historical weather features from Open-Meteo.

This is a practical first-pass weather source. For the final ERA5 version,
replace it with Copernicus CDS once ~/.cdsapirc is configured.
"""

from __future__ import annotations

import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path


BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
OUT_DIR = Path("data/raw/open_meteo")
OUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2019-01-01"
END_DATE = "2023-12-31"
VARIABLES = [
    "temperature_2m",
    "wind_speed_10m",
    "wind_speed_100m",
    "cloud_cover",
    "precipitation",
    "shortwave_radiation",
]

POINTS = {
    "berlin": (52.5200, 13.4050),
    "hamburg": (53.5511, 9.9937),
    "munich": (48.1351, 11.5820),
    "frankfurt": (50.1109, 8.6821),
    "cologne": (50.9375, 6.9603),
}


def fetch_json(params: dict[str, str | float], attempts: int = 4) -> dict:
    url = f"{BASE_URL}?{urllib.parse.urlencode(params)}"
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - script should retry transient API failures.
            last_error = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}: {last_error}")


def download_point(name: str, lat: float, lon: float) -> int:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(VARIABLES),
        "timezone": "UTC",
    }
    data = fetch_json(params)
    raw_path = OUT_DIR / f"{name}_open_meteo_{START_DATE}_{END_DATE}.json"
    raw_path.write_text(json.dumps(data), encoding="utf-8")

    hourly = data["hourly"]
    rows = []
    for i, timestamp in enumerate(hourly["time"]):
        row = {"timestamp_utc": f"{timestamp}:00Z"}
        for var in VARIABLES:
            row[var] = hourly[var][i]
        rows.append(row)

    csv_path = OUT_DIR / f"{name}_open_meteo_{START_DATE}_{END_DATE}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp_utc", *VARIABLES])
        writer.writeheader()
        writer.writerows(rows)

    print(f"{name}: wrote {len(rows)} rows -> {csv_path}")
    return len(rows)


def main() -> None:
    summary = {}
    for name, (lat, lon) in POINTS.items():
        summary[name] = download_point(name, lat, lon)
    summary_path = OUT_DIR / "download_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
