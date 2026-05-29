#!/usr/bin/env python3
"""Download PVGIS-SARAH3 hourly satellite radiation data for Germany points."""

from __future__ import annotations

import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path


BASE_URL = "https://re.jrc.ec.europa.eu/api/v5_3/seriescalc"
OUT_DIR = Path("data/raw/pvgis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

START_YEAR = 2019
END_YEAR = 2023

POINTS = {
    "berlin": (52.5200, 13.4050),
    "hamburg": (53.5511, 9.9937),
    "munich": (48.1351, 11.5820),
    "frankfurt": (50.1109, 8.6821),
    "cologne": (50.9375, 6.9603),
}


def fetch_json(params: dict[str, str | int | float], attempts: int = 4) -> dict:
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
        "lat": lat,
        "lon": lon,
        "startyear": START_YEAR,
        "endyear": END_YEAR,
        "outputformat": "json",
        "raddatabase": "PVGIS-SARAH3",
        "components": 1,
        "browser": 0,
    }
    data = fetch_json(params)
    hourly = data["outputs"]["hourly"]

    raw_path = OUT_DIR / f"{name}_pvgis_sarah3_{START_YEAR}_{END_YEAR}.json"
    raw_path.write_text(json.dumps(data), encoding="utf-8")

    csv_path = OUT_DIR / f"{name}_pvgis_sarah3_{START_YEAR}_{END_YEAR}.csv"
    fields = sorted({key for row in hourly for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(hourly)

    print(f"{name}: wrote {len(hourly)} rows -> {csv_path}")
    return len(hourly)


def main() -> None:
    summary = {}
    for name, (lat, lon) in POINTS.items():
        summary[name] = download_point(name, lat, lon)
    summary_path = OUT_DIR / "download_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
