#!/usr/bin/env python3
"""Download hourly German electricity market series from SMARD.

Source: Bundesnetzagentur SMARD public chart_data endpoint.
Region: DE-LU
Resolution: hour
Range: 2019-01-01 to 2023-12-31, inclusive in UTC filtering.
"""

from __future__ import annotations

import csv
import datetime as dt
import http.client
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


BASE_URL = "https://www.smard.de/app/chart_data"
REGION = "DE-LU"
RESOLUTION = "hour"
START = dt.datetime(2019, 1, 1, tzinfo=dt.timezone.utc)
END_EXCLUSIVE = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)

OUT_DIR = Path("data/raw/smard")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODULES = {
    "day_ahead_price": 4169,
    "load_actual": 410,
    "load_forecast": 411,
    "solar_actual": 4068,
    "wind_onshore_actual": 4067,
    "wind_offshore_actual": 1225,
    "solar_forecast_day_ahead": 125,
    "wind_onshore_forecast_day_ahead": 123,
    "wind_offshore_forecast_day_ahead": 3791,
    "wind_solar_forecast_day_ahead": 5097,
}


def fetch_json(url: str, attempts: int = 4) -> dict:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 electricity-forecast-study"},
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except (
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
            http.client.RemoteDisconnected,
        ) as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}: {last_error}")


def ms_to_utc(ms: int) -> dt.datetime:
    return dt.datetime.fromtimestamp(ms / 1000, tz=dt.timezone.utc)


def download_module(name: str, module_id: int) -> int:
    index_url = f"{BASE_URL}/{module_id}/{REGION}/index_{RESOLUTION}.json"
    index = fetch_json(index_url)
    timestamps = index.get("timestamps", [])

    selected = []
    for chunk_ms in timestamps:
        chunk_time = ms_to_utc(chunk_ms)
        # SMARD hour chunks are weekly. Keep a small buffer around the target range.
        if START - dt.timedelta(days=8) <= chunk_time < END_EXCLUSIVE:
            selected.append(chunk_ms)

    def fetch_chunk(chunk_ms: int) -> list[list[int | float | None]]:
        data_url = (
            f"{BASE_URL}/{module_id}/{REGION}/"
            f"{module_id}_{REGION}_{RESOLUTION}_{chunk_ms}.json"
        )
        data = fetch_json(data_url)
        return data.get("series", [])

    values: dict[int, float | None] = {}
    completed = 0
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(fetch_chunk, chunk_ms): chunk_ms for chunk_ms in selected}
        for future in as_completed(futures):
            completed += 1
            series = future.result()
            if completed % 50 == 0 or completed == len(selected):
                print(f"{name}: downloaded {completed}/{len(selected)} chunks", flush=True)
            for timestamp_ms, value in series:
                timestamp = ms_to_utc(timestamp_ms)
                if START <= timestamp < END_EXCLUSIVE:
                    values[timestamp_ms] = value

    out_path = OUT_DIR / f"{name}_{REGION}_{START.date()}_{(END_EXCLUSIVE - dt.timedelta(days=1)).date()}_{RESOLUTION}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_utc", "timestamp_ms", name])
        for timestamp_ms in sorted(values):
            timestamp = ms_to_utc(timestamp_ms).isoformat().replace("+00:00", "Z")
            writer.writerow([timestamp, timestamp_ms, values[timestamp_ms]])

    print(f"{name}: wrote {len(values)} rows -> {out_path}", flush=True)
    return len(values)


def main() -> None:
    summary = {}
    for name, module_id in MODULES.items():
        summary[name] = download_module(name, module_id)

    summary_path = OUT_DIR / "download_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
