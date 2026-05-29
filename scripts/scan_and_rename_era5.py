#!/usr/bin/env python3
"""Scan ERA5 GRIB headers and rename files to stable project names.

This script uses ecCodes directly so it reads message metadata only, not the
full gridded arrays.
"""

from __future__ import annotations

import json
from pathlib import Path

import eccodes


ERA5_DIR = Path("data/raw/era5")
START_YEAR = 2019
END_YEAR = 2023

FAMILY_BY_VARS = {
    frozenset({"10u", "10v", "100u", "100v"}): "wind",
    frozenset({"tcc", "lcc", "mcc", "hcc"}): "cloud",
    frozenset({"2t", "2d", "sp", "msl", "tp"}): "thermal",
    frozenset({"ssrd", "fdir"}): "solar",
}


def get_key(gid: int, key: str):
    try:
        return eccodes.codes_get(gid, key)
    except Exception:
        return None


def scan_grib(path: Path) -> dict:
    short_names: set[str] = set()
    data_dates: set[int] = set()
    validity_dates: set[int] = set()
    times: set[int] = set()
    steps: set[int] = set()
    grid: dict[str, float | int | None] = {}
    count = 0

    with path.open("rb") as f:
        while True:
            gid = eccodes.codes_grib_new_from_file(f)
            if gid is None:
                break
            count += 1
            try:
                short_name = get_key(gid, "shortName")
                if short_name:
                    short_names.add(str(short_name))

                data_date = get_key(gid, "dataDate")
                if data_date:
                    data_dates.add(int(data_date))

                validity_date = get_key(gid, "validityDate")
                if validity_date:
                    validity_dates.add(int(validity_date))

                data_time = get_key(gid, "dataTime")
                if data_time is not None:
                    times.add(int(data_time))

                step = get_key(gid, "stepRange")
                if step is not None:
                    try:
                        steps.add(int(str(step).split("-")[-1]))
                    except ValueError:
                        pass

                if not grid:
                    for key in [
                        "latitudeOfFirstGridPointInDegrees",
                        "latitudeOfLastGridPointInDegrees",
                        "longitudeOfFirstGridPointInDegrees",
                        "longitudeOfLastGridPointInDegrees",
                        "Ni",
                        "Nj",
                    ]:
                        grid[key] = get_key(gid, key)
            finally:
                eccodes.codes_release(gid)

    coverage_dates = validity_dates or data_dates
    years = sorted({date // 10000 for date in coverage_dates})
    months = sorted({(date // 100) % 100 for date in coverage_dates})

    return {
        "path": str(path),
        "message_count": count,
        "short_names": sorted(short_names),
        "data_years": sorted({date // 10000 for date in data_dates}),
        "data_months": sorted({(date // 100) % 100 for date in data_dates}),
        "years": years,
        "months": months,
        "data_times": sorted(times),
        "steps": sorted(steps),
        "grid": grid,
    }


def infer_family(names: set[str]) -> str:
    for expected, family in FAMILY_BY_VARS.items():
        if expected.issubset(names):
            return family
    raise ValueError(f"unknown variable family for shortNames={sorted(names)}")


def infer_half(months: list[int]) -> str:
    if months == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        return "full_year"
    if months == [1, 2, 3, 4, 5, 6]:
        return "jan_jun"
    if months == [7, 8, 9, 10, 11, 12]:
        return "jul_dec"
    raise ValueError(f"unexpected month coverage: {months}")


def main() -> None:
    records = []
    for path in sorted(ERA5_DIR.glob("*.grib")):
        meta = scan_grib(path)
        names = set(meta["short_names"])
        family = infer_family(names)
        half = infer_half(meta["months"])
        if meta["years"] != list(range(START_YEAR, END_YEAR + 1)):
            raise ValueError(f"{path} unexpected years: {meta['years']}")

        target = ERA5_DIR / f"era5_de_lu_{START_YEAR}_{END_YEAR}_{family}_{half}.grib"
        meta.update(
            {
                "source": str(path),
                "target": str(target),
                "family": family,
                "half": half,
            }
        )
        records.append(meta)

    targets = [record["target"] for record in records]
    if len(targets) != len(set(targets)):
        raise ValueError(f"duplicate rename targets: {targets}")

    print(json.dumps(records, indent=2), flush=True)

    for record in records:
        source = Path(record["source"])
        target = Path(record["target"])
        if source == target:
            continue
        if target.exists():
            raise FileExistsError(f"target already exists: {target}")
        source.rename(target)

    manifest_path = ERA5_DIR / "era5_file_manifest.json"
    manifest_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"wrote manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
