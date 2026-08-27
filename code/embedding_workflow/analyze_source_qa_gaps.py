"""Analyze source-composite QA gaps against Census tract land/water attributes."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import Window

from qa_source_composites import STATE_FIPS


def _point_has_data(src: rasterio.DatasetReader, x: float, y: float, half: int) -> bool:
    row, col = src.index(x, y)
    if row < 0 or col < 0 or row >= src.height or col >= src.width:
        return False
    if half <= 0:
        value = next(src.sample([(x, y)], indexes=1))[0]
        return bool(np.isfinite(value))
    row_off = max(0, row - half)
    col_off = max(0, col - half)
    height = min(src.height - row_off, 2 * half + 1)
    width = min(src.width - col_off, 2 * half + 1)
    data = src.read(1, window=Window(col_off, row_off, width, height), masked=False)
    return bool(np.isfinite(data).any())


def _pct(part: float, whole: float) -> float:
    return float(part / whole * 100.0) if whole else 0.0


def analyze_state(
    state: str,
    composite_root: Path,
    tract_root: Path,
    year: int,
    tract_year: int,
    window_half: int,
) -> tuple[dict, list[dict]]:
    fips = STATE_FIPS[state]
    composite = composite_root / f"s2_annual_{state}_{year}_olmoearth.tif"
    tracts = tract_root / f"tl_{tract_year}_{fips}_tract.zip"
    rows: list[dict] = []
    with rasterio.open(composite) as src:
        gdf = gpd.read_file(tracts)
        keep = ["GEOID", "NAME", "NAMELSAD", "ALAND", "AWATER", "geometry"]
        gdf = gdf[[c for c in keep if c in gdf.columns]].to_crs(src.crs)
        points = gdf.geometry.representative_point()
        for rec, point in zip(gdf.to_dict("records"), points):
            has_data = _point_has_data(src, point.x, point.y, window_half)
            aland = int(rec.get("ALAND") or 0)
            awater = int(rec.get("AWATER") or 0)
            total_area = aland + awater
            rows.append(
                {
                    "state": state,
                    "geoid": rec.get("GEOID"),
                    "name": rec.get("NAMELSAD") or rec.get("NAME"),
                    "has_data": has_data,
                    "aland": aland,
                    "awater": awater,
                    "land_pct": _pct(aland, total_area),
                    "x": point.x,
                    "y": point.y,
                }
            )

    missing = [r for r in rows if not r["has_data"]]
    missing_land = sum(r["aland"] for r in missing)
    missing_water = sum(r["awater"] for r in missing)
    all_land = sum(r["aland"] for r in rows)
    all_water = sum(r["awater"] for r in rows)
    mostly_land = [r for r in missing if r["land_pct"] >= 50.0]
    high_land = [r for r in missing if r["land_pct"] >= 90.0]
    water_only = [r for r in missing if r["aland"] == 0 and r["awater"] > 0]
    summary = {
        "state": state,
        "tracts_total": len(rows),
        "missing_tracts": len(missing),
        "missing_tract_pct": _pct(len(missing), len(rows)),
        "missing_land_area_m2": missing_land,
        "missing_water_area_m2": missing_water,
        "missing_area_land_pct": _pct(missing_land, missing_land + missing_water),
        "state_land_area_m2": all_land,
        "state_water_area_m2": all_water,
        "state_land_area_missing_pct": _pct(missing_land, all_land),
        "missing_mostly_land_tracts": len(mostly_land),
        "missing_high_land_tracts": len(high_land),
        "missing_water_only_tracts": len(water_only),
    }
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", nargs="+", required=True, choices=sorted(STATE_FIPS))
    parser.add_argument("--composite-root", type=Path, required=True)
    parser.add_argument("--tract-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--tract-year", type=int, default=2020)
    parser.add_argument("--window-half", type=int, default=16)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for state in args.states:
        summary, rows = analyze_state(
            state,
            args.composite_root,
            args.tract_root,
            args.year,
            args.tract_year,
            args.window_half,
        )
        summaries.append(summary)
        detail_path = args.output_dir / f"{state}_missing_tracts.csv"
        with detail_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["state", "geoid", "name", "aland", "awater", "land_pct", "x", "y"],
            )
            writer.writeheader()
            for row in rows:
                if not row["has_data"]:
                    writer.writerow({k: row[k] for k in writer.fieldnames})

    (args.output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n"
    )
    print("state missing/total missing_pct missing_area_land_pct state_land_missing_pct mostly_land high_land water_only")
    for s in summaries:
        print(
            f"{s['state']} {s['missing_tracts']}/{s['tracts_total']} "
            f"{s['missing_tract_pct']:.2f}% {s['missing_area_land_pct']:.2f}% "
            f"{s['state_land_area_missing_pct']:.2f}% {s['missing_mostly_land_tracts']} "
            f"{s['missing_high_land_tracts']} {s['missing_water_only_tracts']}"
        )


if __name__ == "__main__":
    main()
