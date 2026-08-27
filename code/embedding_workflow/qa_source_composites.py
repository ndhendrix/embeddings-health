"""QA OlmoEarth source composites against Census tract locations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import Window


STATE_FIPS = {
    "AL": "01",
    "AZ": "04",
    "AR": "05",
    "CA": "06",
    "CO": "08",
    "CT": "09",
    "DE": "10",
    "DC": "11",
    "FL": "12",
    "GA": "13",
    "ID": "16",
    "IL": "17",
    "IN": "18",
    "IA": "19",
    "KS": "20",
    "KY": "21",
    "LA": "22",
    "ME": "23",
    "MD": "24",
    "MA": "25",
    "MI": "26",
    "MN": "27",
    "MS": "28",
    "MO": "29",
    "MT": "30",
    "NE": "31",
    "NV": "32",
    "NH": "33",
    "NJ": "34",
    "NM": "35",
    "NY": "36",
    "NC": "37",
    "ND": "38",
    "OH": "39",
    "OK": "40",
    "OR": "41",
    "PA": "42",
    "RI": "44",
    "SC": "45",
    "SD": "46",
    "TN": "47",
    "TX": "48",
    "UT": "49",
    "VT": "50",
    "VA": "51",
    "WA": "53",
    "WV": "54",
    "WI": "55",
    "WY": "56",
}


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


def qa_state(
    state: str,
    composite_root: Path,
    tract_root: Path,
    output_dir: Path,
    year: int,
    tract_year: int,
    window_half: int,
    fail_below_pct: float,
) -> dict:
    state = state.upper()
    fips = STATE_FIPS[state]
    composite = composite_root / f"s2_annual_{state}_{year}_olmoearth.tif"
    tracts = tract_root / f"tl_{tract_year}_{fips}_tract.zip"
    report = {
        "state": state,
        "fips": fips,
        "year": year,
        "tract_year": tract_year,
        "composite": str(composite),
        "tracts": str(tracts),
        "window_half_pixels": window_half,
    }
    if not composite.is_file() or composite.stat().st_size == 0:
        report.update({"status": "missing_composite", "passed": False})
    elif not tracts.is_file() or tracts.stat().st_size == 0:
        report.update({"status": "missing_tracts", "passed": False})
    else:
        with rasterio.open(composite) as src:
            geoms = gpd.read_file(tracts)[["GEOID", "geometry"]].to_crs(src.crs)
            points = geoms.geometry.representative_point()
            valid = 0
            outside = 0
            for point in points:
                try:
                    if _point_has_data(src, point.x, point.y, window_half):
                        valid += 1
                except Exception:
                    outside += 1
            total = int(len(points))
            pct = (valid / total * 100.0) if total else 0.0
            report.update(
                {
                    "status": "pass" if pct >= fail_below_pct else "fail_low_tract_coverage",
                    "passed": pct >= fail_below_pct,
                    "tracts_total": total,
                    "tract_representative_windows_with_finite_band1": valid,
                    "tract_representative_windows_without_finite_band1": total - valid,
                    "outside_or_error_points": outside,
                    "coverage_pct": pct,
                    "shape": [src.count, src.height, src.width],
                    "crs": str(src.crs),
                    "bounds": list(src.bounds),
                }
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{state}.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return report


def summarize(output_dir: Path, fail_below_pct: float) -> None:
    rows = []
    for path in sorted(output_dir.glob("*.json")):
        data = json.loads(path.read_text())
        if "coverage_pct" in data:
            rows.append(
                (
                    float(data["coverage_pct"]),
                    data["state"],
                    int(data["tract_representative_windows_with_finite_band1"]),
                    int(data["tracts_total"]),
                    data["status"],
                )
            )
        else:
            rows.append((-1.0, data["state"], 0, 0, data["status"]))
    print("state valid total pct status")
    for pct, state, valid, total, status in sorted(rows):
        pct_text = "NA" if pct < 0 else f"{pct:7.2f}%"
        print(f"{state:2s} {valid:6d} {total:6d} {pct_text:>8s} {status}")
    flagged = [row for row in rows if row[0] < fail_below_pct]
    print(f"\nFlagged below {fail_below_pct:.1f}%:")
    for pct, state, valid, total, status in sorted(flagged):
        pct_text = "NA" if pct < 0 else f"{pct:.2f}%"
        print(f"{state}: {valid}/{total} ({pct_text}) {status}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", choices=sorted(STATE_FIPS))
    parser.add_argument("--states", nargs="*", choices=sorted(STATE_FIPS))
    parser.add_argument("--composite-root", type=Path, required=True)
    parser.add_argument("--tract-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--tract-year", type=int, default=2020)
    parser.add_argument("--window-half", type=int, default=16)
    parser.add_argument("--fail-below-pct", type=float, default=90.0)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    if args.summary:
        summarize(args.output_dir, args.fail_below_pct)
        return

    states = args.states or ([args.state] if args.state else sorted(STATE_FIPS))
    for state in states:
        qa_state(
            state,
            args.composite_root,
            args.tract_root,
            args.output_dir,
            args.year,
            args.tract_year,
            args.window_half,
            args.fail_below_pct,
        )


if __name__ == "__main__":
    main()
