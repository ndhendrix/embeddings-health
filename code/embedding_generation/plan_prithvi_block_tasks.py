"""Plan native-block aggregation tasks for incomplete Prithvi states."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import rasterio


STATE_FIPS = {
    "AL": "01", "AZ": "04", "AR": "05", "CA": "06", "CO": "08",
    "CT": "09", "DE": "10", "DC": "11", "FL": "12", "GA": "13",
    "ID": "16", "IL": "17", "IN": "18", "IA": "19", "KS": "20",
    "KY": "21", "LA": "22", "ME": "23", "MD": "24", "MA": "25",
    "MI": "26", "MN": "27", "MS": "28", "MO": "29", "MT": "30",
    "NE": "31", "NV": "32", "NH": "33", "NJ": "34", "NM": "35",
    "NY": "36", "NC": "37", "ND": "38", "OH": "39", "OK": "40",
    "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46",
    "TN": "47", "TX": "48", "UT": "49", "VT": "50", "VA": "51",
    "WA": "53", "WV": "54", "WI": "55", "WY": "56",
}


def resolve_embedding(state_dir: Path, state: str, year: int) -> Path:
    raw = state_dir / f"prithvi_300M-TL_{state}_{year}_raw.tif"
    plain = state_dir / f"prithvi_300M-TL_{state}_{year}.tif"
    if raw.is_file() and raw.stat().st_size:
        return raw
    if plain.is_file() and plain.stat().st_size:
        return plain
    raise FileNotFoundError(f"no complete 300M-TL embedding for {state}")


def build_plan(
    embedding_root: Path,
    aggregate_root: Path,
    tract_root: Path,
    partial_root: Path,
    manifest: Path,
    states_output: Path,
    year: int,
    tract_year: int,
    expected_dim: int,
) -> tuple[int, int]:
    rows: list[tuple[str, Path, Path, Path, int, int]] = []
    planned_states: list[str] = []

    for state_dir in sorted(path for path in embedding_root.iterdir() if path.is_dir()):
        state = state_dir.name
        if state not in STATE_FIPS:
            continue
        final = aggregate_root / f"prithvi_300M-TL_{state}_{year}_tracts.csv"
        if final.is_file() and final.stat().st_size:
            continue

        embedding = resolve_embedding(state_dir, state, year)
        tracts = tract_root / f"tl_{tract_year}_{STATE_FIPS[state]}_tract.zip"
        if not tracts.is_file():
            raise FileNotFoundError(f"missing tract file for {state}: {tracts}")

        with rasterio.open(embedding) as src:
            if src.count != expected_dim:
                raise ValueError(
                    f"expected {expected_dim} bands, found {src.count}: {embedding}"
                )
            block_height, block_width = src.block_shapes[0]
            if any(shape != (block_height, block_width) for shape in src.block_shapes):
                raise ValueError(f"band-dependent block shapes: {embedding}")
            block_rows = math.ceil(src.height / block_height)
            block_cols = math.ceil(src.width / block_width)

        planned_states.append(state)
        for block_row in range(block_rows):
            for block_col in range(block_cols):
                output = (
                    partial_root
                    / state
                    / f"block_r{block_row:04d}_c{block_col:04d}.npz"
                )
                rows.append(
                    (state, embedding, tracts, output, block_row, block_col)
                )

    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerows(rows)

    states_output.parent.mkdir(parents=True, exist_ok=True)
    states_output.write_text("".join(f"{state}\n" for state in planned_states))
    return len(planned_states), len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--aggregate-root", type=Path, required=True)
    parser.add_argument("--tract-root", type=Path, required=True)
    parser.add_argument("--partial-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--states-output", type=Path, required=True)
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--tract-year", type=int, default=2020)
    parser.add_argument("--expected-dim", type=int, default=1024)
    args = parser.parse_args()

    states, tasks = build_plan(
        args.embedding_root,
        args.aggregate_root,
        args.tract_root,
        args.partial_root,
        args.manifest,
        args.states_output,
        args.year,
        args.tract_year,
        args.expected_dim,
    )
    print(f"Planned {tasks} native-block tasks for {states} incomplete states")
    print(f"Manifest: {args.manifest}")
    print(f"States:   {args.states_output}")


if __name__ == "__main__":
    main()
