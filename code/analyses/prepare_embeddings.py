#!/usr/bin/env python3
"""
Combine per-state aggregated embedding CSVs into a single analysis-ready file,
optionally joining ALAND/AWATER from an existing tract-level source.

Run from project root:
    uv run python code/analyses/prepare_embeddings.py \
        --input-dir data/prithvi_aggregated/tiny \
        --output data/prithvi_tiny_2022_all_tracts.csv

Works for any aggregation output directory (tiny, 300M-TL, etc.):
    uv run python code/analyses/prepare_embeddings.py \
        --input-dir data/prithvi_aggregated/300M-TL \
        --output data/prithvi_300M-TL_2022_all_tracts.csv

ALAND/AWATER are Census TIGER tract area attributes required by the analysis.
They are pulled from --area-source (default: data/alphaearth_embeddings.csv)
using lazy column selection, so the full 2.3 GB AlphaEarth file is never
loaded into memory.
"""

import argparse
from pathlib import Path

import duckdb
import polars as pl

_DEFAULT_AREA_SOURCE = Path("data/alphaearth_embeddings.csv")


def load_area_lookup(area_source: Path) -> pl.LazyFrame:
    """Lazily read only GEOID, ALAND, AWATER from a large CSV."""
    print(f"Reading ALAND/AWATER from {area_source} ...")
    return (
        pl.scan_csv(area_source, schema_overrides={"GEOID": pl.Utf8})
        .select(["GEOID", "ALAND", "AWATER"])
        .with_columns(pl.col("GEOID").str.zfill(11))
        .unique(subset=["GEOID"])
    )


def _count_csv_rows(path: Path) -> int:
    """Count data rows without materializing the CSV."""
    with path.open("rb") as f:
        line_count = sum(1 for _ in f)
    return max(0, line_count - 1)


def _combine_csvs_duckdb(nonempty_files: list[Path], output_path: Path) -> int:
    """Concatenate CSVs through DuckDB without materializing rows in Python."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    def quote(value: str) -> str:
        return "'" + value.replace("'", "''") + "'"

    file_list = "[" + ", ".join(quote(str(f)) for f in nonempty_files) + "]"
    out = quote(str(output_path))
    con = duckdb.connect(database=":memory:")
    try:
        con.execute("PRAGMA threads=1")
        con.execute("PRAGMA preserve_insertion_order=false")
        con.execute(
            f"""
            COPY (
              SELECT *
              FROM read_csv(
                {file_list},
                header=true,
                union_by_name=true,
                all_varchar=true,
                types={{'GEOID': 'VARCHAR'}}
              )
            ) TO {out} (HEADER, DELIMITER ',')
            """
        )
    finally:
        con.close()

    with output_path.open("rb") as f:
        header = f.readline()
    if not header:
        raise ValueError(f"DuckDB wrote an empty output file: {output_path}")
    return len(header.rstrip(b"\r\n").split(b","))


def combine_state_files(
    input_dir: Path,
    output_path: Path,
    area_source: Path | None = _DEFAULT_AREA_SOURCE,
) -> None:
    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    print(f"Reading {len(csv_files)} files from {input_dir} ...")

    nonempty_files = []
    skipped = []
    for f in csv_files:
        if f.stat().st_size == 0:
            skipped.append(f)
        else:
            nonempty_files.append(f)

    if skipped:
        print(f"Warning: skipped {len(skipped)} empty file(s) (0 tracts aggregated):")
        for f in skipped:
            print(f"  {f.name}")

    if not nonempty_files:
        raise ValueError(f"All {len(csv_files)} CSV files in {input_dir} were empty.")

    if area_source is None:
        n_cols = _combine_csvs_duckdb(nonempty_files, output_path)
        print(
            f"Combined : {len(nonempty_files)} states, "
            f"{_count_csv_rows(output_path):,} tracts, "
            f"{n_cols} columns"
        )
        print(f"Output   : {output_path}")
        return

    scans = [
        pl.scan_csv(f, infer_schema_length=0, schema_overrides={"GEOID": pl.Utf8})
        for f in nonempty_files
    ]
    combined = pl.concat(scans, how="diagonal_relaxed")
    columns = combined.collect_schema().names()

    # Standardize GEOID to 11-digit zero-padded string.
    if "GEOID" in columns:
        combined = combined.with_columns(pl.col("GEOID").str.zfill(11))

    # Join ALAND/AWATER if not already present and a source is available.
    columns = combined.collect_schema().names()
    if area_source is not None and not {"ALAND", "AWATER"}.issubset(set(columns)):
        if not area_source.exists():
            print(f"Warning: area source not found at {area_source}; skipping ALAND/AWATER.")
        else:
            area = load_area_lookup(area_source)
            combined = combined.join(area, on="GEOID", how="left")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.sink_csv(output_path)

    output_scan = pl.scan_csv(output_path, schema_overrides={"GEOID": pl.Utf8})
    output_columns = output_scan.collect_schema().names()
    if "ALAND" in output_columns:
        n_missing = output_scan.select(pl.col("ALAND").is_null().sum()).collect().item()
        if n_missing:
            print(f"Warning: {n_missing:,} tracts had no ALAND/AWATER match.")
        else:
            print("ALAND/AWATER joined successfully for all tracts.")

    print(
        f"Combined : {len(nonempty_files)} states, "
        f"{_count_csv_rows(output_path):,} tracts, "
        f"{len(output_columns)} columns"
    )
    print(f"Output   : {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine per-state aggregated embedding CSVs into one file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directory containing per-state CSV files from aggregate.py",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination CSV path for the combined output",
    )
    parser.add_argument(
        "--area-source",
        type=Path,
        default=_DEFAULT_AREA_SOURCE,
        help=(
            "CSV with GEOID, ALAND, AWATER columns to join in. "
            "Only GEOID/ALAND/AWATER are read, so large files are handled efficiently. "
            "Pass --no-area to skip."
        ),
    )
    parser.add_argument(
        "--no-area",
        action="store_true",
        help="Skip the ALAND/AWATER join entirely.",
    )
    args = parser.parse_args()

    area_source = None if args.no_area else args.area_source
    combine_state_files(args.input_dir, args.output, area_source=area_source)


if __name__ == "__main__":
    main()
