"""Quarantine stale source composites and derived overlap outputs for repair states."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_STATES = [
    "VA",
    "CA",
    "MT",
    "NE",
    "MI",
    "NC",
    "WV",
    "GA",
    "MS",
    "OK",
    "MO",
    "TX",
    "NV",
    "IL",
    "FL",
    "WY",
    "KY",
    "ID",
    "NY",
]


def _unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    for i in range(1, 1000):
        candidate = path.with_name(f"{path.name}.duplicate_{i}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not find unique quarantine destination for {path}")


def _path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def _record(path: Path, kind: str, scratch_root: Path, quarantine_root: Path) -> dict:
    stat = path.stat()
    relative = path.relative_to(scratch_root)
    return {
        "kind": kind,
        "src": str(path),
        "dst": str(_unique_destination(quarantine_root / relative)),
        "size_bytes": _path_size(path),
        "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


def collect_records(
    states: list[str],
    scratch_root: Path,
    source_root: Path,
    overlap_root: Path,
    quarantine_root: Path,
    year: int,
    model: str,
) -> list[dict]:
    records = []
    for state in states:
        for path in sorted(source_root.glob(f"s2_annual_{state}_{year}_olmoearth*")):
            if path.is_file():
                records.append(_record(path, "source_composite", scratch_root, quarantine_root))

        clay_state_dir = overlap_root / model / state
        if clay_state_dir.exists():
            records.append(_record(clay_state_dir, "clay_overlap_state_dir", scratch_root, quarantine_root))
    return records


def write_readme(path: Path, states: list[str], dry_run: bool) -> None:
    readme = path / "README.txt"
    readme.write_text(
        "\n".join(
            [
                "Stale source-composite and derived Clay/aggregation quarantine.",
                "",
                f"Created UTC: {datetime.now(timezone.utc).isoformat()}",
                f"Dry run: {dry_run}",
                f"States: {' '.join(states)}",
                "",
                "Files were moved here because source composite QA found low tract coverage.",
                "Repair jobs are rebuilding source composites in separate repair roots.",
                "See manifest.jsonl for original and quarantine paths.",
                "",
            ]
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch-root", type=Path, default=Path("/scratch/users/nhendrix/embeddings-health"))
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--overlap-root", type=Path)
    parser.add_argument("--quarantine-root", type=Path)
    parser.add_argument("--states", nargs="*", default=DEFAULT_STATES)
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--model", default="clay-1.5")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    scratch_root = args.scratch_root.resolve()
    source_root = (args.source_root or scratch_root / "olmoearth_composites").resolve()
    overlap_root = (args.overlap_root or scratch_root / "embedding_workflow_overlap_v1").resolve()
    quarantine_root = (
        args.quarantine_root
        or scratch_root / "quarantine" / "stale_source_aggregate_20260725"
    ).resolve()
    states = [state.upper() for state in args.states]

    records = collect_records(
        states=states,
        scratch_root=scratch_root,
        source_root=source_root,
        overlap_root=overlap_root,
        quarantine_root=quarantine_root,
        year=args.year,
        model=args.model,
    )

    quarantine_root.mkdir(parents=True, exist_ok=True)
    manifest = quarantine_root / ("manifest.jsonl" if args.execute else "manifest.dryrun.jsonl")
    with manifest.open("w") as fh:
        for record in records:
            fh.write(json.dumps(record, sort_keys=True) + "\n")

    if args.execute:
        for record in records:
            src = Path(record["src"])
            dst = Path(record["dst"])
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))

    write_readme(quarantine_root, states, dry_run=not args.execute)
    total_size = sum(int(record["size_bytes"]) for record in records)
    by_kind: dict[str, int] = {}
    for record in records:
        by_kind[record["kind"]] = by_kind.get(record["kind"], 0) + 1

    print(
        json.dumps(
            {
                "execute": args.execute,
                "states": states,
                "records": len(records),
                "by_kind": by_kind,
                "total_size_bytes": total_size,
                "manifest": str(manifest),
                "quarantine_root": str(quarantine_root),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
