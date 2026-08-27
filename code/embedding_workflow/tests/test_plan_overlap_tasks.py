"""Completion checks for model-aware overlap planning."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plan_overlap_tasks import validated_output


def test_corrected_olmoearth_output_is_reusable(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "source.tif"
    tile = tmp_path / "tile.tif"
    source.touch()
    tile.touch()
    tile.with_suffix(".validation.json").write_text(
        json.dumps(
            {
                "model": "olmoearth-v1.2-nano",
                "tags": {
                    "model": "olmoearth-v1.2-nano",
                    "model_revision": "revision",
                    "source_composite": str(source),
                    "source_size": str(source.stat().st_size),
                    "source_mtime_ns": str(source.stat().st_mtime_ns),
                    "workflow": "overlap-center50-v2",
                    "input_normalization": "computed-2std",
                },
            }
        )
    )

    assert validated_output(
        tile, source, "olmoearth-v1.2-nano", "revision", "olmoearth"
    )


def test_old_or_wrong_source_olmoearth_output_is_replanned(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "source.tif"
    other_source = tmp_path / "other.tif"
    tile = tmp_path / "tile.tif"
    source.touch()
    other_source.touch()
    tile.touch()
    validation = tile.with_suffix(".validation.json")
    report = {
        "model": "olmoearth-v1.2-nano",
        "tags": {
            "model": "olmoearth-v1.2-nano",
            "model_revision": "revision",
            "source_composite": str(source),
            "source_size": str(source.stat().st_size),
            "source_mtime_ns": str(source.stat().st_mtime_ns),
            "workflow": "overlap-center50-v1",
        },
    }
    validation.write_text(json.dumps(report))
    assert not validated_output(
        tile, source, "olmoearth-v1.2-nano", "revision", "olmoearth"
    )

    report["tags"].update(
        workflow="overlap-center50-v2", input_normalization="computed-2std"
    )
    validation.write_text(json.dumps(report))
    assert not validated_output(
        tile, other_source, "olmoearth-v1.2-nano", "revision", "olmoearth"
    )


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        test_corrected_olmoearth_output_is_reusable(root / "corrected")
        test_old_or_wrong_source_olmoearth_output_is_replanned(root / "rejected")
    print("PASS: planner only reuses corrected outputs from the selected source")
