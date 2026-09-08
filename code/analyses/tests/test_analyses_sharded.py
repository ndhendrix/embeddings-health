#!/usr/bin/env python3
"""Small end-to-end regression test for the sharded LightGBM pipeline."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

ANALYSES_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(ANALYSES_DIR))

from analyses_sharded import finalize, plan, run  # noqa: E402


def main() -> None:
    rng = np.random.default_rng(42)
    states = ["01", "04", "05", "06", "08", "09"]
    rows_per_state = 60
    n_rows = len(states) * rows_per_state
    state_values = np.repeat(states, rows_per_state)
    feature_1 = rng.normal(size=n_rows).astype(np.float32)
    feature_2 = rng.normal(size=n_rows).astype(np.float32)
    aland = rng.uniform(1e5, 1e8, size=n_rows).astype(np.float32)
    awater = rng.uniform(0, 1e6, size=n_rows).astype(np.float32)
    readi = rng.normal(size=n_rows)
    svi = rng.normal(size=n_rows)
    sdi = rng.normal(size=n_rows)
    target_acs = 2 * feature_1 - feature_2 + rng.normal(scale=0.2, size=n_rows)
    outcome = feature_1 + 0.5 * readi + rng.normal(scale=0.3, size=n_rows)
    deciles = (np.arange(n_rows) % 6 + 1).astype(np.uint8)

    table = pl.DataFrame({
        "tract_fips": [f"{state_values[i]}{i:09d}" for i in range(n_rows)],
        "state_fips": state_values,
        "index_order": np.arange(n_rows, dtype=np.uint32),
        "index_eligible": np.ones(n_rows, dtype=bool),
        "area_decile": deciles,
        "F1_MEAN": feature_1,
        "F2_MEAN": feature_2,
        "ALAND": aland,
        "AWATER": awater,
        "target_acs": target_acs,
        "outcome": outcome,
        "ReADI_CT_NR": readi,
        "SVI_NR": svi,
        "SDI_NR": sdi,
    })
    metadata = {
        "format_version": 1,
        "model": "synthetic",
        "n_rows": n_rows,
        "feature_dtype": "float32",
        "embedding_columns": ["F1_MEAN", "F2_MEAN"],
        "feature_columns": ["F1_MEAN", "F2_MEAN", "ALAND", "AWATER"],
        "area_columns": ["ALAND", "AWATER"],
        "places_measures": ["outcome"],
        "places_labels": {"outcome": "Synthetic outcome"},
        "q1_targets": [{"group": "ACS", "variable": "target_acs"}],
        "q23_outcomes": ["outcome"],
        "q4_outcomes": ["outcome"],
        "index_membership": {"target_acs": ["ADI"]},
        "index_features": ["ReADI_CT_NR", "SVI_NR", "SDI_NR"],
        "shared_holdout_states": ["06"],
        "states": states,
        "q4_states": ["01"],
        "q4_deciles": [1],
        "fips_to_abbr": {"01": "AL"},
        "area_decile_edges": [],
    }

    with tempfile.TemporaryDirectory(prefix="analysis-shards-") as temporary:
        root = Path(temporary)
        prepared = root / "prepared"
        run_root = root / "run"
        outputs = root / "outputs"
        prepared.mkdir()
        table.write_parquet(prepared / "analysis_data.parquet")
        (prepared / "metadata.json").write_text(json.dumps(metadata))

        plan(prepared, run_root, q1_batch_size=10, q23_batch_size=5)
        summary = json.loads((run_root / "plan.json").read_text())
        assert summary["tasks"] == {
            "q1": 1,
            "q23": 1,
            "q4_state": 1,
            "q4_decile": 1,
        }

        run(prepared, run_root, "q1", 0, threads=1)
        q1_path = run_root / "fragments" / "q1" / "task_0000.csv"
        q1_mtime = q1_path.stat().st_mtime_ns
        run(prepared, run_root, "q1", 0, threads=1)
        assert q1_path.stat().st_mtime_ns == q1_mtime

        run(prepared, run_root, "q23", 0, threads=1)
        run(prepared, run_root, "q4_state", 0, threads=1)
        run(prepared, run_root, "q4_decile", 0, threads=1)
        finalize(prepared, run_root, outputs)

        validation = json.loads((outputs / "sharded_analysis.validation.json").read_text())
        assert validation["status"] == "complete"
        assert validation["rows"] == {
            "acs_correlations.csv": 1,
            "places_reg.csv": 6,
            "places_residual_by_index.csv": 3,
            "q4_state.csv": 1,
            "q4_decile.csv": 1,
        }
        for name in validation["rows"]:
            assert (outputs / name).stat().st_size > 0

    print("sharded analysis synthetic regression: PASS")


if __name__ == "__main__":
    main()
