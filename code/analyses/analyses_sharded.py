#!/usr/bin/env python3
"""Restartable, result-equivalent LightGBM shards for embeddings-health."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold


MODEL_ORDER = [
    "ReADI", "SVI", "SDI", "All indices", "Embeddings",
    "All indices + Embeddings",
]
INDEX_NAMES = ["ReADI", "SVI", "SDI"]
INDEX_COLUMN = {"ReADI": "ReADI_CT_NR", "SVI": "SVI_NR", "SDI": "SDI_NR"}
MODEL_COLORS = {
    "ReADI": "#E53935",
    "SVI": "#FF7043",
    "SDI": "#FFA726",
    "All indices": "#AB47BC",
    "Embeddings": "#2196F3",
    "All indices + Embeddings": "#43A047",
}


def _metadata(prepared_dir: Path) -> dict[str, Any]:
    path = prepared_dir / "metadata.json"
    table = prepared_dir / "analysis_data.parquet"
    if not path.is_file() or not table.is_file():
        raise FileNotFoundError(f"Incomplete prepared data in {prepared_dir}")
    data = json.loads(path.read_text())
    if data.get("format_version") != 1:
        raise ValueError(f"Unsupported prepared-data format: {data.get('format_version')}")
    return data


def _atomic_csv(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    frame.write_csv(temporary)
    os.replace(temporary, path)


def _atomic_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    count = 0
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    os.replace(temporary, path)
    return count


def _batches(items: list[Any], size: int) -> list[list[Any]]:
    if size < 1:
        raise ValueError("Batch size must be positive")
    return [items[start:start + size] for start in range(0, len(items), size)]


def plan(prepared_dir: Path, run_root: Path, q1_batch_size: int, q23_batch_size: int) -> None:
    meta = _metadata(prepared_dir)
    manifest_dir = run_root / "manifests"
    definitions = {
        "q1": [
            {"task_id": task_id, "targets": batch}
            for task_id, batch in enumerate(_batches(meta["q1_targets"], q1_batch_size))
        ],
        "q23": [
            {"task_id": task_id, "outcomes": batch}
            for task_id, batch in enumerate(_batches(meta["q23_outcomes"], q23_batch_size))
        ],
        "q4_state": [
            {"task_id": task_id, "state_fips": state}
            for task_id, state in enumerate(meta["q4_states"])
        ],
        "q4_decile": [
            {"task_id": task_id, "decile": int(decile)}
            for task_id, decile in enumerate(meta["q4_deciles"])
        ],
    }
    counts = {}
    for stage, rows in definitions.items():
        counts[stage] = _atomic_jsonl(rows, manifest_dir / f"{stage}.jsonl")
    summary = {
        "format_version": 1,
        "prepared_dir": str(prepared_dir),
        "q1_batch_size": q1_batch_size,
        "q23_batch_size": q23_batch_size,
        "tasks": counts,
        "expected_rows": {
            "q1": len(meta["q1_targets"]),
            "q2": len(meta["q23_outcomes"]) * len(MODEL_ORDER),
            "q3": len(meta["q23_outcomes"]) * len(INDEX_NAMES),
            "q4_state": len(meta["q4_states"]),
            "q4_decile": len(meta["q4_deciles"]),
        },
    }
    _atomic_json(summary, run_root / "plan.json")
    print(json.dumps(summary, indent=2, sort_keys=True))


def _task(run_root: Path, stage: str, task_id: int) -> dict[str, Any]:
    manifest = run_root / "manifests" / f"{stage}.jsonl"
    with manifest.open() as stream:
        for line_number, line in enumerate(stream):
            if line_number == task_id:
                task = json.loads(line)
                if task["task_id"] != task_id:
                    raise ValueError(f"Manifest task mismatch at row {task_id}")
                return task
    raise IndexError(f"No {stage} manifest row for task {task_id}")


def _full_model(threads: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        random_state=42,
        verbosity=-1,
        n_jobs=threads,
    )


def _q4_model(threads: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        n_estimators=30,
        learning_rate=0.15,
        num_leaves=16,
        min_child_samples=20,
        random_state=42,
        verbosity=-1,
        n_jobs=threads,
    )


def _valid(values: np.ndarray) -> np.ndarray:
    return ~np.isnan(values)


def _q1_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray, threads: int) -> tuple[float, float]:
    scores = []
    for train, validation in GroupKFold(n_splits=5).split(X, y, groups=groups):
        model = _full_model(threads)
        model.fit(X[train], y[train])
        scores.append(r2_score(y[validation], model.predict(X[validation])))
        del model
        gc.collect()
    return float(np.mean(scores)), float(np.std(scores))


def _run_q1(prepared_dir: Path, run_root: Path, task_id: int, threads: int) -> None:
    output = run_root / "fragments" / "q1" / f"task_{task_id:04d}.csv"
    if output.is_file() and output.stat().st_size > 0:
        print(f"SKIP complete fragment: {output}")
        return
    meta = _metadata(prepared_dir)
    task = _task(run_root, "q1", task_id)
    features = meta["feature_columns"]
    variables = [target["variable"] for target in task["targets"]]
    columns = list(dict.fromkeys(["state_fips"] + features + variables))
    frame = pl.read_parquet(prepared_dir / "analysis_data.parquet", columns=columns)
    X = frame.select(features).to_numpy().astype(np.float32, copy=False)
    states = frame["state_fips"].to_numpy()
    targets = {column: frame[column].to_numpy().astype(float, copy=False) for column in variables}
    del frame
    gc.collect()

    rows = []
    membership = meta["index_membership"]
    for target in task["targets"]:
        group = target["group"]
        variable = target["variable"]
        y_all = targets[variable]
        mask = _valid(y_all)
        if mask.sum() < 100:
            raise ValueError(f"{variable} unexpectedly has fewer than 100 valid rows")
        mean, std = _q1_cv(X[mask], y_all[mask], states[mask], threads)
        rows.append({
            "group": group,
            "variable": variable,
            "r2_mean": mean,
            "r2_std": std,
            "indices": ", ".join(membership.get(variable, [])),
        })
        print(f"Q1 {variable}: mean={mean:.6f} std={std:.6f}")
    _atomic_csv(pl.DataFrame(rows), output)


def _feature_pair(
    model_name: str,
    rows: np.ndarray,
    embeddings: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    if model_name == "ReADI":
        return indices[rows, 0:1]
    if model_name == "SVI":
        return indices[rows, 1:2]
    if model_name == "SDI":
        return indices[rows, 2:3]
    if model_name == "All indices":
        return indices[rows]
    if model_name == "Embeddings":
        return embeddings[rows]
    if model_name == "All indices + Embeddings":
        return np.column_stack((indices[rows], embeddings[rows]))
    raise ValueError(f"Unknown model: {model_name}")


def _run_q23(prepared_dir: Path, run_root: Path, task_id: int, threads: int) -> None:
    output_dir = run_root / "fragments" / "q23"
    q2_output = output_dir / f"task_{task_id:04d}.q2.csv"
    q3_output = output_dir / f"task_{task_id:04d}.q3.csv"
    marker = output_dir / f"task_{task_id:04d}.done.json"
    if all(path.is_file() and path.stat().st_size > 0 for path in [q2_output, q3_output, marker]):
        print(f"SKIP complete fragment pair: {marker}")
        return

    meta = _metadata(prepared_dir)
    task = _task(run_root, "q23", task_id)
    features = meta["feature_columns"]
    outcomes = task["outcomes"]
    index_columns = meta["index_features"]
    columns = list(dict.fromkeys(["state_fips"] + features + index_columns + outcomes))
    frame = (
        pl.scan_parquet(prepared_dir / "analysis_data.parquet")
        .filter(pl.col("index_eligible"))
        .sort("index_order")
        .select(columns)
        .collect()
    )
    embeddings = frame.select(features).to_numpy().astype(np.float32, copy=False)
    indices = frame.select(index_columns).to_numpy().astype(float, copy=False)
    states = frame["state_fips"].to_numpy()
    target_values = {column: frame[column].to_numpy().astype(float, copy=False) for column in outcomes}
    del frame
    gc.collect()

    holdout = set(meta["shared_holdout_states"])
    base_test = np.isin(states, list(holdout))
    base_train = ~base_test
    q2_rows = []
    q3_rows = []
    for outcome in outcomes:
        y = target_values[outcome]
        valid = _valid(y)
        train_rows = np.flatnonzero(base_train & valid)
        test_rows = np.flatnonzero(base_test & valid)
        if len(train_rows) < 100 or len(test_rows) < 50:
            raise ValueError(f"{outcome} unexpectedly lacks train/test observations")

        for model_name in MODEL_ORDER:
            X_train = _feature_pair(model_name, train_rows, embeddings, indices)
            X_test = _feature_pair(model_name, test_rows, embeddings, indices)
            model = _full_model(threads)
            model.fit(X_train, y[train_rows])
            score = float(r2_score(y[test_rows], model.predict(X_test)))
            q2_rows.append({
                "outcome": outcome,
                "model": model_name,
                "r2": score,
                "n_train": len(train_rows),
                "n_test": len(test_rows),
            })
            del X_train, X_test, model
            gc.collect()

        for index_name in INDEX_NAMES:
            column_index = INDEX_NAMES.index(index_name)
            model_index = _full_model(threads)
            model_index.fit(indices[train_rows, column_index:column_index + 1], y[train_rows])
            prediction_train = model_index.predict(indices[train_rows, column_index:column_index + 1])
            prediction_test = model_index.predict(indices[test_rows, column_index:column_index + 1])
            residual_train = y[train_rows] - prediction_train
            residual_test = y[test_rows] - prediction_test
            index_r2 = float(r2_score(y[test_rows], prediction_test))

            model_residual = _full_model(threads)
            model_residual.fit(embeddings[train_rows], residual_train)
            residual_r2 = float(r2_score(residual_test, model_residual.predict(embeddings[test_rows])))
            additional = max(0.0, residual_r2) * max(0.0, 1.0 - index_r2)
            q3_rows.append({
                "index": index_name,
                "outcome": outcome,
                "r2_index": index_r2,
                "r2_residual": residual_r2,
                "additional_var": additional,
                "n_train": len(train_rows),
                "n_test": len(test_rows),
            })
            del model_index, model_residual
            gc.collect()
        print(f"Q2/Q3 complete: {outcome}")

    _atomic_csv(pl.DataFrame(q2_rows), q2_output)
    _atomic_csv(pl.DataFrame(q3_rows), q3_output)
    _atomic_json({"q2_rows": len(q2_rows), "q3_rows": len(q3_rows)}, marker)


def _cv_subset(X: np.ndarray, y: np.ndarray, threads: int) -> float:
    if len(y) < 10:
        return float("nan")
    scores = []
    for train, validation in KFold(n_splits=2, shuffle=True, random_state=42).split(X):
        model = _q4_model(threads)
        model.fit(X[train], y[train])
        scores.append(r2_score(y[validation], model.predict(X[validation])))
        del model
    return float(np.nanmean(scores))


def _q4_arrays(frame: pl.DataFrame, meta: dict[str, Any]) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    embeddings = frame.select(meta["feature_columns"]).to_numpy().astype(np.float32, copy=False)
    indices = {
        name: frame[INDEX_COLUMN[name]].to_numpy().astype(float, copy=False).reshape(-1, 1)
        for name in INDEX_NAMES
    }
    targets = {
        outcome: frame[outcome].to_numpy().astype(float, copy=False)
        for outcome in meta["q4_outcomes"]
    }
    return embeddings, indices, targets


def _q4_means(
    embeddings: np.ndarray,
    indices: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    threads: int,
) -> tuple[dict[str, float], int]:
    scores = {name: [] for name in INDEX_NAMES + ["Embeddings"]}
    for outcome, y_all in targets.items():
        valid = _valid(y_all)
        if valid.sum() < 50:
            continue
        y = y_all[valid]
        for name in INDEX_NAMES:
            scores[name].append(_cv_subset(indices[name][valid], y, threads))
        scores["Embeddings"].append(_cv_subset(embeddings[valid], y, threads))
    if not scores["ReADI"]:
        raise ValueError("Q4 shard contains no outcome with at least 50 observations")
    means = {name: float(np.nanmean(values)) for name, values in scores.items()}
    return means, len(scores["ReADI"])


def _run_q4_state(prepared_dir: Path, run_root: Path, task_id: int, threads: int) -> None:
    output = run_root / "fragments" / "q4_state" / f"task_{task_id:04d}.csv"
    if output.is_file() and output.stat().st_size > 0:
        print(f"SKIP complete fragment: {output}")
        return
    meta = _metadata(prepared_dir)
    task = _task(run_root, "q4_state", task_id)
    state = task["state_fips"]
    columns = list(dict.fromkeys(meta["feature_columns"] + meta["index_features"] + meta["q4_outcomes"]))
    frame = (
        pl.scan_parquet(prepared_dir / "analysis_data.parquet")
        .filter((pl.col("state_fips") == state) & pl.col("index_eligible"))
        .sort("index_order")
        .select(columns)
        .collect()
    )
    embeddings, indices, targets = _q4_arrays(frame, meta)
    means, n_outcomes = _q4_means(embeddings, indices, targets, threads)
    row = {
        "state": meta["fips_to_abbr"].get(state, state),
        "r2_readi": means["ReADI"],
        "r2_svi": means["SVI"],
        "r2_sdi": means["SDI"],
        "r2_emb": means["Embeddings"],
        "n_outcomes": n_outcomes,
        "n_tracts": len(frame),
    }
    _atomic_csv(pl.DataFrame([row]), output)


def _run_q4_decile(prepared_dir: Path, run_root: Path, task_id: int, threads: int) -> None:
    output = run_root / "fragments" / "q4_decile" / f"task_{task_id:04d}.csv"
    if output.is_file() and output.stat().st_size > 0:
        print(f"SKIP complete fragment: {output}")
        return
    meta = _metadata(prepared_dir)
    task = _task(run_root, "q4_decile", task_id)
    decile = int(task["decile"])
    columns = list(dict.fromkeys(meta["feature_columns"] + meta["index_features"] + meta["q4_outcomes"]))
    frame = (
        pl.scan_parquet(prepared_dir / "analysis_data.parquet")
        .filter(pl.col("area_decile") == decile)
        .sort("index_order")
        .select(columns)
        .collect()
    )
    embeddings, indices, targets = _q4_arrays(frame, meta)
    means, _ = _q4_means(embeddings, indices, targets, threads)
    row = {
        "decile": decile,
        "median_aland_km2": float(np.nanmedian(frame["ALAND"].to_numpy()) / 1e6),
        "r2_readi": means["ReADI"],
        "r2_svi": means["SVI"],
        "r2_sdi": means["SDI"],
        "r2_emb": means["Embeddings"],
        "n_tracts": len(frame),
    }
    _atomic_csv(pl.DataFrame([row]), output)


def run(prepared_dir: Path, run_root: Path, stage: str, task_id: int, threads: int) -> None:
    started = time.time()
    if stage == "q1":
        _run_q1(prepared_dir, run_root, task_id, threads)
    elif stage == "q23":
        _run_q23(prepared_dir, run_root, task_id, threads)
    elif stage == "q4_state":
        _run_q4_state(prepared_dir, run_root, task_id, threads)
    elif stage == "q4_decile":
        _run_q4_decile(prepared_dir, run_root, task_id, threads)
    else:
        raise ValueError(f"Unknown stage: {stage}")
    print(f"Complete: stage={stage} task={task_id} elapsed={(time.time() - started) / 60:.1f} min")


def _manifest_rows(run_root: Path, stage: str) -> list[dict[str, Any]]:
    path = run_root / "manifests" / f"{stage}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _read_fragments(paths: list[Path], stage: str) -> pl.DataFrame:
    missing = [path for path in paths if not path.is_file() or path.stat().st_size == 0]
    if missing:
        preview = ", ".join(path.name for path in missing[:10])
        raise FileNotFoundError(f"{stage}: {len(missing)} missing fragments: {preview}")
    return pl.concat([pl.read_csv(path) for path in paths], how="diagonal_relaxed")


def _validate_keys(frame: pl.DataFrame, columns: list[str], expected: set[tuple[Any, ...]], stage: str) -> None:
    actual_rows = [tuple(row) for row in frame.select(columns).iter_rows()]
    actual = set(actual_rows)
    if len(actual_rows) != len(actual):
        raise ValueError(f"{stage}: duplicate result keys")
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        raise ValueError(f"{stage}: missing={len(missing)} extra={len(extra)}")


def _plot_results(
    meta: dict[str, Any],
    outputs_dir: Path,
    q1: pl.DataFrame,
    q2: pl.DataFrame,
    q3: pl.DataFrame,
    q4_state: pl.DataFrame,
    q4_decile: pl.DataFrame,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = meta["places_labels"]
    model = meta["model"]
    acs = q1.filter(pl.col("group") == "ACS").sort("r2_mean")
    fig, ax = plt.subplots(figsize=(10, max(8, 0.38 * len(acs))))
    values = acs["r2_mean"].to_list()
    ax.barh(
        acs["variable"].to_list(), values, xerr=acs["r2_std"].to_list(),
        color=["#2196F3" if value >= 0 else "#E53935" for value in values],
        capsize=3, alpha=0.85, edgecolor="white", linewidth=0.5,
    )
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Cross-validated R²  (5-fold)", fontsize=11)
    ax.set_xlim(-0.15, 1.0)
    ax.set_title("ACS tract variables", fontsize=12, fontweight="bold")
    ax.tick_params(axis="y", labelsize=9)
    plt.suptitle(
        f"Variance in tract-level ACS variables explained by {model} embeddings  |  "
        f"n = {meta['n_rows']:,} tracts (2022)", fontsize=12, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(outputs_dir / "r2_embeddings.png", dpi=150, bbox_inches="tight")
    plt.close()

    q2_rows = q2.to_dicts()
    q2_lookup = {(row["outcome"], row["model"]): row["r2"] for row in q2_rows}
    outcome_order = sorted(meta["q23_outcomes"], key=lambda value: q2_lookup[(value, "ReADI")])
    fig, ax = plt.subplots(figsize=(9, max(4, 0.6 * len(outcome_order))))
    for model_name in MODEL_ORDER:
        xs = [q2_lookup[(outcome, model_name)] for outcome in outcome_order]
        ax.scatter(xs, range(len(outcome_order)), color=MODEL_COLORS[model_name], label=model_name,
                   zorder=3, s=80, edgecolors="white", linewidths=0.6)
    for index, outcome in enumerate(outcome_order):
        ax.plot([q2_lookup[(outcome, name)] for name in MODEL_ORDER],
                [index] * len(MODEL_ORDER), color="grey", linewidth=0.8, alpha=0.5)
    ax.set_yticks(range(len(outcome_order)))
    ax.set_yticklabels([labels.get(outcome, outcome) for outcome in outcome_order], fontsize=10)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("R²  on held-out states (20%)", fontsize=11)
    holdout_names = sorted(meta["fips_to_abbr"].get(value, value) for value in meta["shared_holdout_states"])
    ax.set_title(f"PLACES outcomes: {model} embeddings vs. social risk indices\n"
                 f"Held-out states: {', '.join(holdout_names)}", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outputs_dir / "places_reg.png", dpi=150, bbox_inches="tight")
    plt.close()

    q3_rows = q3.to_dicts()
    q3_lookup = {(row["index"], row["outcome"]): row for row in q3_rows}
    fig, axes = plt.subplots(len(INDEX_NAMES), 2, figsize=(14, max(6, 0.5 * len(outcome_order)) * 3))
    for row_index, index_name in enumerate(INDEX_NAMES):
        order = sorted(meta["q23_outcomes"], key=lambda value: q3_lookup[(index_name, value)]["r2_residual"])
        positions = np.arange(len(order))
        residual = [q3_lookup[(index_name, value)]["r2_residual"] for value in order]
        additional = [q3_lookup[(index_name, value)]["additional_var"] for value in order]
        text = [labels.get(value, value) for value in order]
        axes[row_index, 0].barh(positions, residual, color=[MODEL_COLORS[index_name] if value >= 0 else "#9E9E9E" for value in residual])
        axes[row_index, 0].axvline(0, color="black", linewidth=0.8, linestyle="--")
        axes[row_index, 0].set_yticks(positions); axes[row_index, 0].set_yticklabels(text, fontsize=9)
        axes[row_index, 0].set_xlabel(f"R²  of embeddings on {index_name} residuals")
        axes[row_index, 0].set_title(f"{index_name}: share of unexplained variance\ncaptured by embeddings", fontweight="bold")
        axes[row_index, 1].barh(positions, additional, color="#43A047")
        axes[row_index, 1].axvline(0, color="black", linewidth=0.8, linestyle="--")
        axes[row_index, 1].set_yticks(positions); axes[row_index, 1].set_yticklabels(text, fontsize=9)
        axes[row_index, 1].set_xlabel("Additional variance explained")
        axes[row_index, 1].set_title(f"{index_name}: absolute additional variance\nexplained by embeddings", fontweight="bold")
    plt.suptitle(f"Q3: {model} signal beyond each index — held-out states: {', '.join(holdout_names)}",
                 fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(outputs_dir / "places_residual_by_index.png", dpi=150, bbox_inches="tight")
    plt.close()

    states = q4_state.to_pandas()
    deciles = q4_decile.sort("decile").to_pandas()
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    for axis, column, name in [
        (axes[0, 0], "r2_readi", "ReADI"),
        (axes[0, 1], "r2_svi", "SVI"),
        (axes[1, 0], "r2_sdi", "SDI"),
    ]:
        axis.scatter(states[column], states["r2_emb"], s=states["n_tracts"] / 30,
                     color="#2196F3", alpha=0.7, edgecolors="white", linewidths=0.5)
        for _, row in states.iterrows():
            axis.annotate(row["state"], (row[column], row["r2_emb"]), fontsize=6.5,
                          ha="center", va="bottom", xytext=(0, 3), textcoords="offset points")
        low = min(states[column].min(), states["r2_emb"].min()) - 0.02
        high = max(states[column].max(), states["r2_emb"].max()) + 0.02
        axis.plot([low, high], [low, high], color="grey", linewidth=1, linestyle="--")
        axis.set_xlim([low, high]); axis.set_ylim([low, high])
        axis.set_xlabel(f"Avg R²  — {name} (within-state CV)")
        axis.set_ylabel("Avg R²  — Embeddings (within-state CV)")
        axis.set_title(f"{name} vs. Embeddings by state\n(bubble size ∝ n tracts)", fontweight="bold")
        axis.grid(alpha=0.25)
    axis = axes[1, 1]
    for column, name in [("r2_readi", "ReADI"), ("r2_svi", "SVI"), ("r2_sdi", "SDI"), ("r2_emb", "Embeddings")]:
        axis.plot(deciles["decile"], deciles[column], color=MODEL_COLORS.get(name, "#2196F3"),
                  marker="o", linewidth=1.8, markersize=6, label=name)
    axis.set_xticks(deciles["decile"])
    axis.set_xlabel("Tract-size decile  (D1 = smallest / most urban)")
    axis.set_ylabel("Avg R²  across PLACES outcomes (within-decile CV)")
    axis.set_title("Performance by tract size\n(proxy for urban ↔ rural)", fontweight="bold")
    axis.legend(); axis.grid(alpha=0.25)
    plt.suptitle(f"Q4: Heterogeneity of {model} vs. index performance", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(outputs_dir / "q4_heterogeneity.png", dpi=150, bbox_inches="tight")
    plt.close()


def finalize(prepared_dir: Path, run_root: Path, outputs_dir: Path) -> None:
    meta = _metadata(prepared_dir)
    q1_tasks = _manifest_rows(run_root, "q1")
    q23_tasks = _manifest_rows(run_root, "q23")
    q4_state_tasks = _manifest_rows(run_root, "q4_state")
    q4_decile_tasks = _manifest_rows(run_root, "q4_decile")
    q1 = _read_fragments([run_root / "fragments" / "q1" / f"task_{row['task_id']:04d}.csv" for row in q1_tasks], "q1")
    q2 = _read_fragments([run_root / "fragments" / "q23" / f"task_{row['task_id']:04d}.q2.csv" for row in q23_tasks], "q2")
    q3 = _read_fragments([run_root / "fragments" / "q23" / f"task_{row['task_id']:04d}.q3.csv" for row in q23_tasks], "q3")
    q4_state = _read_fragments([run_root / "fragments" / "q4_state" / f"task_{row['task_id']:04d}.csv" for row in q4_state_tasks], "q4_state")
    q4_decile = _read_fragments([run_root / "fragments" / "q4_decile" / f"task_{row['task_id']:04d}.csv" for row in q4_decile_tasks], "q4_decile")

    _validate_keys(q1, ["group", "variable"], {(row["group"], row["variable"]) for row in meta["q1_targets"]}, "q1")
    _validate_keys(q2, ["outcome", "model"], {(outcome, model) for outcome in meta["q23_outcomes"] for model in MODEL_ORDER}, "q2")
    _validate_keys(q3, ["index", "outcome"], {(name, outcome) for outcome in meta["q23_outcomes"] for name in INDEX_NAMES}, "q3")
    _validate_keys(q4_state, ["state"], {(meta["fips_to_abbr"].get(state, state),) for state in meta["q4_states"]}, "q4_state")
    _validate_keys(q4_decile, ["decile"], {(int(decile),) for decile in meta["q4_deciles"]}, "q4_decile")

    outputs_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "acs_correlations.csv": q1.sort("r2_mean", descending=True),
        "places_reg.csv": q2.sort(["outcome", "model"]),
        "places_residual_by_index.csv": q3.sort(["index", "r2_residual"], descending=[False, True]),
        "q4_state.csv": q4_state.sort("state"),
        "q4_decile.csv": q4_decile.sort("decile"),
    }
    for name, frame in outputs.items():
        _atomic_csv(frame, outputs_dir / name)
    _plot_results(meta, outputs_dir, q1, q2, q3, q4_state, q4_decile)
    validation = {
        "status": "complete",
        "prepared_dir": str(prepared_dir),
        "run_root": str(run_root),
        "rows": {name: len(frame) for name, frame in outputs.items()},
    }
    _atomic_json(validation, outputs_dir / "sharded_analysis.validation.json")
    print(json.dumps(validation, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--prepared-dir", type=Path, required=True)
    plan_parser.add_argument("--run-root", type=Path, required=True)
    plan_parser.add_argument("--q1-batch-size", type=int, default=10)
    plan_parser.add_argument("--q23-batch-size", type=int, default=5)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--prepared-dir", type=Path, required=True)
    run_parser.add_argument("--run-root", type=Path, required=True)
    run_parser.add_argument("--stage", choices=["q1", "q23", "q4_state", "q4_decile"], required=True)
    run_parser.add_argument("--task-id", type=int, required=True)
    run_parser.add_argument("--threads", type=int, default=1)

    final_parser = subparsers.add_parser("finalize")
    final_parser.add_argument("--prepared-dir", type=Path, required=True)
    final_parser.add_argument("--run-root", type=Path, required=True)
    final_parser.add_argument("--outputs-dir", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "plan":
        plan(args.prepared_dir, args.run_root, args.q1_batch_size, args.q23_batch_size)
    elif args.command == "run":
        run(args.prepared_dir, args.run_root, args.stage, args.task_id, args.threads)
    elif args.command == "finalize":
        finalize(args.prepared_dir, args.run_root, args.outputs_dir)


if __name__ == "__main__":
    main()
