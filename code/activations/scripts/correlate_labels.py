#!/usr/bin/env python
"""Correlate embedding dimensions against OSM ground-truth labels.

Answers two different questions, and the difference between them matters more
than either number.

**Does any single dimension track this label?** Pearson correlation between each
of the 128 dimensions and each label's per-patch coverage. Cheap, interpretable,
and often disappointing -- individual dimensions are typically polysemantic, so a
concept can be plainly present in the representation while no single coordinate
carries it. A weak best-dimension correlation is therefore not evidence the model
is blind to the label.

**Is the label linearly decodable at all?** A least-squares probe from all 128
dimensions to the label, scored **leave-one-location-out and then pooled**: every
location gets predictions made while it was held out, and one R² is computed over
all of them together. In-sample R² over 28,000 spatially autocorrelated patches
would be near meaningless -- neighbouring patches share ground, so a model can
interpolate between them without learning anything generalisable. Holding out
whole locations is the only split here that answers "would this transfer to a
place the probe has never seen".

Pooling rather than averaging per-fold R² is not cosmetic. A label concentrated
in one location gives the other folds almost no target variance, so R² divides by
nearly nothing: ``farmland`` produced a single fold at -12 that swamped the other
six and made the label look actively anti-predicted rather than simply too rare
to measure. One pooled denominator, which includes between-location variance,
cannot be dominated that way. Per-fold values are still reported, since their
spread is what exposes a label that only works in one place.

Both are exploratory. Patch counts run to tens of thousands but the effective
sample size is far smaller, so no p-values are reported: a correlation that looks
overwhelming at n=24,000 may rest on a handful of independent regions.

Example:
    python scripts/correlate_labels.py --all --top 8
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

from olmoearth_activations import locations as loc

app = typer.Typer(add_completion=False)
logger = logging.getLogger(__name__)


def _load_pair(
    artifacts: Path, name: str
) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
    """Load one location's embeddings and labels, aligned on the patch grid.

    Returns:
        ``(embeddings, labels)`` where embeddings is ``(n_patches, dim)`` and
        each label is ``(n_patches,)``. None if either file is missing or the
        grids disagree.
    """
    emb_path = artifacts / name / f"{name}.npz"
    lab_path = artifacts / name / f"{name}_labels.npz"
    if not emb_path.exists() or not lab_path.exists():
        return None

    with np.load(emb_path, allow_pickle=True) as data:
        emb = data["embeddings"]
    with np.load(lab_path, allow_pickle=True) as data:
        names = [str(v) for v in data["label_names"]]
        grids = data["labels"]

    if emb.shape[:2] != grids.shape[1:]:
        # Almost always a chip_px mismatch between the encode run and the label
        # run. Silently broadcasting would correlate unrelated ground.
        logger.warning(
            "%s: embedding grid %s but label grid %s -- skipping. Were they "
            "produced with the same config?",
            name,
            emb.shape[:2],
            grids.shape[1:],
        )
        return None

    flat = emb.reshape(-1, emb.shape[-1])
    labels = {n: grids[i].reshape(-1) for i, n in enumerate(names)}
    return flat, labels


def _corr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson correlation of every column of x against y.

    Columns with no variance return 0 rather than NaN, so a constant dimension
    reads as "no relationship" instead of poisoning a sort.
    """
    xc = x - x.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    xs = np.sqrt((xc**2).sum(axis=0))
    ys = np.sqrt((yc**2).sum())
    denom = xs * ys
    return np.where(denom > 0, (xc * yc[:, None]).sum(axis=0) / np.maximum(denom, 1e-12), 0.0)


def _probe_predict(
    train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray
) -> np.ndarray | None:
    """Fit a least-squares linear probe and predict for held-out data.

    Returns:
        Predictions for ``test_x``, or None if there is too little training data
        to fit 128 coefficients.
    """
    if train_x.shape[0] < train_x.shape[1] + 2 or test_x.shape[0] < 2:
        return None
    # Centre on training statistics only; using test statistics would leak.
    mu, sigma = train_x.mean(axis=0), train_x.std(axis=0)
    sigma = np.where(sigma > 0, sigma, 1.0)
    a = np.hstack([(train_x - mu) / sigma, np.ones((train_x.shape[0], 1))])
    b = np.hstack([(test_x - mu) / sigma, np.ones((test_x.shape[0], 1))])
    coef, *_ = np.linalg.lstsq(a, train_y, rcond=None)
    return b @ coef


def _r2(true: np.ndarray, pred: np.ndarray) -> float:
    """R² of predictions against truth, or NaN if truth has no variance."""
    ss_tot = float(((true - true.mean()) ** 2).sum())
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - float(((true - pred) ** 2).sum()) / ss_tot


@app.command()
def main(
    all_locations: bool = typer.Option(
        False, "--all", help="Use every location with both embeddings and labels."
    ),
    only: Optional[str] = typer.Option(
        None, help="Comma-separated subset of location names."
    ),
    artifacts: Path = typer.Option(Path("artifacts"), help="Artifacts directory."),
    top: int = typer.Option(5, help="Dimensions to list per label."),
    min_coverage: float = typer.Option(
        0.005,
        help=(
            "Skip a label whose mean coverage is below this. A label present in "
            "0.1% of patches cannot support a correlation."
        ),
    ),
    out: Optional[Path] = typer.Option(None, help="Write the full table to CSV."),
    registry: Optional[Path] = typer.Option(None, help="Alternative locations YAML."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Report per-dimension correlations and a held-out linear probe per label."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-7s %(name)s: %(message)s",
    )

    if only:
        names = [n.strip() for n in only.split(",") if n.strip()]
    elif all_locations:
        names = list(loc.load_locations(registry))
    else:
        raise typer.BadParameter("pass --all or --only")

    per_site: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
    for name in names:
        pair = _load_pair(artifacts, name)
        if pair is None:
            continue
        per_site[name] = pair

    if not per_site:
        raise typer.BadParameter(
            f"no location under {artifacts} has both <name>.npz and "
            f"<name>_labels.npz. Run encode_location.py then fetch_labels.py."
        )

    sites = sorted(per_site)
    total_patches = sum(per_site[s][0].shape[0] for s in sites)
    dim = per_site[sites[0]][0].shape[1]
    typer.echo(
        f"{len(sites)} location(s), {total_patches} patches, {dim} dimensions\n"
        f"  {', '.join(sites)}\n"
    )

    label_names = sorted(
        {name for site in sites for name in per_site[site][1]}
    )
    rows: list[dict[str, object]] = []

    for label in label_names:
        present = [s for s in sites if label in per_site[s][1]]
        x = np.vstack([per_site[s][0] for s in present])
        y = np.concatenate([per_site[s][1][label] for s in present])

        if float(y.mean()) < min_coverage:
            typer.echo(
                f"{label:12s} SKIPPED -- mean coverage {float(y.mean()):.5f} "
                f"below --min-coverage {min_coverage}"
            )
            continue

        correlations = _corr(x, y)
        order = np.argsort(-np.abs(correlations))[:top]

        # Leave-one-location-out, then pool: predictions for every location are
        # collected while that location was held out, and ONE R2 is computed over
        # the pooled result.
        #
        # Averaging per-fold R2 instead -- the obvious approach -- is unstable
        # when a label is concentrated in one location. Hold out a location with
        # almost no farmland and its target variance is nearly zero, so any
        # non-zero prediction divides by almost nothing: a single degenerate fold
        # produced R2 = -12 and swamped the other six. Pooling puts every fold on
        # one denominator that includes between-location variance, so no single
        # fold can dominate. Per-fold values are still reported, because their
        # spread is what reveals a label that only works in one place.
        oof_true: list[np.ndarray] = []
        oof_pred: list[np.ndarray] = []
        fold_scores: list[tuple[str, float]] = []
        for held in present:
            train_sites = [s for s in present if s != held]
            if not train_sites:
                continue
            tx = np.vstack([per_site[s][0] for s in train_sites])
            ty = np.concatenate([per_site[s][1][label] for s in train_sites])
            held_y = per_site[held][1][label]
            pred = _probe_predict(tx, ty, per_site[held][0])
            if pred is None:
                continue
            oof_true.append(held_y)
            oof_pred.append(pred)
            fold_scores.append((held, _r2(held_y, pred)))

        pooled_r2 = (
            _r2(np.concatenate(oof_true), np.concatenate(oof_pred))
            if oof_true
            else float("nan")
        )
        finite = [v for _, v in fold_scores if np.isfinite(v)]
        best = int(order[0])
        typer.echo(
            f"{label:12s} coverage {float(y.mean()):.4f}  "
            f"{len(present)} location(s)  "
            f"best dim {best:3d} r={correlations[best]:+.3f}  "
            f"probe R2 (pooled out-of-fold) {pooled_r2:+.3f}"
        )
        typer.echo(
            "             top dims: "
            + "  ".join(f"{int(d)}:{correlations[d]:+.3f}" for d in order)
        )
        if fold_scores:
            spread = ", ".join(f"{s}={v:+.2f}" for s, v in fold_scores)
            typer.echo(f"             per held-out location: {spread}")
        if finite and min(finite) < -1.0:
            typer.echo(
                f"             note: worst fold R2 {min(finite):+.2f}. A fold "
                f"that extreme means the held-out location has almost no "
                f"variance in this label -- read the pooled figure, not the folds."
            )
        typer.echo("")

        for d in order:
            rows.append(
                {
                    "label": label,
                    "dim": int(d),
                    "pearson_r": float(correlations[d]),
                    "abs_r": float(abs(correlations[d])),
                    "label_mean_coverage": float(y.mean()),
                    "probe_r2_oof": pooled_r2,
                    "probe_r2_fold_min": float(min(finite)) if finite else float("nan"),
                    "probe_r2_fold_max": float(max(finite)) if finite else float("nan"),
                    "n_patches": int(y.shape[0]),
                    "n_locations": len(present),
                }
            )

    if not rows:
        typer.echo("No label had enough coverage to report.")
        return

    frame = pd.DataFrame(rows)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out, index=False)
        typer.echo(f"wrote {out}")

    typer.echo(
        "\nReading these numbers:\n"
        "  * probe_r2_oof is the honest headline: one R2 over predictions made for\n"
        "    each location while it was held out. It answers 'would this transfer'.\n"
        "  * n_locations matters as much as the score. A label present in two\n"
        "    locations is train-on-one test-on-one, and is not comparable to one\n"
        "    measured across seven.\n"
        "  * label_mean_coverage matters too. Below roughly 0.03 there are too few\n"
        "    positive patches to probe, whatever the model does or does not encode.\n"
        "  * a weak best-dim r with a healthy probe R2 means the concept is present\n"
        "    but spread across dimensions -- the case sparse autoencoders exist for.\n"
        "  * no p-values: neighbouring patches share ground, so the effective\n"
        "    sample is far below the patch count. Treat everything as exploratory,\n"
        f"    especially with only {len(sites)} location(s)."
    )


if __name__ == "__main__":
    app()
