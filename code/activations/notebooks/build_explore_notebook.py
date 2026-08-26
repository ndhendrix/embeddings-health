#!/usr/bin/env python
"""Generate explore_activations.ipynb.

The notebook is written from here rather than by hand so its code cells can be
linted and diffed as ordinary Python. Regenerate after editing:

    python notebooks/build_explore_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    """Add a markdown cell."""
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    """Add a code cell."""
    CELLS.append(("code", text.strip("\n")))


# ----------------------------------------------------------------- title

md(
    """
# Exploring OlmoEarth activations

Everything produced by the pipeline, in one place:

| Stage | Script | Output |
|---|---|---|
| Encode | `encode_location.py` | `<name>.npz`, `<name>_chip.npz` |
| Label | `fetch_labels.py` | `<name>_labels.npz` |
| Correlate | `correlate_labels.py` | `label_correlations.csv` |

Nothing here runs the model or downloads anything. It reads what is already on
disk, so every cell is cheap and re-runnable.

**Set `LOCATION` and `LABEL` in the parameters cell** and re-run from there down.
"""
)

# ----------------------------------------------------------------- setup

md("## Setup")

code(
    '''
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_repo() -> Path:
    """Walk up for the repository root, so the notebook works from anywhere."""
    for parent in [Path.cwd(), *Path.cwd().parents]:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("not inside the repository")


REPO = find_repo()
PKG = REPO / "code" / "activations"
ARTIFACTS = PKG / "artifacts"
CORRELATIONS = ARTIFACTS / "label_correlations.csv"

# Import the package itself, for the location registry and label metadata.
sys.path.insert(0, str(PKG))
from olmoearth_activations import locations as loc  # noqa: E402

print(f"repo      {REPO}")
print(f"artifacts {ARTIFACTS}")
print(f"locations {', '.join(loc.load_locations())}")
'''
)

# ------------------------------------------------------------- loaders

md(
    """
## Loaders

One function per artifact type. `RGB` uses bands 2/1/0 — the band order is
`B02 B03 B04 B08 ...`, so red/green/blue are indices 2, 1, 0 — with a percentile
stretch, because a few bright roofs otherwise crush everything else to grey.
"""
)

code(
    '''
def load_embeddings(name: str) -> np.ndarray:
    """Return (H, W, D) embeddings for a location."""
    with np.load(ARTIFACTS / name / f"{name}.npz", allow_pickle=True) as data:
        return data["embeddings"]


def load_chip(name: str) -> np.ndarray:
    """Return the raw (H, W, 12) reflectance chip."""
    with np.load(ARTIFACTS / name / f"{name}_chip.npz") as data:
        return data["chip"]


def load_labels(name: str) -> dict[str, np.ndarray]:
    """Return {label: (H', W') coverage fraction} for a location."""
    path = ARTIFACTS / name / f"{name}_labels.npz"
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=True) as data:
        names = [str(v) for v in data["label_names"]]
        grids = data["labels"]
        sources = (
            [str(v) for v in data["label_sources"]]
            if "label_sources" in data
            else ["osm"] * len(names)
        )
    LABEL_SOURCE.update(dict(zip(names, sources)))
    return {n: grids[i] for i, n in enumerate(names)}


LABEL_SOURCE: dict[str, str] = {}


def display_label(label: str) -> str:
    """Annotate a label with its source, matching fetch_labels.py."""
    tag = {"worldcover": "WorldCover", "osm": "OSM"}.get(
        LABEL_SOURCE.get(label, "osm"), "?"
    )
    shown = label[4:] if label.startswith("osm_") else label
    return f"{tag}-{shown}"


def rgb(chip: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    """Percentile-stretched true-colour composite."""
    out = chip[:, :, [2, 1, 0]].astype(np.float32)
    lo, hi = np.percentile(out, low), np.percentile(out, high)
    return np.clip((out - lo) / max(hi - lo, 1e-6), 0, 1)


def available() -> pd.DataFrame:
    """What exists on disk, per location."""
    rows = []
    for name in loc.load_locations():
        emb = ARTIFACTS / name / f"{name}.npz"
        labels = load_labels(name)
        rows.append(
            {
                "location": name,
                "embeddings": emb.exists(),
                "chip": (ARTIFACTS / name / f"{name}_chip.npz").exists(),
                "n_labels": len(labels),
                "labels": ", ".join(sorted(labels)) or "-",
            }
        )
    return pd.DataFrame(rows)


available()
'''
)

# ------------------------------------------------------- parameters cell

md(
    """
## Parameters

**This is the cell to edit.** Everything below reads from it.
"""
)

code(
    '''
LOCATION = "westminster"   # any location with artifacts -- see the table above
LABEL = "canopy"           # any label present for that location
TOP_K = 6                  # how many dimensions to show

emb = load_embeddings(LOCATION)
chip = load_chip(LOCATION)
labels = load_labels(LOCATION)

print(f"{LOCATION}: embeddings {emb.shape}, chip {chip.shape}")
print(f"labels available: {', '.join(display_label(k) for k in sorted(labels))}")
if LABEL not in labels:
    print(f"\\n!! {LABEL!r} not among them -- pick one of the above")
'''
)

# ------------------------------------------------ most variable dimensions

md(
    """
## 1. Most spatially variable dimensions

Ranked by standard deviation across patches. Worth remembering this is a
*selection effect*: sorting 128 dimensions by variance surfaces the loudest, not
the most structured, so a noisy-looking panel here is expected rather than
alarming.
"""
)

code(
    '''
def most_variable(emb: np.ndarray, k: int) -> pd.DataFrame:
    """Rank dimensions by spatial standard deviation."""
    flat = emb.reshape(-1, emb.shape[-1])
    frame = pd.DataFrame(
        {
            "dim": np.arange(flat.shape[1]),
            "std": flat.std(axis=0),
            "range": flat.max(axis=0) - flat.min(axis=0),
            "mean": flat.mean(axis=0),
        }
    )
    return frame.sort_values("std", ascending=False).head(k).reset_index(drop=True)


ranked = most_variable(emb, TOP_K)
display(ranked)

fig, axes = plt.subplots(1, TOP_K, figsize=(2.5 * TOP_K, 3.0), squeeze=False)
for ax, row in zip(axes[0], ranked.itertuples(index=False)):
    ax.imshow(emb[:, :, int(row.dim)], cmap="viridis", interpolation="nearest")
    ax.set_title(f"dim {int(row.dim)}\\nsd={row.std:.3f}", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"{LOCATION}: most spatially variable dimensions", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.90))
plt.show()
'''
)

# --------------------------------------------------- top dimension vs RGB

md("## 2. The most variable dimension, beside the imagery")

code(
    '''
best = int(ranked.iloc[0]["dim"])

fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
axes[0].imshow(rgb(chip))
axes[0].set_title(f"{LOCATION}: RGB (B04/B03/B02)", fontsize=10)
im = axes[1].imshow(emb[:, :, best], cmap="viridis", interpolation="nearest")
axes[1].set_title(f"dim {best} (most variable)", fontsize=10)
fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.03)
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
plt.show()
'''
)

# --------------------------------------------------------- scene labelled

md(
    """
## 3. The scene, labelled

Every label for this location. These layers **overlap and do not partition** the
chip — a patch can be both canopy and park — and the WorldCover ones cover every
pixel while the OSM ones do not.
"""
)

code(
    '''
def show_labels(name: str, labels: dict[str, np.ndarray], chip: np.ndarray) -> None:
    """RGB plus every label mask, overlaid on the imagery."""
    order = sorted(labels)
    cols = len(order) + 1
    fig, axes = plt.subplots(1, cols, figsize=(2.7 * cols, 3.4), squeeze=False)
    axes[0][0].imshow(rgb(chip))
    axes[0][0].set_title("RGB", fontsize=10)
    axes[0][0].set_xticks([]); axes[0][0].set_yticks([])
    for ax, label in zip(axes[0][1:], order):
        ax.imshow(rgb(chip), alpha=0.5)
        ax.imshow(
            labels[label], cmap="viridis", alpha=0.65, vmin=0, vmax=1,
            interpolation="nearest",
            extent=(0, chip.shape[1], chip.shape[0], 0),
        )
        ax.set_title(
            f"{display_label(label)}\\nmean {float(labels[label].mean()):.3f}",
            fontsize=8,
        )
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{name}: label coverage", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    plt.show()


show_labels(LOCATION, labels, chip)
'''
)

# ------------------------------------------------------ run correlations

md(
    """
## 4. Run the correlation

Shells out to `correlate_labels.py` so the notebook and the command line cannot
drift apart. Set `RERUN = False` to just load the existing CSV.

The number to trust is **`probe_r2_oof`** — a linear probe fit on every location
but one, predicting the held-out one, with all those predictions pooled into a
single R². In-sample R² over ~28,000 spatially autocorrelated patches would be
close to meaningless, since neighbouring patches share ground.

`worst_fold` is the least favourable individual fold. A very negative value there
alongside a reasonable pooled score means the label is concentrated in one
location, not that the model is anti-predicting it.
"""
)

code(
    '''
RERUN = True

if RERUN:
    result = subprocess.run(
        [
            sys.executable, "scripts/correlate_labels.py",
            "--all", "--top", "8", "--out", str(CORRELATIONS),
        ],
        cwd=PKG, capture_output=True, text=True,
    )
    print(result.stdout[-4000:])
    if result.returncode != 0:
        print("STDERR:\\n", result.stderr[-2000:])

corr = pd.read_csv(CORRELATIONS)
print(f"\\n{len(corr)} rows, labels: {', '.join(sorted(corr['label'].unique()))}")
corr.head(12)
'''
)

md(
    """
### Which labels are linearly decodable at all?

One row per label. A strong `probe_r2_oof` alongside a weak `best_r` means the
concept is real but **spread across dimensions** — no single coordinate carries
it. That is the situation sparse autoencoders exist to address.

Read `n_locations` and `coverage` alongside the score. Below roughly 0.03
coverage there are too few positive patches to probe at all, whatever the model
encodes — which is a limit of this sample, not of the model.
"""
)

code(
    '''
summary = (
    corr.groupby("label")
    .agg(
        best_dim=("abs_r", lambda s: int(corr.loc[s.idxmax(), "dim"])),
        best_r=("pearson_r", lambda s: s.loc[s.abs().idxmax()]),
        probe_r2_oof=("probe_r2_oof", "first"),
        worst_fold=("probe_r2_fold_min", "first"),
        coverage=("label_mean_coverage", "first"),
        n_locations=("n_locations", "first"),
        n_patches=("n_patches", "first"),
    )
    .sort_values("probe_r2_oof", ascending=False)
)
display(summary.style.format({
    "best_r": "{:+.3f}", "probe_r2_oof": "{:+.3f}", "worst_fold": "{:+.2f}",
    "coverage": "{:.4f}", "n_patches": "{:,.0f}",
}))

fig, ax = plt.subplots(figsize=(7, 0.45 * len(summary) + 1.4))
# n_locations is part of the reading, not a footnote: a label measured on two
# locations is train-on-one test-on-one and is not comparable to one measured
# across seven. Faded bars mark those.
colours = [
    ("tab:green" if v > 0 else "tab:red")
    for v in summary["probe_r2_oof"]
]
alphas = [1.0 if n >= 5 else 0.45 for n in summary["n_locations"]]
bars = ax.barh(summary.index, summary["probe_r2_oof"], color=colours)
for bar, alpha in zip(bars, alphas):
    bar.set_alpha(alpha)
for y, (v, n) in enumerate(zip(summary["probe_r2_oof"], summary["n_locations"])):
    ax.text(v, y, f"  n={n}", va="center", fontsize=8)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("probe $R^2$, pooled out-of-fold  (faded = fewer than 5 locations)")
ax.set_title("Is each label linearly decodable from the 128-d embedding?")
ax.invert_yaxis()
fig.tight_layout()
plt.show()
'''
)

# ------------------------------------------ the parameterised comparison

md(
    """
### Coverage is the limiting factor, not the label

Plotted because it was the clearest pattern in the first run: probe scores track
how much of the chip a label covers. Anything left of the dashed line has too few
positive patches for seven locations to support a probe, whatever the model
encodes.
"""
)

code(
    '''
fig, ax = plt.subplots(figsize=(7, 4.6))
for label, row in summary.iterrows():
    ax.scatter(row["coverage"], row["probe_r2_oof"],
               s=28 + 6 * row["n_locations"],
               color="tab:green" if row["probe_r2_oof"] > 0 else "tab:red")
    ax.annotate(f"  {label}", (row["coverage"], row["probe_r2_oof"]), fontsize=8)
ax.axhline(0, color="black", linewidth=0.8)
ax.axvline(0.03, color="grey", linestyle="--", linewidth=0.9)
ax.text(0.031, ax.get_ylim()[0], " too rare to probe", fontsize=8, color="grey")
ax.set_xscale("log")
ax.set_xlabel("label mean coverage (log)")
ax.set_ylabel("probe $R^2$, pooled out-of-fold")
ax.set_title("Probe success against label abundance (marker size = n locations)")
fig.tight_layout()
plt.show()
'''
)

md(
    """
## 5. Top dimensions for a label, beside that label's mask

**The main cell.** Reads `label_correlations.csv`, pulls the highest-|r|
dimensions for `LABEL`, and puts each one next to the actual label mask for
`LOCATION`.

Change `LABEL` / `LOCATION` in the parameters cell and re-run. The correlations
are pooled across *all* locations, while the maps are for the one you chose — so
a dimension can rank highly overall and look unconvincing on a single chip.
"""
)

code(
    '''
def compare_label_to_dims(
    location: str, label: str, top_k: int = 6, corr: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Show a label mask beside the dimensions that correlate best with it."""
    table = pd.read_csv(CORRELATIONS) if corr is None else corr
    rows = table[table["label"] == label].copy()
    if rows.empty:
        raise ValueError(
            f"{label!r} not in {CORRELATIONS.name}. Available: "
            f"{', '.join(sorted(table['label'].unique()))}"
        )
    rows["abs_r"] = rows["pearson_r"].abs()
    rows = rows.sort_values("abs_r", ascending=False).head(top_k)

    emb = load_embeddings(location)
    chip = load_chip(location)
    labels = load_labels(location)
    if label not in labels:
        raise ValueError(
            f"{location!r} has no {label!r} label. It has: "
            f"{', '.join(sorted(labels))}"
        )
    mask = labels[label]

    n = len(rows)
    fig, axes = plt.subplots(1, n + 2, figsize=(2.6 * (n + 2), 3.5), squeeze=False)

    axes[0][0].imshow(rgb(chip))
    axes[0][0].set_title(f"{location}\\nRGB", fontsize=9)

    axes[0][1].imshow(mask, cmap="YlGn", vmin=0, vmax=1, interpolation="nearest")
    axes[0][1].set_title(
        f"{display_label(label)}\\nmean {float(mask.mean()):.3f}", fontsize=9
    )

    for ax, row in zip(axes[0][2:], rows.itertuples(index=False)):
        # Flip negatively-correlated dimensions so bright always means "more of
        # this label". Without it a strong negative relationship reads as an
        # unrelated pattern to the eye.
        values = emb[:, :, int(row.dim)]
        shown = -values if row.pearson_r < 0 else values
        ax.imshow(shown, cmap="YlGn", interpolation="nearest")
        sign = "(inverted)" if row.pearson_r < 0 else ""
        ax.set_title(
            f"dim {int(row.dim)}  r={row.pearson_r:+.3f}\\n{sign}", fontsize=9
        )

    for ax in axes[0]:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(
        f"{display_label(label)} vs its best-correlated dimensions "
        f"(r pooled over {int(rows.iloc[0]['n_locations'])} locations, "
        f"maps from {location})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    plt.show()
    return rows[["dim", "pearson_r", "probe_r2_oof", "n_patches", "n_locations"]]


compare_label_to_dims(LOCATION, LABEL, TOP_K, corr)
'''
)

md(
    """
### Sweep every label for this location

The same comparison for each label at once — useful for spotting which concepts
the model separates cleanly and which it does not.
"""
)

code(
    '''
for label in sorted(set(corr["label"]) & set(labels)):
    try:
        compare_label_to_dims(LOCATION, label, 4, corr)
    except ValueError as exc:
        print(f"skipped {label}: {exc}")
'''
)

md(
    """
### Same dimension across locations

Fix a dimension and a label, then look at every location. A dimension that
genuinely encodes a concept should track it everywhere, not just where it was
found.
"""
)

code(
    '''
DIM = int(corr[corr["label"] == LABEL].iloc[0]["dim"])

sites = [
    n for n in loc.load_locations()
    if (ARTIFACTS / n / f"{n}.npz").exists() and LABEL in load_labels(n)
]
fig, axes = plt.subplots(2, len(sites), figsize=(2.5 * len(sites), 5.6), squeeze=False)
for j, site in enumerate(sites):
    site_labels = load_labels(site)
    axes[0][j].imshow(
        site_labels[LABEL], cmap="YlGn", vmin=0, vmax=1, interpolation="nearest"
    )
    axes[0][j].set_title(
        f"{site}\\n{display_label(LABEL)} {float(site_labels[LABEL].mean()):.3f}",
        fontsize=8,
    )
    axes[1][j].imshow(
        load_embeddings(site)[:, :, DIM], cmap="YlGn", interpolation="nearest"
    )
    axes[1][j].set_title(f"dim {DIM}", fontsize=8)
    for i in (0, 1):
        axes[i][j].set_xticks([]); axes[i][j].set_yticks([])
fig.suptitle(
    f"dim {DIM} vs {display_label(LABEL)} across every location", fontsize=11
)
fig.tight_layout(rect=(0, 0, 1, 0.91))
plt.show()
'''
)

md(
    """
---

## Reading these results honestly

- **No p-values anywhere.** Neighbouring patches share ground, so ~28,000
  patches carry far less independent information than the count suggests. With
  seven locations, everything here is exploratory.
- **`probe_r2_oof` is the headline**, not `pearson_r`. It is scored on
  locations the probe never saw, so it answers "would this transfer". Negative
  means it did not transfer at all, which is why negatives are shown rather than
  clipped.
- **Circularity.** WorldCover and the OSM `building`/`highway`/`parking` rasters
  were all OlmoEarth pretraining modalities, so a strong result partly reflects
  the model having learned its own training target. Fine for interpreting what a
  dimension means; weaker evidence about generalisation.
- **`WorldCover-built` is not `OSM-buildings`.** The first is every impervious
  surface — roofs, roads, parking, driveways — the second is roof outlines alone.
  On `dc_medical` they read 0.607 and 0.171.
- **`WorldCover-farmland` is row crops only.** Pasture and hay land in `grass`;
  `agriculture` combines both.
- **`OSM-walkability` is pedestrian infrastructure, not a walkability index.**
  Footways, paths, steps, pedestrian streets, cycleways. Real walkability
  combines intersection density, land-use mix and destination access; for that,
  at tract level, use EPA's National Walkability Index. This label is also
  confounded with how thoroughly an area has been mapped.
- **`OSM-healthcare` and `OSM-food_retail` are mostly points.** A pharmacy is one
  OSM node with no outline, expanded here to a 60 m disc so it occupies any
  patches at all. `hospitals` is separated out because campuses do have real
  outlines.
- **Coverage predicts success better than the label does.** Across the first run,
  everything above ~0.28 coverage transferred and everything at or below 0.03 did
  not. So a null for `healthcare` means "too rare in seven chips", not "the model
  cannot see it".
"""
)

# ------------------------------------------------------------- assemble

notebook = {
    "cells": [
        {
            "cell_type": kind,
            "metadata": {},
            "source": body.splitlines(keepends=True),
            **({"outputs": [], "execution_count": None} if kind == "code" else {}),
        }
        for kind, body in CELLS
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).resolve().parent / "explore_activations.ipynb"
out.write_text(json.dumps(notebook, indent=1))
print(f"wrote {out}")
print(f"{len(CELLS)} cells ({sum(1 for k, _ in CELLS if k == 'code')} code)")
