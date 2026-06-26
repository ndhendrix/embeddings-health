"""Figure 2: per-outcome R² comparison across predictor types and embedding models.

Usage:
    python figure2.py                     # CHECKUP (default)
    python figure2.py --outcome DIABETES
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Repository layout ─────────────────────────────────────────────────────────
REPO    = Path(__file__).resolve().parents[2]
OUTPUTS = REPO / "outputs"
FIG_DIR = OUTPUTS / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Canonical index baseline ──────────────────────────────────────────────────
# Prithvi Tiny and 300M-TL share the same holdout states, so their index-only
# R² values are identical. We use Prithvi Tiny's CSV as the index baseline
# until all models are harmonised to a single train/test split.
INDEX_CSV = OUTPUTS / "prithvi_tiny" / "prithvi_tiny_places_reg.csv"

# ── Model registry ─────────────────────────────────────────────────────────────
# Replace None with the model's places_reg.csv path when embeddings are ready.
MODEL_SOURCES: dict[str, Path | None] = {
    "AlphaEarth":         OUTPUTS / "alphaearth_foundations" / "places_reg.csv",
    "Prithvi Tiny-TL":    OUTPUTS / "prithvi_tiny" / "prithvi_tiny_places_reg.csv",
    "Prithvi 300M-TL":    OUTPUTS / "prithvi_300M-TL" / "places_reg.csv",
    "OlmoEarth-1.1 Nano": None,
    "OlmoEarth-1.1 Base": None,
    "Clay v1.5":          None,
}
# Stable seeds so mocked values don't change between runs
_MOCK_SEEDS = {"OlmoEarth-1.1 Nano": 101, "OlmoEarth-1.1 Base": 202, "Clay v1.5": 303}

# ── Outcome labels ─────────────────────────────────────────────────────────────
OUTCOME_LABELS: dict[str, str] = {
    "ACCESS2":     "Lack of health insurance",
    "ARTHRITIS":   "Arthritis",
    "BINGE":       "Binge drinking",
    "BPHIGH":      "High blood pressure",
    "BPMED":       "BP medication use",
    "CANCER":      "Cancer (excl. skin)",
    "CASTHMA":     "Current asthma",
    "CHD":         "Coronary heart disease",
    "CHECKUP":     "Annual checkup",
    "CHOLSCREEN":  "Cholesterol screening",
    "COGNITION":   "Cognitive decline",
    "COLON_SCREEN":"Colorectal cancer screening",
    "COPD":        "COPD",
    "CSMOKING":    "Current smoking",
    "DENTAL":      "Dental visit",
    "DEPRESSION":  "Depression",
    "DIABETES":    "Diabetes",
    "DISABILITY":  "Any disability",
    "EMOTIONSPT":  "Emotional support",
    "FOODINSECU":  "Food insecurity",
    "FOODSTAMP":   "Food stamp use",
    "GHLTH":       "Fair/poor general health",
    "HEARING":     "Hearing disability",
    "HIGHCHOL":    "High cholesterol",
    "HOUSINSECU":  "Housing insecurity",
    "INDEPLIVE":   "Independent living difficulty",
    "LACKTRPT":    "Lack of transportation",
    "LONELINESS":  "Loneliness",
    "LPA":         "Physical inactivity",
    "MAMMOUSE":    "Mammography use",
    "MHLTH":       "Poor mental health days",
    "MOBILITY":    "Mobility disability",
    "OBESITY":     "Obesity",
    "PHLTH":       "Poor physical health days",
    "SELFCARE":    "Self-care disability",
    "SHUTUTILITY": "Utility shutoff",
    "SLEEP":       "Short sleep duration",
    "STROKE":      "Stroke",
    "TEETHLOST":   "Tooth loss",
    "VISION":      "Vision disability",
}

# ── Colorblind-friendly single-hue palettes (Oranges / Blues / Purples) ───────
# All three are distinguishable under deuteranopia and protanopia.
PALETTE_INDICES  = "Oranges"
PALETTE_EMB      = "Blues"
PALETTE_COMBINED = "Purples"


def _gradient(cmap_name: str, n: int, lo: float = 0.38, hi: float = 0.88) -> list:
    """Sample n evenly-spaced colors from a matplotlib sequential colormap."""
    cmap = plt.colormaps[cmap_name]
    return [cmap(v) for v in np.linspace(lo, hi, max(n, 1))]


# ── Data helpers ───────────────────────────────────────────────────────────────
def _load_r2(csv_path: Path, outcome: str, model_col: str) -> float:
    df  = pl.read_csv(csv_path)
    row = df.filter((pl.col("outcome") == outcome) & (pl.col("model") == model_col))
    return float(row["r2"][0]) if len(row) else float("nan")


def _mock_r2(seed: int, lo: float, hi: float) -> float:
    return float(np.random.default_rng(seed).uniform(lo, hi))


def _build_data(outcome: str) -> dict:
    # Index baseline
    idx_models = ["ReADI", "SVI", "SDI", "All indices"]
    idx_r2 = [_load_r2(INDEX_CSV, outcome, m) for m in idx_models]

    # Real values to calibrate mock range
    real_emb  = [_load_r2(p, outcome, "Embeddings")
                 for p in MODEL_SOURCES.values() if p is not None]
    real_comb = [_load_r2(p, outcome, "All indices + Embeddings")
                 for p in MODEL_SOURCES.values() if p is not None]

    emb_lo  = min(real_emb)  * 0.85;  emb_hi  = max(real_emb)  * 1.10
    comb_lo = min(real_comb) * 0.90;  comb_hi = max(real_comb) * 1.05

    emb_labels, emb_r2, comb_r2, mocked = [], [], [], []
    for name, path in MODEL_SOURCES.items():
        seed = _MOCK_SEEDS.get(name, 0)
        emb_labels.append(name)
        mocked.append(path is None)
        if path is not None:
            emb_r2.append(_load_r2(path, outcome, "Embeddings"))
            comb_r2.append(_load_r2(path, outcome, "All indices + Embeddings"))
        else:
            emb_r2.append(_mock_r2(seed,     emb_lo,  emb_hi))
            comb_r2.append(_mock_r2(seed + 1, comb_lo, comb_hi))

    return dict(
        idx_labels=idx_models,  idx_r2=idx_r2,
        emb_labels=emb_labels,  emb_r2=emb_r2,
        comb_r2=comb_r2,        mocked=mocked,
    )


# ── Drawing ────────────────────────────────────────────────────────────────────
def _draw_panel(
    ax: plt.Axes,
    labels: list[str],
    r2_vals: list[float],
    mocked: list[bool],
    palette: str,
    title: str,
    x_min: float,
    x_max: float,
) -> None:
    n     = len(labels)
    order = np.argsort(r2_vals)          # ascending → lightest at bottom, darkest at top
    colors = _gradient(palette, n)

    s_labels = [labels[i] + (" *" if mocked[i] else "") for i in order]
    s_r2     = [r2_vals[i] for i in order]

    ax.barh(range(n), s_r2, color=colors, edgecolor="white", linewidth=0.4, height=0.68)
    ax.set_yticks(range(n))
    ax.set_yticklabels(s_labels, fontsize=9)
    ax.set_xlim(x_min, x_max)
    ax.axvline(0, color="#444444", linewidth=0.8, linestyle="--", alpha=0.55)
    ax.set_title(title, fontsize=10, fontweight="bold", loc="left", pad=3)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.tick_params(axis="both", length=0, labelsize=9)
    ax.grid(axis="x", alpha=0.2, linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("#cccccc")


def figure2(outcome: str = "CHECKUP") -> Path:
    d = _build_data(outcome)
    title = OUTCOME_LABELS.get(outcome, outcome.replace("_", " ").title())

    n_idx = len(d["idx_labels"])
    n_emb = len(d["emb_labels"])

    # Shared x range across all panels
    all_vals = d["idx_r2"] + d["emb_r2"] + d["comb_r2"]
    finite   = [v for v in all_vals if not np.isnan(v)]
    x_min = min(min(finite), 0.0) - 0.02
    x_max = max(finite) * 1.08

    # Figure height: proportional to total bar rows, with inter-panel gaps
    row_h = 0.42
    gap_h = 0.40
    fig_h = 0.55 + n_idx * row_h + gap_h + n_emb * row_h + gap_h + n_emb * row_h + 0.65

    fig, axes = plt.subplots(
        3, 1, figsize=(7.5, fig_h),
        gridspec_kw={"height_ratios": [n_idx, n_emb, n_emb], "hspace": 0.42},
    )

    no_mock = [False] * n_idx
    _draw_panel(axes[0], d["idx_labels"], d["idx_r2"],  no_mock,     PALETTE_INDICES,  "Indices Alone",                    x_min, x_max)
    _draw_panel(axes[1], d["emb_labels"], d["emb_r2"],  d["mocked"], PALETTE_EMB,      "Embeddings Alone",                 x_min, x_max)
    _draw_panel(axes[2], d["emb_labels"], d["comb_r2"], d["mocked"], PALETTE_COMBINED, "Embeddings With Combined Indices", x_min, x_max)

    axes[2].set_xlabel("R²  (held-out states)", fontsize=9)
    title_obj = fig.suptitle(title, fontsize=12, fontweight="bold", y=1.00)

    # Draw underline beneath the suptitle using its rendered bounding box
    from matplotlib.lines import Line2D
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = title_obj.get_window_extent(renderer=renderer)
    inv  = fig.transFigure.inverted()
    x0, y0 = inv.transform((bbox.x0, bbox.y0))
    x1, _  = inv.transform((bbox.x1, bbox.y0))
    fig.add_artist(Line2D([x0, x1], [y0 - 0.008, y0 - 0.008],
                          transform=fig.transFigure, color="black", linewidth=0.9))

    if any(d["mocked"]):
        fig.text(
            0.02, 0.0,
            "* placeholder — embedding run not yet complete",
            fontsize=7.5, fontstyle="italic", color="#888888",
            transform=fig.transFigure, va="bottom",
        )

    out = FIG_DIR / f"figure2_{outcome}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Render Figure 2 for one PLACES outcome.")
    ap.add_argument("--outcome", default="CHECKUP",
                    help="PLACES outcome code (default: CHECKUP)")
    args = ap.parse_args()
    figure2(args.outcome)
