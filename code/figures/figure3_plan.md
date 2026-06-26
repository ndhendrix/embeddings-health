# Figure 3 Plan

## Goal

A grouped horizontal bar chart showing the absolute incremental R² added by each
embedding model beyond what a social risk index alone explains, broken out by
PLACES health outcome. The full figure has one panel per index (ReADI, SDI, SVI);
the first build covers one panel.

---

## Layout (single-index panel)

```
┌──────────────────────────────────────────────┐
│  ReADI — Incremental predictive value         │
│                                               │
│  Annual checkup        ██ ████ ███            │
│  Fair/poor gen. health ███ ████ ██            │
│  ...                                          │
│  (40 outcomes, sorted descending by mean Δ R²)│
│                                               │
│  x-axis: ΔR² (additional variance explained) │
│  color  : one per embedding model             │
└──────────────────────────────────────────────┘
```

Each outcome row contains 3 bars (one per model currently available; 3 mocked
placeholders shown with lighter fills and asterisk labels until OlmoEarth Nano,
OlmoEarth Base, and Clay v1.5 are ready). Groups are separated by a small gap.

---

## Data

| Column | Meaning |
|---|---|
| `additional_var` | ΔR² = R²(index + embeddings) − R²(index alone) |
| `index` | Social risk index (ReADI / SDI / SVI) |
| `outcome` | PLACES health outcome code |

**Sources**
| Model label | File |
|---|---|
| AlphaEarth | `outputs/alphaearth_foundations/places_residual_by_index.csv` |
| Prithvi Tiny-TL | `outputs/prithvi_tiny/prithvi_tiny_places_residual_by_index.csv` |
| Prithvi 300M-TL | `outputs/prithvi_300M-TL/places_residual_by_index.csv` |
| OlmoEarth-1.1 Nano | not yet available — mocked |
| OlmoEarth-1.1 Base | not yet available — mocked |
| Clay v1.5 | not yet available — mocked |

---

## Visual design

- **Sort**: outcomes sorted descending by mean ΔR² across the real models, so the
  most "embeddable" outcomes appear at the top.
- **Colors**: qualitative palette (Okabe-Ito, colorblind-safe) — one fixed color
  per model across all panels so the legend carries over.
- **Mocked bars**: same color but 35% lighter, label gets a `*` suffix.
- **Bar height**: 0.12 per bar, 0.06 gap within a group, 0.18 gap between outcome
  groups — produces a legible figure at ~13 inches tall for 40 outcomes × 6 models.
- **Reference line**: vertical dashed line at ΔR² = 0.
- **Legend**: placed outside the plot area at top-right.
- **X-axis label**: "Additional R² beyond index alone".

---

## Output

- Script: `code/figures/figure3.py`  
  CLI: `python figure3.py --index ReADI` (default ReADI)
- Output: `outputs/figures/figure3_ReADI.png` (300 dpi)
- When the full figure is assembled later, each index panel will be a subfigure
  stacked vertically (or laid out in columns) inside one multi-panel figure.

---

## Extensibility

Adding a fourth real model requires: (1) add an entry to `MODEL_SOURCES` pointing
at its `places_residual_by_index.csv`; (2) run the script again. No structural
changes needed.
