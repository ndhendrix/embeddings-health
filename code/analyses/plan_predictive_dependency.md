# Exploratory Predictive Dependency Analysis — Plan

## Goal

Map the structure of shared predictive signal across ACS variables in the AlphaEarth embedding space. The core question: when the embedding predicts % minority or income, how much of that predictive signal is shared with its prediction of housing structure variables? The output is a *predictive dependency graph* — not a causal DAG, but the raw material for constructing one with collaborators.

---

## Conceptual framing

We already have out-of-fold R² values for each ACS variable (from the existing analysis). The new quantity we need is: **cross-prediction R²** — how well does the embedding's prediction of variable A explain the actual values of variable B?

Call Ŷ_X the embedding model's out-of-fold prediction for ACS variable X. We want to know:

- **Shared channel**: R²(Y_actual ~ Ŷ_M) — how much of the actual variance in Y is captured by M's embedding channel alone
- **Direct channel**: what the embedding explains in Y *after* M's channel is removed
- **Proportion mediated**: (Total R² − Direct R²) / Total R²

This proportion is the weight on the arrow M → Y in the predictive dependency graph.

---

## Variable set

Focus on 20 variables organized into four tiers, ordered by approximate directness of visual inference:

**Tier 1 — Built form (directly observable)**
- `heating_fuel_utility_gas_pct` (R²=0.56) — strong urban/rural signal
- `mobile_homes_pct` (R²=0.40)
- `multi_unit_housing_pct` (R²=0.38)
- `structures_20plus_units_pct` (R²=0.34)
- `renter_occupied_pct` (R²=0.40)

**Tier 2 — Access and resources (inferred from density/form)**
- `no_vehicle_pct` (R²=0.26)
- `median_gross_rent` (R²=0.31)
- `median_owner_cost_mortgage` (R²=0.36)
- `median_home_value` (R²=0.24)
- `housing_cost_burdened_pct` (R²=0.27)

**Tier 3 — Compositional/demographic (socially inferred)**
- `minority_pct` (R²=0.38)
- `lep_total_pct` (R²=0.16)
- `single_parent_pct` (R²=0.10)
- `age_65plus_pct` (R²=0.11)

**Tier 4 — Socioeconomic outcomes (most distal)**
- `median_family_income` (R²=0.30)
- `per_capita_income` (R²=0.27)
- `pov_total_pct` (R²=0.16)
- `no_hs_diploma_pct` (R²=0.18)
- `unemployment_pct` (R²=0.05)
- `uninsured_pct` (R²=0.13)

---

## Analytical steps

### Step 1: Generate out-of-fold predictions for all 20 variables

Refit the existing GroupKFold (by state) ridge regression pipeline, but this time save the out-of-fold predictions Ŷ_X for every target variable simultaneously. (Currently the analysis only saves R²; we need the prediction vectors.)

Output: a `(n_tracts × 20)` matrix of out-of-fold predictions — one column per variable.

### Step 2: Compute the cross-prediction matrix

For every pair (X, Y) of the 20 variables, regress Y_actual on Ŷ_X and record R². This is a 20×20 matrix where entry [X, Y] answers: "how much of Y's actual variance is explained by the embedding's learned representation of X?"

The diagonal is the standard out-of-fold R² already reported. Off-diagonal entries are new.

Output: `cross_prediction_matrix.csv`

### Step 3: Mediation decomposition for focal pairs

For each pair where the off-diagonal entry is large (suggesting a candidate mediation pathway):

1. R²_total = diagonal entry for Y (embedding → Y directly)
2. R²_via_M = off-diagonal entry [M, Y] (embedding's M-channel → Y)
3. R²_direct = regress Y_actual on (embeddings, Ŷ_M together) → residual R² from embedding after partialing M

Proportion mediated by M = (R²_total − R²_direct) / R²_total

In practice, step 3 fits: `Y_actual ~ Ŷ_M + E` where E are the raw embeddings, and computes how much the embedding coefficients shrink when Ŷ_M is included. Since Ŷ_M is itself a linear function of E, we use partial R² (the increment from adding raw embeddings after Ŷ_M).

This will be computed for theoretically motivated pairs first:
- `renter_occupied_pct` → `minority_pct`
- `multi_unit_housing_pct` → `minority_pct`
- `heating_fuel_utility_gas_pct` → `median_family_income`
- `median_gross_rent` → `pov_total_pct`
- `no_vehicle_pct` → `unemployment_pct`

### Step 4: Factor analysis of prediction vectors

Run PCA on the (n_tracts × 20) prediction matrix from Step 1. Each principal component is a latent "embedding channel" — a pattern in the embedding space that simultaneously predicts multiple outcomes. Plot the variable loadings on the first 3-4 components.

This step is a sanity check: if the tier structure (built form → access → demographic → socioeconomic) is real, we expect variables within tiers to load on the same components.

---

## Variable labels

Raw variable names are replaced with reader-friendly labels throughout all outputs:

| Variable | Label |
|---|---|
| `heating_fuel_utility_gas_pct` | Utility gas heating (%) |
| `mobile_homes_pct` | Mobile homes (%) |
| `multi_unit_housing_pct` | Multi-unit housing (%) |
| `structures_20plus_units_pct` | Large apt. buildings (%) |
| `renter_occupied_pct` | Renter-occupied (%) |
| `no_vehicle_pct` | No vehicle (%) |
| `median_gross_rent` | Median gross rent |
| `median_owner_cost_mortgage` | Median owner costs |
| `median_home_value` | Median home value |
| `housing_cost_burdened_pct` | Cost-burdened housing (%) |
| `minority_pct` | Minority race/ethnicity (%) |
| `lep_total_pct` | Limited English proficiency (%) |
| `single_parent_pct` | Single-parent households (%) |
| `age_65plus_pct` | Age ≥65 (%) |
| `median_family_income` | Median family income |
| `per_capita_income` | Per capita income |
| `pov_total_pct` | Below poverty level (%) |
| `no_hs_diploma_pct` | No high school diploma (%) |
| `unemployment_pct` | Unemployment rate (%) |
| `uninsured_pct` | Uninsured (%) |

---

## Outputs

1. **Cross-prediction heatmap** — 20×20 symmetric heatmap, hierarchically clustered on both axes (Ward linkage), color-scaled by R². Both axes use the friendly labels above, with variables grouped visually by tier via a color-coded sidebar. Diagonal entries (standard embedding R²) marked distinctly. This is the primary visualization.

2. **PCA biplot** — loadings of the 20 variables on PC1 vs. PC2 of the prediction matrix, with tier as color and friendly labels as annotations. Shows the latent structure visually.

---

## Framing for the paper

All results will be described as "predictive dependency structure" — not causal pathways. The language throughout will be: "the embedding's predictive signal for X shares Y% of its variance with its signal for M" rather than "X mediates the effect of the embedding on M." The path diagram will be explicitly labeled a "candidate causal graph for future confirmatory analysis."

---

## Implementation notes

- Language: Python (consistent with existing pipeline)
- Use the same GroupKFold(n_splits=5, shuffle=False) split by state as the existing analysis
- Ridge regression with the existing alpha selection procedure
- No additional packages required beyond the existing environment
- All outputs to `outputs/`, notebook in `code/analyses/`
- Estimated runtime: ~10–15 minutes for Step 1 (fitting 20 models); Steps 2–4 are fast
