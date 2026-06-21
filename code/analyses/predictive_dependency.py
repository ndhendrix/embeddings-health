#!/usr/bin/env python3
"""
Predictive dependency analysis for ACS variables.

Steps:
  1. Load & prep data (same pipeline as analyses.ipynb)
  2. For each of 20 target variables, fit LightGBM via GroupKFold and
     save out-of-fold (OOF) predictions.
  3. Build 20×20 cross-prediction R² matrix.
  4. Plot hierarchically-clustered heatmap with tier sidebar.
  5. Plot PCA biplot of prediction vectors, colored by tier.

Run from project root:
    uv run python code/analyses/predictive_dependency.py
"""

import argparse
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from adjustText import adjust_text
from scipy.cluster.hierarchy import linkage
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("data")
OUTPUTS_DIR = Path("outputs")

USE_CACHED = True   # set False to refit models from scratch

# ── Embedding feature columns ──────────────────────────────────────────────────
# EMB_COLS and FEATURE_COLS are populated at runtime by load_data() so the
# script works with any embedding source (AlphaEarth, Prithvi tiny, etc.).
STAT_SUFFIXES = ["MEAN", "MEDIAN", "MINIMUM", "MAXIMUM", "STD"]
EMB_COLS     = []  # set by load_data
FEATURE_COLS = []  # set by load_data

# ── Target variables and metadata ─────────────────────────────────────────────
TARGET_VARS = [
    "heating_fuel_utility_gas_pct",
    "mobile_homes_pct",
    "multi_unit_housing_pct",
    "structures_20plus_units_pct",
    "renter_occupied_pct",
    "no_vehicle_pct",
    "median_gross_rent",
    "median_owner_cost_mortgage",
    "median_home_value",
    "housing_cost_burdened_pct",
    "minority_pct",
    "lep_total_pct",
    "single_parent_pct",
    "age_65plus_pct",
    "median_family_income",
    "per_capita_income",
    "pov_total_pct",
    "no_hs_diploma_pct",
    "unemployment_pct",
    "uninsured_pct",
]

FRIENDLY_LABELS = {
    "heating_fuel_utility_gas_pct": "Utility gas heating (%)",
    "mobile_homes_pct":             "Mobile homes (%)",
    "multi_unit_housing_pct":       "Multi-unit housing (%)",
    "structures_20plus_units_pct":  "Large apt. buildings (%)",
    "renter_occupied_pct":          "Renter-occupied (%)",
    "no_vehicle_pct":               "No vehicle (%)",
    "median_gross_rent":            "Median gross rent",
    "median_owner_cost_mortgage":   "Median owner costs",
    "median_home_value":            "Median home value",
    "housing_cost_burdened_pct":    "Cost-burdened housing (%)",
    "minority_pct":                 "Minority race/ethnicity (%)",
    "lep_total_pct":                "Limited English proficiency (%)",
    "single_parent_pct":            "Single-parent households (%)",
    "age_65plus_pct":               "Age ≥65 (%)",
    "median_family_income":         "Median family income",
    "per_capita_income":            "Per capita income",
    "pov_total_pct":                "Below poverty level (%)",
    "no_hs_diploma_pct":            "No high school diploma (%)",
    "unemployment_pct":             "Unemployment rate (%)",
    "uninsured_pct":                "Uninsured (%)",
}

TIERS = {
    "heating_fuel_utility_gas_pct": 1,
    "mobile_homes_pct":             1,
    "multi_unit_housing_pct":       1,
    "structures_20plus_units_pct":  1,
    "renter_occupied_pct":          1,
    "no_vehicle_pct":               2,
    "median_gross_rent":            2,
    "median_owner_cost_mortgage":   2,
    "median_home_value":            2,
    "housing_cost_burdened_pct":    2,
    "minority_pct":                 3,
    "lep_total_pct":                3,
    "single_parent_pct":            3,
    "age_65plus_pct":               3,
    "median_family_income":         4,
    "per_capita_income":            4,
    "pov_total_pct":                4,
    "no_hs_diploma_pct":            4,
    "unemployment_pct":             4,
    "uninsured_pct":                4,
}

TIER_COLORS = {1: "#1565C0", 2: "#2E7D32", 3: "#E65100", 4: "#6A1B9A"}
TIER_NAMES  = {
    1: "Built form",
    2: "Access & resources",
    3: "Compositional/demographic",
    4: "Socioeconomic outcomes",
}

LGBM_KW = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
               min_child_samples=20, random_state=42, verbosity=-1)


# ── Step 0: Load and prepare data ─────────────────────────────────────────────

def load_data(
    embeddings_path: Path = DATA_DIR / "alphaearth_embeddings.csv",
) -> pd.DataFrame:
    """Return a pandas DataFrame with FEATURE_COLS + TARGET_VARS + 'state' column."""
    global EMB_COLS, FEATURE_COLS

    # Embeddings (2022 only, drop tracts with null channel data)
    _raw = pl.read_csv(
        embeddings_path,
        infer_schema_length=10_000,
        schema_overrides={"GEOID": pl.Utf8},
    )

    # Detect embedding columns: any *_STAT column that isn't geographic metadata
    _meta = {"GEOID", "year", "ALAND", "AWATER", "tract_fips", "NAME"}
    _stat_set = set(STAT_SUFFIXES)
    _null_col = next(
        (c for c in _raw.columns
         if any(c.endswith(f"_{s}") for s in _stat_set) and c not in _meta),
        None,
    )

    emb = (
        _raw
        .filter(pl.col("year") == 2022)
        .pipe(lambda df: df.filter(pl.col(_null_col).is_not_null()) if _null_col else df)
        .with_columns(pl.col("GEOID").str.zfill(11).alias("tract_fips"))
    )

    EMB_COLS = [
        c for c in emb.columns
        if any(c.endswith(f"_{s}") for s in _stat_set) and c not in _meta
    ]
    FEATURE_COLS = EMB_COLS + [c for c in ["ALAND", "AWATER"] if c in emb.columns]

    emb = emb.select(["tract_fips"] + FEATURE_COLS)
    print(f"Embeddings: {len(emb):,} tracts, {len(EMB_COLS)} embedding features")

    # ACS — rename raw Census variable codes to human names
    ACS_RENAME = {
        "B17021_002E": "pov_below_related_children",
        "B17021_003E": "pov_below_other",
        "B17021_001E": "pov_universe",
        "B19113_001E": "median_family_income",
        "B19301_001E": "per_capita_income",
        "B19001_002E": "hh_income_lt10k",
        "B19001_017E": "hh_income_ge200k",
        "DP03_0009PE": "unemployment_pct",
        "B15003_001E": "educ_universe",
        "B15003_002E": "educ_no_schooling",
        "B15003_003E": "educ_nursery",
        "B15003_004E": "educ_kindergarten",
        "B15003_005E": "educ_grade1",
        "B15003_006E": "educ_grade2",
        "B15003_007E": "educ_grade3",
        "B15003_008E": "educ_grade4",
        "B15003_009E": "educ_grade5",
        "B15003_010E": "educ_grade6",
        "B15003_011E": "educ_grade7",
        "B15003_012E": "educ_grade8",
        "B15003_013E": "educ_grade9",
        "B15003_014E": "educ_grade10",
        "B15003_015E": "educ_grade11",
        "B15003_016E": "educ_grade12_no_diploma",
        "S0601_C01_033E": "no_hs_diploma_pct",
        "B16005_001E": "lang_universe",
        "B16005_007E": "lep_spanish_native_not_well",
        "B16005_008E": "lep_spanish_native_not_at_all",
        "B16005_012E": "lep_spanish_foreign_not_well",
        "B16005_013E": "lep_spanish_foreign_not_at_all",
        "B16005_017E": "lep_indo_euro_native_not_well",
        "B16005_018E": "lep_indo_euro_native_not_at_all",
        "B16005_022E": "lep_indo_euro_foreign_not_well",
        "B16005_023E": "lep_indo_euro_foreign_not_at_all",
        "B16005_029E": "lep_asian_pac_native_not_well",
        "B16005_030E": "lep_asian_pac_native_not_at_all",
        "B16005_034E": "lep_asian_pac_foreign_not_well",
        "B16005_035E": "lep_asian_pac_foreign_not_at_all",
        "B16005_039E": "lep_other_native_not_well",
        "B16005_040E": "lep_other_native_not_at_all",
        "B16005_044E": "lep_other_foreign_not_well",
        "B16005_045E": "lep_other_foreign_not_at_all",
        "B25064_001E": "median_gross_rent",
        "B25077_001E": "median_home_value",
        "B25003_001E": "tenure_universe",
        "B25003_002E": "owner_occupied_units",
        "B25040_001E": "heating_fuel_universe",
        "B25040_003E": "heating_fuel_utility_gas",
        "B25044_001E": "vehicle_availability_universe",
        "B25044_003E": "owner_occupied_no_vehicle",
        "B25044_010E": "renter_occupied_no_vehicle",
        "S2503_C01_001E": "renter_units_universe",
        "S2503_C01_028E": "rent_30_34pct_of_income",
        "S2503_C01_032E": "rent_35_39pct_of_income",
        "S2503_C01_036E": "rent_40_49pct_of_income",
        "S2503_C01_040E": "rent_50plus_pct_of_income",
        "DP05_0001E": "total_pop",
        "DP05_0079E": "nh_white_alone",
        "DP02_0001E": "total_households",
        "DP02_0007PE": "single_parent_male_hh_pct",
        "DP02_0011PE": "single_parent_female_hh_pct",
        "DP04_0012PE": "structures_10_19_units_pct",
        "DP04_0013PE": "structures_20plus_units_pct",
        "DP04_0014PE": "mobile_homes_pct",
        "DP04_0047PE": "renter_occupied_pct",
        "DP04_0058PE": "no_vehicle_pct",
        "DP04_0078PE": "overcrowded_owner_pct",
        "DP04_0079PE": "overcrowded_renter_pct",
        "S0101_C02_030E": "age_65plus_pct",
        "S2701_C05_001E": "uninsured_pct",
    }

    _acs_raw = (
        pl.read_csv(DATA_DIR / "acs.csv", infer_schema_length=5_000)
        .with_columns(
            pl.col("GEOID").cast(pl.Utf8).str.zfill(11).alias("tract_fips")
        )
        .drop(["GEOID", "NAME"])
    )
    acs = _acs_raw.rename({k: v for k, v in ACS_RENAME.items() if k in _acs_raw.columns})

    # Compute percentage conversions
    _available = set(acs.columns)
    _CONVERSIONS = [
        ("pov_below_related_children", "pov_universe",                "pov_below_related_children_pct"),
        ("pov_below_other",            "pov_universe",                "pov_below_other_pct"),
        ("hh_income_lt10k",            "total_households",            "hh_income_lt10k_pct"),
        ("hh_income_ge200k",           "total_households",            "hh_income_ge200k_pct"),
        ("heating_fuel_utility_gas",   "heating_fuel_universe",       "heating_fuel_utility_gas_pct"),
        ("owner_occupied_units",       "tenure_universe",             "owner_occupied_pct"),
        ("owner_occupied_no_vehicle",  "vehicle_availability_universe","owner_occupied_no_vehicle_pct"),
        ("renter_occupied_no_vehicle", "vehicle_availability_universe","renter_occupied_no_vehicle_pct"),
        ("rent_30_34pct_of_income",    "renter_units_universe",       "cost_burdened_rent_30_34_pct"),
        ("rent_35_39pct_of_income",    "renter_units_universe",       "cost_burdened_rent_35_39_pct"),
        ("rent_40_49pct_of_income",    "renter_units_universe",       "cost_burdened_rent_40_49_pct"),
        ("rent_50plus_pct_of_income",  "renter_units_universe",       "cost_burdened_rent_50plus_pct"),
        ("nh_white_alone",             "total_pop",                   "nh_white_pct"),
        ("lep_spanish_native_not_well",     "lang_universe", "lep_spanish_native_not_well_pct"),
        ("lep_spanish_native_not_at_all",   "lang_universe", "lep_spanish_native_not_at_all_pct"),
        ("lep_spanish_foreign_not_well",    "lang_universe", "lep_spanish_foreign_not_well_pct"),
        ("lep_spanish_foreign_not_at_all",  "lang_universe", "lep_spanish_foreign_not_at_all_pct"),
        ("lep_indo_euro_native_not_well",   "lang_universe", "lep_indo_euro_native_not_well_pct"),
        ("lep_indo_euro_native_not_at_all", "lang_universe", "lep_indo_euro_native_not_at_all_pct"),
        ("lep_indo_euro_foreign_not_well",  "lang_universe", "lep_indo_euro_foreign_not_well_pct"),
        ("lep_indo_euro_foreign_not_at_all","lang_universe", "lep_indo_euro_foreign_not_at_all_pct"),
        ("lep_asian_pac_native_not_well",   "lang_universe", "lep_asian_pac_native_not_well_pct"),
        ("lep_asian_pac_native_not_at_all", "lang_universe", "lep_asian_pac_native_not_at_all_pct"),
        ("lep_asian_pac_foreign_not_well",  "lang_universe", "lep_asian_pac_foreign_not_well_pct"),
        ("lep_asian_pac_foreign_not_at_all","lang_universe", "lep_asian_pac_foreign_not_at_all_pct"),
        ("lep_other_native_not_well",       "lang_universe", "lep_other_native_not_well_pct"),
        ("lep_other_native_not_at_all",     "lang_universe", "lep_other_native_not_at_all_pct"),
        ("lep_other_foreign_not_well",      "lang_universe", "lep_other_foreign_not_well_pct"),
        ("lep_other_foreign_not_at_all",    "lang_universe", "lep_other_foreign_not_at_all_pct"),
        ("educ_no_schooling",    "educ_universe", "educ_no_schooling_pct"),
        ("educ_nursery",         "educ_universe", "educ_nursery_pct"),
        ("educ_kindergarten",    "educ_universe", "educ_kindergarten_pct"),
        ("educ_grade1",          "educ_universe", "educ_grade1_pct"),
        ("educ_grade2",          "educ_universe", "educ_grade2_pct"),
        ("educ_grade3",          "educ_universe", "educ_grade3_pct"),
        ("educ_grade4",          "educ_universe", "educ_grade4_pct"),
        ("educ_grade5",          "educ_universe", "educ_grade5_pct"),
        ("educ_grade6",          "educ_universe", "educ_grade6_pct"),
        ("educ_grade7",          "educ_universe", "educ_grade7_pct"),
        ("educ_grade8",          "educ_universe", "educ_grade8_pct"),
        ("educ_grade9",          "educ_universe", "educ_grade9_pct"),
        ("educ_grade10",         "educ_universe", "educ_grade10_pct"),
        ("educ_grade11",         "educ_universe", "educ_grade11_pct"),
        ("educ_grade12_no_diploma","educ_universe","educ_grade12_no_diploma_pct"),
    ]
    _valid = [(n, d, name) for n, d, name in _CONVERSIONS if n in _available and d in _available]
    acs = acs.with_columns([
        (pl.col(num) / pl.col(den) * 100).alias(name)
        for num, den, name in _valid
    ])

    # Remove ACS jam codes (sentinel values < -99999)
    _num_cols = [c for c in acs.columns if c != "tract_fips"]
    acs = acs.with_columns([
        pl.when(pl.col(c) < -99999).then(None).otherwise(pl.col(c)).alias(c)
        for c in _num_cols
    ])

    # Compute composite variables
    _lep_cols    = [c for c in acs.columns if c.startswith("lep_") and c.endswith("_pct")]
    _burden_cols = [
        "cost_burdened_rent_30_34_pct", "cost_burdened_rent_35_39_pct",
        "cost_burdened_rent_40_49_pct", "cost_burdened_rent_50plus_pct",
    ]
    _burden_cols_avail = [c for c in _burden_cols if c in acs.columns]
    _educ_lt9_cols = [
        "educ_no_schooling_pct", "educ_nursery_pct", "educ_kindergarten_pct",
        "educ_grade1_pct", "educ_grade2_pct", "educ_grade3_pct",
        "educ_grade4_pct", "educ_grade5_pct", "educ_grade6_pct", "educ_grade7_pct",
    ]
    _educ_lt9_avail = [c for c in _educ_lt9_cols if c in acs.columns]

    composite_exprs = []
    if _lep_cols:
        composite_exprs.append(
            pl.sum_horizontal([pl.col(c) for c in _lep_cols]).alias("lep_total_pct")
        )
    if "single_parent_male_hh_pct" in acs.columns and "single_parent_female_hh_pct" in acs.columns:
        composite_exprs.append(
            (pl.col("single_parent_male_hh_pct") + pl.col("single_parent_female_hh_pct")).alias("single_parent_pct")
        )
    if "structures_10_19_units_pct" in acs.columns and "structures_20plus_units_pct" in acs.columns:
        composite_exprs.append(
            (pl.col("structures_10_19_units_pct") + pl.col("structures_20plus_units_pct")).alias("multi_unit_housing_pct")
        )
    if _burden_cols_avail:
        composite_exprs.append(
            pl.sum_horizontal([pl.col(c) for c in _burden_cols_avail]).alias("housing_cost_burdened_pct")
        )
    if "nh_white_pct" in acs.columns:
        composite_exprs.append(
            (100.0 - pl.col("nh_white_pct")).alias("minority_pct")
        )
    if "pov_below_related_children_pct" in acs.columns and "pov_below_other_pct" in acs.columns:
        composite_exprs.append(
            (pl.col("pov_below_related_children_pct") + pl.col("pov_below_other_pct")).alias("pov_total_pct")
        )

    if composite_exprs:
        acs = acs.with_columns(composite_exprs)

    # Join ReADI for median_owner_cost_mortgage
    readi = (
        pl.read_csv(DATA_DIR / "ReADI_CT_2022.csv", infer_schema_length=5_000)
        .select(["GEOID", "MEDMORT"])
        .with_columns(pl.col("GEOID").cast(pl.Utf8).str.zfill(11).alias("tract_fips"))
        .drop("GEOID")
        .rename({"MEDMORT": "median_owner_cost_mortgage"})
    )
    acs = acs.join(readi, on="tract_fips", how="left")

    # Join embeddings + ACS
    df = emb.join(acs.select(["tract_fips"] + TARGET_VARS), on="tract_fips", how="inner")
    df = df.with_columns(
        pl.col("tract_fips").str.slice(0, 2).alias("state")
    )
    print(f"Merged dataset: {len(df):,} tracts × {df.shape[1]} columns")

    missing = [v for v in TARGET_VARS if v not in df.columns]
    if missing:
        raise ValueError(f"Missing target variables: {missing}")

    return df.to_pandas()


# ── Step 1: Out-of-fold predictions ───────────────────────────────────────────

def get_oof_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame of OOF predictions, indexed by df.index, one col per target."""

    cache_path = OUTPUTS_DIR / "oof_predictions.csv"
    if USE_CACHED and cache_path.exists():
        print("Loading cached OOF predictions...")
        return pd.read_csv(cache_path, index_col=0)

    X_all    = df[FEATURE_COLS].to_numpy(dtype=float)
    states   = df["state"].to_numpy()
    kf       = GroupKFold(n_splits=5)
    oof      = pd.DataFrame(index=df.index, columns=TARGET_VARS, dtype=float)

    for var in TARGET_VARS:
        y_series = df[var]
        valid    = y_series.notna().to_numpy()
        y        = y_series[valid].to_numpy(dtype=float)
        X        = X_all[valid]
        grps     = states[valid]
        idx_valid = np.where(valid)[0]

        pred = np.full(valid.sum(), np.nan)
        for train_idx, val_idx in kf.split(X, y, groups=grps):
            model = lgb.LGBMRegressor(**LGBM_KW)
            model.fit(X[train_idx], y[train_idx])
            pred[val_idx] = model.predict(X[val_idx])

        oof.loc[idx_valid, var] = pred
        r2 = r2_score(y, pred)
        print(f"  {var:<45s}  R²={r2:.3f}  n={valid.sum():,}")

    oof.to_csv(cache_path)
    print(f"OOF predictions saved → {cache_path}")
    return oof


# ── Step 2: Cross-prediction matrix ───────────────────────────────────────────

def cross_prediction_matrix(df: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    """
    For each pair (predictor, outcome), compute R²(Ŷ_outcome ~ Ŷ_predictor).

    Answers: "What fraction of the embedding's outcome-prediction is reproduced
    by its predictor-prediction?" Diagonal = 1.0 by construction.
    """
    cache_path = OUTPUTS_DIR / "cross_prediction_matrix.csv"
    if USE_CACHED and cache_path.exists():
        print("Loading cached cross-prediction matrix...")
        return pd.read_csv(cache_path, index_col=0)

    mat = pd.DataFrame(np.nan, index=TARGET_VARS, columns=TARGET_VARS)

    for predictor in TARGET_VARS:
        yhat_pred = oof[predictor].to_numpy(dtype=float)
        for outcome in TARGET_VARS:
            if predictor == outcome:
                mat.loc[predictor, outcome] = 1.0
                continue
            yhat_out = oof[outcome].to_numpy(dtype=float)
            both_valid = ~np.isnan(yhat_pred) & ~np.isnan(yhat_out)
            if both_valid.sum() < 50:
                continue
            # OLS fit (intercept + slope) for unit invariance;
            # result equals cor(Ŷ_predictor, Ŷ_outcome)² ∈ [0, 1]
            A = np.column_stack([np.ones(both_valid.sum()), yhat_pred[both_valid]])
            coef, _, _, _ = np.linalg.lstsq(A, yhat_out[both_valid], rcond=None)
            mat.loc[predictor, outcome] = r2_score(
                yhat_out[both_valid], A @ coef
            )

    mat.to_csv(cache_path)
    print(f"Cross-prediction matrix saved → {cache_path}")
    return mat


# ── Step 3: Mediation decomposition for focal pairs ───────────────────────────

FOCAL_PAIRS = [
    ("renter_occupied_pct",           "minority_pct"),
    ("multi_unit_housing_pct",        "minority_pct"),
    ("heating_fuel_utility_gas_pct",  "median_family_income"),
    ("median_gross_rent",             "pov_total_pct"),
    ("no_vehicle_pct",                "unemployment_pct"),
]


def mediation_decomposition(
    df: pd.DataFrame, oof: pd.DataFrame, mat: pd.DataFrame
) -> pd.DataFrame:
    """
    For each focal (mediator, outcome) pair compute:
      r2_total   — diagonal of cross-prediction matrix for outcome
      r2_via_m   — off-diagonal entry [mediator, outcome]
      r2_direct  — variance in outcome explained by embeddings beyond mediator channel
      prop_mediated — (r2_total - r2_direct) / r2_total

    r2_direct is estimated by:
      1. Residualise outcome on oof_mediator (OLS, 1 predictor).
      2. Fit LightGBM on embeddings → residuals via GroupKFold.
      3. Scale back to outcome variance: r2_direct = r2_resid * (1 - r2_via_m).
    """
    cache_path = OUTPUTS_DIR / "mediation_decomposition.csv"
    if USE_CACHED and cache_path.exists():
        print("Loading cached mediation decomposition...")
        return pd.read_csv(cache_path)

    X_all  = df[FEATURE_COLS].to_numpy(dtype=float)
    states = df["state"].to_numpy()
    kf     = GroupKFold(n_splits=5)
    rows   = []

    for mediator, outcome in FOCAL_PAIRS:
        yhat_m   = oof[mediator].to_numpy(dtype=float)
        yhat_out = oof[outcome].to_numpy(dtype=float)
        y_actual = df[outcome].to_numpy(dtype=float)
        both     = ~np.isnan(yhat_m) & ~np.isnan(y_actual)

        # r2_total: OOF R² — how well the model predicts outcome vs actual values
        valid_out = ~np.isnan(yhat_out) & ~np.isnan(y_actual)
        r2_total  = r2_score(y_actual[valid_out], yhat_out[valid_out])

        # Residualise outcome on oof_mediator (OLS intercept + slope)
        yhat_m_b = yhat_m[both]
        y_b      = y_actual[both]
        A        = np.column_stack([np.ones(both.sum()), yhat_m_b])
        coef, _, _, _ = np.linalg.lstsq(A, y_b, rcond=None)
        residuals_b  = y_b - A @ coef

        # r2_via_m: OLS R² of Y_actual on Ŷ_mediator
        r2_via_m = r2_score(y_b, A @ coef)

        # LightGBM: embeddings → residuals (GroupKFold)
        X_b      = X_all[both]
        states_b = states[both]
        pred_res = np.full(both.sum(), np.nan)
        for tr, va in kf.split(X_b, residuals_b, groups=states_b):
            m = lgb.LGBMRegressor(**LGBM_KW)
            m.fit(X_b[tr], residuals_b[tr])
            pred_res[va] = m.predict(X_b[va])

        r2_resid  = max(0.0, r2_score(residuals_b, pred_res))
        r2_direct = r2_resid * max(0.0, 1.0 - r2_via_m)
        prop_med  = (r2_total - r2_direct) / r2_total if r2_total > 0 else np.nan

        rows.append({
            "mediator":        FRIENDLY_LABELS[mediator],
            "outcome":         FRIENDLY_LABELS[outcome],
            "r2_total":        round(r2_total, 3),
            "r2_via_mediator": round(r2_via_m, 3),
            "r2_direct":       round(r2_direct, 3),
            "prop_mediated":   round(prop_med, 3),
        })
        print(
            f"  {FRIENDLY_LABELS[mediator]:<35s} → {FRIENDLY_LABELS[outcome]:<35s}"
            f"  total={r2_total:.3f}  via_M={r2_via_m:.3f}"
            f"  direct={r2_direct:.3f}  mediated={prop_med:.1%}"
        )

    result = pd.DataFrame(rows)
    result.to_csv(cache_path, index=False)
    print(f"Mediation decomposition saved → {cache_path}")
    return result


# ── Step 4: Heatmap ───────────────────────────────────────────────────────────

def plot_heatmap(mat: pd.DataFrame) -> None:
    labels = [FRIENDLY_LABELS[v] for v in mat.index]
    mat_np = mat.to_numpy(dtype=float)

    # Hierarchical clustering (same linkage for rows and cols)
    Z = linkage(mat_np, method="ward", metric="euclidean")

    # Build row_colors as a Series so clustermap reorders it correctly
    row_color_series = pd.Series(
        [TIER_COLORS[TIERS[v]] for v in mat.index],
        index=mat.index,
        name="Tier",
    )

    g = sns.clustermap(
        pd.DataFrame(mat_np, index=mat.index, columns=mat.columns),
        row_linkage=Z,
        col_linkage=Z,
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        linewidths=0.3,
        linecolor="white",
        figsize=(15, 14),
        cbar_kws={"label": "Shared prediction R²  [R²(Ŷ_outcome ~ Ŷ_predictor)]", "shrink": 0.5},
        xticklabels=labels,
        yticklabels=labels,
        row_colors=row_color_series,
        col_colors=row_color_series,
    )

    g.ax_heatmap.set_xticklabels(
        g.ax_heatmap.get_xticklabels(), rotation=45, ha="right", fontsize=9
    )
    g.ax_heatmap.set_yticklabels(
        g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=9
    )

    # Outline diagonal cells so they read as distinct
    reordered = g.dendrogram_row.reordered_ind
    for display_pos, orig_idx in enumerate(reordered):
        g.ax_heatmap.add_patch(
            mpatches.Rectangle(
                (display_pos, display_pos),
                1, 1,
                fill=False, edgecolor="black", linewidth=1.5, zorder=5
            )
        )

    # Tier legend
    legend_handles = [
        mpatches.Patch(color=TIER_COLORS[t], label=TIER_NAMES[t])
        for t in sorted(TIER_COLORS)
    ]
    g.ax_heatmap.legend(
        handles=legend_handles,
        title="Tier",
        loc="upper left",
        bbox_to_anchor=(1.25, 1.02),
        fontsize=8,
        title_fontsize=9,
        frameon=True,
    )

    g.fig.suptitle(
        "Shared embedding signal across ACS variables\n"
        "R²(Ŷ_outcome ~ Ŷ_predictor): fraction of outcome prediction reproduced by predictor signal",
        y=1.01, fontsize=11, fontweight="bold",
    )

    out_path = OUTPUTS_DIR / "cross_prediction_heatmap.png"
    g.fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Heatmap saved → {out_path}")
    plt.close(g.fig)


# ── Step 4: PCA biplot ────────────────────────────────────────────────────────

def plot_pca_biplot(oof: pd.DataFrame) -> None:
    # Use rows where all 20 predictions are available
    valid_mask = oof.notna().all(axis=1)
    X = oof.loc[valid_mask, TARGET_VARS].to_numpy(dtype=float)
    print(f"PCA on {valid_mask.sum():,} tracts with complete OOF predictions")

    # Standardize each variable's predictions before PCA so that dollar-scale
    # variables (home value, income) don't dominate percentage variables.
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    pca.fit(X_scaled)
    loadings = pd.DataFrame(
        pca.components_.T,
        index=TARGET_VARS,
        columns=["PC1", "PC2"]
    )

    fig, ax = plt.subplots(figsize=(11, 9))

    texts = []
    for var, row in loadings.iterrows():
        tier  = TIERS[var]
        color = TIER_COLORS[tier]
        label = FRIENDLY_LABELS[var]
        ax.scatter(row["PC1"], row["PC2"], color=color, s=70, zorder=3,
                   edgecolors="white", linewidths=0.5)
        texts.append(ax.text(row["PC1"], row["PC2"], label,
                             fontsize=8, color=color))

    adjust_text(texts, ax=ax,
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
                expand=(1.2, 1.4))

    ax.axhline(0, color="lightgray", linewidth=0.8, zorder=0)
    ax.axvline(0, color="lightgray", linewidth=0.8, zorder=0)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)", fontsize=11)
    ax.set_title(
        "PCA of embedding prediction vectors\n(each point = one ACS variable's learned signal)",
        fontsize=11, fontweight="bold",
    )

    legend_handles = [
        mpatches.Patch(color=TIER_COLORS[t], label=TIER_NAMES[t])
        for t in sorted(TIER_COLORS)
    ]
    ax.legend(handles=legend_handles, title="Tier", fontsize=8,
              title_fontsize=9, loc="lower left")

    fig.tight_layout()
    out_path = OUTPUTS_DIR / "cross_prediction_pca.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"PCA biplot saved → {out_path}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _parser = argparse.ArgumentParser(
        description="Predictive dependency analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _parser.add_argument(
        "--embeddings",
        type=Path,
        default=DATA_DIR / "alphaearth_embeddings.csv",
        help="Path to combined embeddings CSV",
    )
    _parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for output files",
    )
    _args = _parser.parse_args()

    OUTPUTS_DIR = _args.outputs_dir
    OUTPUTS_DIR.mkdir(exist_ok=True)

    warnings.filterwarnings("ignore", message="X does not have valid feature names")

    print("=== Step 0: Loading data ===")
    df = load_data(_args.embeddings)

    print("\n=== Step 1: Out-of-fold predictions ===")
    oof = get_oof_predictions(df)

    print("\n=== Step 2: Cross-prediction matrix ===")
    mat = cross_prediction_matrix(df, oof)

    print("\n=== Step 3: Mediation decomposition ===")
    med = mediation_decomposition(df, oof, mat)
    print(med.to_string(index=False))

    print("\n=== Step 4: Heatmap ===")
    plot_heatmap(mat)

    print("\n=== Step 5: PCA biplot ===")
    plot_pca_biplot(oof)

    print("\nDone.")
