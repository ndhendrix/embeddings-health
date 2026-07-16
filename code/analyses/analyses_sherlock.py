#!/usr/bin/env python3
"""
Sherlock-adapted analysis runner for the embeddings-health project.

Usage (submit via sbatch or run interactively on a compute node):
    python code/analyses/analyses_sherlock.py --model alphaearth
    python code/analyses/analyses_sherlock.py --model prithvi_tiny
    python code/analyses/analyses_sherlock.py --model prithvi_300m_tl --no-cache

Reads embeddings and social-risk data from $SCRATCH/embeddings-health/.
ACS data lives at $REPO_DIR/data/acs.csv (auto-generated via get_data.py if absent).
Outputs are written to $SCRATCH/embeddings-health/outputs/<model>/.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must precede pyplot import

import numpy as np
import polars as pl
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import r2_score
import lightgbm as lgb
import matplotlib.pyplot as plt

_t0 = time.time()

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="embeddings-health analysis on Sherlock")
parser.add_argument(
    "--model",
    required=True,
    choices=["alphaearth", "prithvi_tiny", "prithvi_300m_tl", "clay"],
)
parser.add_argument(
    "--no-cache",
    action="store_true",
    help="Refit all LightGBM models from scratch (ignore cached CSVs)",
)
args = parser.parse_args()

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRATCH_ROOT = Path(os.environ["SCRATCH"]) / "embeddings-health"
REPO_DIR     = Path(__file__).parents[2]   # .../embeddings-health

SRI_DIR    = SCRATCH_ROOT / "social_risk_indices"
PLACES_CSV = (
    SCRATCH_ROOT
    / "PLACES__Local_Data_for_Better_Health__Census_Tract_Data__2025_release.csv"
)
ACS_CSV = REPO_DIR / "data" / "acs.csv"

_EMBEDDINGS_MAP = {
    "alphaearth":      SCRATCH_ROOT / "alphaearth" / "alphaearth_embeddings.csv",
    "prithvi_tiny":    SCRATCH_ROOT / "prithvi_aggregated" / "prithvi_tiny_2022_all_tracts.csv",
    "prithvi_300m_tl": SCRATCH_ROOT / "prithvi_aggregated" / "prithvi_300M-TL_2022_all_tracts.csv",
    "clay":            SCRATCH_ROOT / "clay_aggregated" / "clay_v1.5_2022_all_tracts.csv",
}
EMBEDDINGS_PATH = _EMBEDDINGS_MAP[args.model]
OUTPUTS_DIR     = SCRATCH_ROOT / "outputs" / args.model
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

USE_CACHED    = not args.no_cache
STAT_SUFFIXES = ["MEAN", "MEDIAN", "MINIMUM", "MAXIMUM", "STD"]

print(f"=== embeddings-health analysis ===")
print(f"Model      : {args.model}")
print(f"Embeddings : {EMBEDDINGS_PATH}")
print(f"Outputs    : {OUTPUTS_DIR}")
print(f"Use cached : {USE_CACHED}")
print()

# ── ACS data (auto-generate if absent) ────────────────────────────────────────
if not ACS_CSV.exists():
    print(f"{ACS_CSV} not found — running get_data.py (~5 min for all 50 states)...")
    ACS_CSV.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [sys.executable, str(REPO_DIR / "code" / "analyses" / "get_data.py")],
        capture_output=True,
        text=True,
        cwd=str(REPO_DIR),
    )
    print(result.stdout[-3_000:] if len(result.stdout) > 3_000 else result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr[-500:])
else:
    print(f"ACS data present: {ACS_CSV}")

# ── Load embeddings ────────────────────────────────────────────────────────────
_meta     = {"GEOID", "year", "ALAND", "AWATER", "tract_fips"}
_stat_set = set(STAT_SUFFIXES)

_raw = pl.read_csv(
    EMBEDDINGS_PATH,
    infer_schema_length=10_000,
    schema_overrides={"GEOID": pl.Utf8},
)
_null_col = next(
    (c for c in _raw.columns
     if any(c.endswith(f"_{s}") for s in _stat_set) and c not in _meta),
    None,
)

emb_tract = (
    _raw
    .filter(pl.col("year") == 2022)
    .pipe(lambda df: df.filter(pl.col(_null_col).is_not_null()) if _null_col else df)
    .with_columns(pl.col("GEOID").str.zfill(11).alias("tract_fips"))
)

EMB_COLS     = [c for c in emb_tract.columns
                if any(c.endswith(f"_{s}") for s in _stat_set) and c not in _meta]
FEATURE_COLS = EMB_COLS + [c for c in ["ALAND", "AWATER"] if c in emb_tract.columns]

emb_tract = emb_tract.select(["tract_fips"] + FEATURE_COLS)

print(f"Embeddings loaded : {len(emb_tract):,} tracts")
print(f"Embedding features: {len(EMB_COLS)}  +  {len(FEATURE_COLS) - len(EMB_COLS)} area cols")

# ── Load PLACES ───────────────────────────────────────────────────────────────
_places_raw = pl.read_csv(PLACES_CSV, infer_schema_length=10_000)

PLACES_LABELS: dict[str, str] = dict(
    _places_raw.select(["MeasureId", "Short_Question_Text"]).unique().iter_rows()
)

_places_long = (
    _places_raw
    .filter(pl.col("Year").is_in([2022, 2023]))
    .select(["Year", "LocationID", "MeasureId", "Data_Value"])
    .with_columns(
        pl.col("LocationID").cast(pl.Utf8).str.zfill(11).alias("tract_fips")
    )
    .drop("LocationID")
)

_measures_2022 = set(
    _places_long.filter(pl.col("Year") == 2022)["MeasureId"].unique().to_list()
)
_measures_2023_only = set(
    _places_long.filter(pl.col("Year") == 2023)["MeasureId"].unique().to_list()
) - _measures_2022

places_wide = (
    pl.concat([
        _places_long.filter((pl.col("Year") == 2022) & pl.col("MeasureId").is_in(_measures_2022)),
        _places_long.filter((pl.col("Year") == 2023) & pl.col("MeasureId").is_in(_measures_2023_only)),
    ])
    .drop("Year")
    .pivot(on="MeasureId", index="tract_fips", values="Data_Value")
)

PLACES_MEASURES = sorted([c for c in places_wide.columns if c != "tract_fips"])
print(f"PLACES: {len(places_wide):,} tracts × {len(PLACES_MEASURES)} measures")

# ── Load ACS ──────────────────────────────────────────────────────────────────
ACS_RENAME = {
    "B17021_001E": "pov_universe",
    "B17021_002E": "pov_below_related_children",
    "B17021_003E": "pov_below_other",
    "S1701_C01_001E": "pov_determination_universe",
    "S1701_C01_040E": "pct_below_150pct_poverty",
    "S1701_C02_001E": "pct_below_100pct_poverty",
    "B19001_002E": "hh_income_lt10k",
    "B19001_017E": "hh_income_ge200k",
    "B19113_001E": "median_family_income",
    "B19301_001E": "per_capita_income",
    "B23025_003E": "labor_force",
    "B23025_005E": "unemployed",
    "DP03_0009PE": "unemployment_pct",
    "S2301_C04_001E": "not_in_labor_force_pct",
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
    "B15003_017E": "educ_hs_diploma",
    "B15003_018E": "educ_ged",
    "B15003_019E": "educ_some_college_lt1yr",
    "B15003_020E": "educ_some_college_ge1yr",
    "B15003_021E": "educ_associates",
    "B15003_022E": "educ_bachelors",
    "B15003_023E": "educ_masters",
    "B15003_024E": "educ_professional_degree",
    "B15003_025E": "educ_doctorate",
    "S0601_C01_001E": "pop_25plus",
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
    "C24010_001E": "occ_universe",
    "C24010_019E": "occ_male_mgmt_business_science_arts",
    "C24010_020E": "occ_male_business_financial",
    "C24010_021E": "occ_male_computer_engineering_science",
    "C24010_022E": "occ_male_education_legal_arts",
    "C24010_023E": "occ_male_healthcare_practitioners",
    "C24010_024E": "occ_male_service",
    "C24010_025E": "occ_male_sales_office",
    "C24010_039E": "occ_female_mgmt_business_science_arts",
    "C24010_040E": "occ_female_business_financial",
    "C24010_041E": "occ_female_computer_engineering_science",
    "C24010_042E": "occ_female_education_legal_arts",
    "C24010_043E": "occ_female_healthcare_practitioners",
    "C24010_044E": "occ_female_service",
    "C24010_045E": "occ_female_sales_office",
    "B25064_001E": "median_gross_rent",
    "B25077_001E": "median_home_value",
    "B25003_001E": "tenure_universe",
    "B25003_002E": "owner_occupied_units",
    "B25014_001E": "occupants_per_room_universe",
    "B25014_005E": "overcrowded_owner_1_to_1_5_per_room",
    "B25014_006E": "overcrowded_owner_1_5_to_2_per_room",
    "B25014_007E": "overcrowded_owner_2plus_per_room",
    "B25040_001E": "heating_fuel_universe",
    "B25040_003E": "heating_fuel_utility_gas",
    "B25044_001E": "vehicle_availability_universe",
    "B25044_003E": "owner_occupied_no_vehicle",
    "B25044_010E": "renter_occupied_no_vehicle",
    "B25047_001E": "plumbing_universe",
    "B25047_002E": "incomplete_plumbing",
    "B26001_001E": "group_quarters_pop",
    "S2503_C01_001E": "renter_units_universe",
    "S2503_C01_028E": "rent_30_34pct_of_income",
    "S2503_C01_032E": "rent_35_39pct_of_income",
    "S2503_C01_036E": "rent_40_49pct_of_income",
    "S2503_C01_040E": "rent_50plus_pct_of_income",
    "DP05_0001E": "total_pop",
    "DP05_0019PE": "age_lte17_pct",
    "DP05_0079E": "nh_white_alone",
    "S0101_C02_030E": "age_65plus_pct",
    "DP02_0001E": "total_households",
    "DP02_0007PE": "single_parent_male_hh_pct",
    "DP02_0011PE": "single_parent_female_hh_pct",
    "DP02_0072PE": "disability_pct",
    "DP04_0012PE": "structures_10_19_units_pct",
    "DP04_0013PE": "structures_20plus_units_pct",
    "DP04_0014PE": "mobile_homes_pct",
    "DP04_0047PE": "renter_occupied_pct",
    "DP04_0058PE": "no_vehicle_pct",
    "DP04_0078PE": "overcrowded_owner_pct",
    "DP04_0079PE": "overcrowded_renter_pct",
    "S2701_C05_001E": "uninsured_pct",
}

_CONVERSIONS = [
    ("pov_below_related_children",         "pov_universe",                None),
    ("pov_below_other",                    "pov_universe",                None),
    ("hh_income_lt10k",                    "total_households",            None),
    ("hh_income_ge200k",                   "total_households",            None),
    ("educ_no_schooling",                  "educ_universe",               None),
    ("educ_nursery",                       "educ_universe",               None),
    ("educ_kindergarten",                  "educ_universe",               None),
    ("educ_grade1",                        "educ_universe",               None),
    ("educ_grade2",                        "educ_universe",               None),
    ("educ_grade3",                        "educ_universe",               None),
    ("educ_grade4",                        "educ_universe",               None),
    ("educ_grade5",                        "educ_universe",               None),
    ("educ_grade6",                        "educ_universe",               None),
    ("educ_grade7",                        "educ_universe",               None),
    ("educ_grade8",                        "educ_universe",               None),
    ("educ_grade9",                        "educ_universe",               None),
    ("educ_grade10",                       "educ_universe",               None),
    ("educ_grade11",                       "educ_universe",               None),
    ("educ_grade12_no_diploma",            "educ_universe",               None),
    ("educ_hs_diploma",                    "educ_universe",               None),
    ("educ_ged",                           "educ_universe",               None),
    ("educ_some_college_lt1yr",            "educ_universe",               None),
    ("educ_some_college_ge1yr",            "educ_universe",               None),
    ("educ_associates",                    "educ_universe",               None),
    ("educ_bachelors",                     "educ_universe",               None),
    ("educ_masters",                       "educ_universe",               None),
    ("educ_professional_degree",           "educ_universe",               None),
    ("educ_doctorate",                     "educ_universe",               None),
    ("lep_spanish_native_not_well",        "lang_universe",               None),
    ("lep_spanish_native_not_at_all",      "lang_universe",               None),
    ("lep_spanish_foreign_not_well",       "lang_universe",               None),
    ("lep_spanish_foreign_not_at_all",     "lang_universe",               None),
    ("lep_indo_euro_native_not_well",      "lang_universe",               None),
    ("lep_indo_euro_native_not_at_all",    "lang_universe",               None),
    ("lep_indo_euro_foreign_not_well",     "lang_universe",               None),
    ("lep_indo_euro_foreign_not_at_all",   "lang_universe",               None),
    ("lep_asian_pac_native_not_well",      "lang_universe",               None),
    ("lep_asian_pac_native_not_at_all",    "lang_universe",               None),
    ("lep_asian_pac_foreign_not_well",     "lang_universe",               None),
    ("lep_asian_pac_foreign_not_at_all",   "lang_universe",               None),
    ("lep_other_native_not_well",          "lang_universe",               None),
    ("lep_other_native_not_at_all",        "lang_universe",               None),
    ("lep_other_foreign_not_well",         "lang_universe",               None),
    ("lep_other_foreign_not_at_all",       "lang_universe",               None),
    ("occ_male_mgmt_business_science_arts",    "occ_universe",            None),
    ("occ_male_business_financial",            "occ_universe",            None),
    ("occ_male_computer_engineering_science",  "occ_universe",            None),
    ("occ_male_education_legal_arts",          "occ_universe",            None),
    ("occ_male_healthcare_practitioners",      "occ_universe",            None),
    ("occ_male_service",                       "occ_universe",            None),
    ("occ_male_sales_office",                  "occ_universe",            None),
    ("occ_female_mgmt_business_science_arts",  "occ_universe",            None),
    ("occ_female_business_financial",          "occ_universe",            None),
    ("occ_female_computer_engineering_science","occ_universe",            None),
    ("occ_female_education_legal_arts",        "occ_universe",            None),
    ("occ_female_healthcare_practitioners",    "occ_universe",            None),
    ("occ_female_service",                     "occ_universe",            None),
    ("occ_female_sales_office",                "occ_universe",            None),
    ("owner_occupied_units",               "tenure_universe",             "owner_occupied_pct"),
    ("overcrowded_owner_1_to_1_5_per_room","occupants_per_room_universe", None),
    ("overcrowded_owner_1_5_to_2_per_room","occupants_per_room_universe", None),
    ("overcrowded_owner_2plus_per_room",   "occupants_per_room_universe", None),
    ("heating_fuel_utility_gas",           "heating_fuel_universe",       "heating_fuel_utility_gas_pct"),
    ("owner_occupied_no_vehicle",          "vehicle_availability_universe","owner_occupied_no_vehicle_pct"),
    ("renter_occupied_no_vehicle",         "vehicle_availability_universe","renter_occupied_no_vehicle_pct"),
    ("incomplete_plumbing",                "plumbing_universe",           "incomplete_plumbing_pct"),
    ("group_quarters_pop",                 "total_pop",                   "group_quarters_pct"),
    ("nh_white_alone",                     "total_pop",                   "nh_white_pct"),
    ("rent_30_34pct_of_income",            "renter_units_universe",       "cost_burdened_rent_30_34_pct"),
    ("rent_35_39pct_of_income",            "renter_units_universe",       "cost_burdened_rent_35_39_pct"),
    ("rent_40_49pct_of_income",            "renter_units_universe",       "cost_burdened_rent_40_49_pct"),
    ("rent_50plus_pct_of_income",          "renter_units_universe",       "cost_burdened_rent_50plus_pct"),
]

if ACS_CSV.exists():
    _acs_raw = (
        pl.read_csv(ACS_CSV, infer_schema_length=5_000)
        .with_columns(pl.col("GEOID").cast(pl.Utf8).str.zfill(11).alias("tract_fips"))
        .drop(["GEOID", "NAME"])
    )
    acs = _acs_raw.rename({k: v for k, v in ACS_RENAME.items() if k in _acs_raw.columns})

    _available = set(acs.columns)
    _valid = [(n, d, name) for n, d, name in _CONVERSIONS if n in _available and d in _available]
    acs = acs.with_columns([
        (pl.col(num) / pl.col(den) * 100).alias(name or f"{num}_pct")
        for num, den, name in _valid
    ]).drop([
        c for c in (
            [num for num, _, _ in _valid]
            + ["pov_universe", "pov_determination_universe", "educ_universe", "pop_25plus",
               "lang_universe", "occ_universe", "tenure_universe", "occupants_per_room_universe",
               "heating_fuel_universe", "vehicle_availability_universe", "plumbing_universe",
               "renter_units_universe", "labor_force", "unemployed"]
        ) if c in _available
    ])

    _num_cols = [c for c in acs.columns if c != "tract_fips"]
    acs = acs.with_columns([
        pl.when(pl.col(c) < -99999).then(None).otherwise(pl.col(c)).alias(c)
        for c in _num_cols
    ])

    ACS_VARS = [c for c in acs.columns if c != "tract_fips"]
    print(f"ACS: {len(acs):,} tracts × {len(ACS_VARS)} variables")
else:
    acs = None
    ACS_VARS = []
    print("ACS not available — Q1 will be skipped")

# ── Composite index variables ──────────────────────────────────────────────────
if acs is not None:
    _lep_cols    = [c for c in acs.columns if c.startswith("lep_") and c.endswith("_pct")]
    _lt9_cols    = [
        "educ_no_schooling_pct", "educ_nursery_pct", "educ_kindergarten_pct",
        "educ_grade1_pct", "educ_grade2_pct", "educ_grade3_pct",
        "educ_grade4_pct", "educ_grade5_pct", "educ_grade6_pct", "educ_grade7_pct",
    ]
    _lt12_cols   = _lt9_cols + [
        "educ_grade8_pct", "educ_grade9_pct", "educ_grade10_pct",
        "educ_grade11_pct", "educ_grade12_no_diploma_pct",
    ]
    _hsplus_cols = [
        "educ_hs_diploma_pct", "educ_ged_pct", "educ_some_college_lt1yr_pct",
        "educ_some_college_ge1yr_pct", "educ_associates_pct", "educ_bachelors_pct",
        "educ_masters_pct", "educ_professional_degree_pct", "educ_doctorate_pct",
    ]
    _occ_cols    = [c for c in acs.columns if c.startswith("occ_") and c.endswith("_pct")]
    _burden_cols = [
        "cost_burdened_rent_30_34_pct", "cost_burdened_rent_35_39_pct",
        "cost_burdened_rent_40_49_pct", "cost_burdened_rent_50plus_pct",
    ]

    acs = acs.with_columns([
        pl.sum_horizontal([pl.col(c) for c in _lep_cols]).alias("lep_total_pct"),
        (pl.col("single_parent_male_hh_pct") + pl.col("single_parent_female_hh_pct")).alias("single_parent_pct"),
        (pl.col("overcrowded_owner_pct") + pl.col("overcrowded_renter_pct")).alias("crowded_housing_pct"),
        (pl.col("structures_10_19_units_pct") + pl.col("structures_20plus_units_pct")).alias("multi_unit_housing_pct"),
        pl.sum_horizontal([pl.col(c) for c in _burden_cols]).alias("housing_cost_burdened_pct"),
        (100.0 - pl.col("nh_white_pct")).alias("minority_pct"),
        (pl.col("pov_below_related_children_pct") + pl.col("pov_below_other_pct")).alias("pov_total_pct"),
        pl.sum_horizontal([pl.col(c) for c in _lt9_cols]).alias("educ_lt9th_grade_pct"),
        pl.sum_horizontal([pl.col(c) for c in _hsplus_cols]).alias("educ_hs_or_higher_pct"),
        pl.sum_horizontal([pl.col(c) for c in _lt12_cols]).alias("educ_lt12th_grade_pct"),
        pl.sum_horizontal([pl.col(c) for c in _occ_cols]).alias("white_collar_pct"),
        (pl.col("owner_occupied_no_vehicle_pct") + pl.col("renter_occupied_no_vehicle_pct")).alias("no_vehicle_b25044_pct"),
        pl.sum_horizontal([
            pl.col("overcrowded_owner_1_to_1_5_per_room_pct"),
            pl.col("overcrowded_owner_1_5_to_2_per_room_pct"),
            pl.col("overcrowded_owner_2plus_per_room_pct"),
        ]).alias("crowded_b25014_pct"),
        (pl.col("hh_income_lt10k_pct") /
         pl.when(pl.col("hh_income_ge200k_pct") > 0)
           .then(pl.col("hh_income_ge200k_pct"))
           .otherwise(None)
        ).alias("income_disparity_ratio"),
    ])

    ACS_VARS = [c for c in acs.columns if c != "tract_fips"]

INDEX_MEMBERSHIP: dict[str, set[str]] = {
    "lep_total_pct":               {"SVI"},
    "single_parent_pct":           {"SVI", "SDI"},
    "crowded_housing_pct":         {"SVI", "SDI"},
    "multi_unit_housing_pct":      {"SVI"},
    "housing_cost_burdened_pct":   {"SVI"},
    "minority_pct":                {"SVI"},
    "pov_total_pct":               {"ADI"},
    "educ_lt9th_grade_pct":        {"ADI"},
    "educ_hs_or_higher_pct":       {"ADI"},
    "educ_lt12th_grade_pct":       {"SDI"},
    "white_collar_pct":            {"ADI"},
    "no_vehicle_b25044_pct":       {"ADI"},
    "crowded_b25014_pct":          {"ADI"},
    "income_disparity_ratio":      {"ADI"},
    "median_owner_cost_mortgage":  {"ADI"},
    "pct_no_telephone":            {"ADI"},
    "pct_below_150pct_poverty":                    {"SVI"},
    "unemployment_pct":                            {"SVI", "ADI"},
    "cost_burdened_rent_30_34_pct":                {"SVI"},
    "cost_burdened_rent_35_39_pct":                {"SVI"},
    "cost_burdened_rent_40_49_pct":                {"SVI"},
    "cost_burdened_rent_50plus_pct":               {"SVI"},
    "no_hs_diploma_pct":                           {"SVI"},
    "uninsured_pct":                               {"SVI"},
    "age_65plus_pct":                              {"SVI"},
    "age_lte17_pct":                               {"SVI"},
    "disability_pct":                              {"SVI"},
    "single_parent_male_hh_pct":                   {"SVI", "SDI"},
    "single_parent_female_hh_pct":                 {"SVI", "SDI"},
    "lep_spanish_native_not_well_pct":             {"SVI"},
    "lep_spanish_native_not_at_all_pct":           {"SVI"},
    "lep_spanish_foreign_not_well_pct":            {"SVI"},
    "lep_spanish_foreign_not_at_all_pct":          {"SVI"},
    "lep_indo_euro_native_not_well_pct":           {"SVI"},
    "lep_indo_euro_native_not_at_all_pct":         {"SVI"},
    "lep_indo_euro_foreign_not_well_pct":          {"SVI"},
    "lep_indo_euro_foreign_not_at_all_pct":        {"SVI"},
    "lep_asian_pac_native_not_well_pct":           {"SVI"},
    "lep_asian_pac_native_not_at_all_pct":         {"SVI"},
    "lep_asian_pac_foreign_not_well_pct":          {"SVI"},
    "lep_asian_pac_foreign_not_at_all_pct":        {"SVI"},
    "lep_other_native_not_well_pct":               {"SVI"},
    "lep_other_native_not_at_all_pct":             {"SVI"},
    "lep_other_foreign_not_well_pct":              {"SVI"},
    "lep_other_foreign_not_at_all_pct":            {"SVI"},
    "nh_white_pct":                                {"SVI"},
    "structures_10_19_units_pct":                  {"SVI"},
    "structures_20plus_units_pct":                 {"SVI"},
    "mobile_homes_pct":                            {"SVI"},
    "overcrowded_owner_pct":                       {"SVI", "SDI"},
    "overcrowded_renter_pct":                      {"SVI", "SDI"},
    "no_vehicle_pct":                  {"SVI", "SDI", "ADI"},
    "group_quarters_pct":                          {"SVI"},
    "pct_below_100pct_poverty":                    {"SDI"},
    "educ_no_schooling_pct":                       {"SDI", "ADI"},
    "educ_nursery_pct":                            {"SDI", "ADI"},
    "educ_kindergarten_pct":                       {"SDI", "ADI"},
    "educ_grade1_pct":                             {"SDI", "ADI"},
    "educ_grade2_pct":                             {"SDI", "ADI"},
    "educ_grade3_pct":                             {"SDI", "ADI"},
    "educ_grade4_pct":                             {"SDI", "ADI"},
    "educ_grade5_pct":                             {"SDI", "ADI"},
    "educ_grade6_pct":                             {"SDI", "ADI"},
    "educ_grade7_pct":                             {"SDI", "ADI"},
    "educ_grade8_pct":                             {"SDI"},
    "educ_grade9_pct":                             {"SDI"},
    "educ_grade10_pct":                            {"SDI"},
    "educ_grade11_pct":                            {"SDI"},
    "educ_grade12_no_diploma_pct":                 {"SDI"},
    "renter_occupied_pct":                         {"SDI"},
    "not_in_labor_force_pct":                      {"SDI"},
    "pov_below_related_children_pct":              {"ADI"},
    "pov_below_other_pct":                         {"ADI"},
    "educ_hs_diploma_pct":                         {"ADI"},
    "educ_ged_pct":                                {"ADI"},
    "educ_some_college_lt1yr_pct":                 {"ADI"},
    "educ_some_college_ge1yr_pct":                 {"ADI"},
    "educ_associates_pct":                         {"ADI"},
    "educ_bachelors_pct":                          {"ADI"},
    "educ_masters_pct":                            {"ADI"},
    "educ_professional_degree_pct":                {"ADI"},
    "educ_doctorate_pct":                          {"ADI"},
    "median_family_income":                        {"ADI"},
    "per_capita_income":                           {"ADI"},
    "hh_income_lt10k_pct":                         {"ADI"},
    "hh_income_ge200k_pct":                        {"ADI"},
    "occ_male_mgmt_business_science_arts_pct":     {"ADI"},
    "occ_male_business_financial_pct":             {"ADI"},
    "occ_male_computer_engineering_science_pct":   {"ADI"},
    "occ_male_education_legal_arts_pct":           {"ADI"},
    "occ_male_healthcare_practitioners_pct":       {"ADI"},
    "occ_male_service_pct":                        {"ADI"},
    "occ_male_sales_office_pct":                   {"ADI"},
    "occ_female_mgmt_business_science_arts_pct":   {"ADI"},
    "occ_female_business_financial_pct":           {"ADI"},
    "occ_female_computer_engineering_science_pct": {"ADI"},
    "occ_female_education_legal_arts_pct":         {"ADI"},
    "occ_female_healthcare_practitioners_pct":     {"ADI"},
    "occ_female_service_pct":                      {"ADI"},
    "occ_female_sales_office_pct":                 {"ADI"},
    "median_home_value":                           {"ADI"},
    "median_gross_rent":                           {"ADI"},
    "overcrowded_owner_1_to_1_5_per_room_pct":     {"ADI"},
    "overcrowded_owner_1_5_to_2_per_room_pct":     {"ADI"},
    "overcrowded_owner_2plus_per_room_pct":        {"ADI"},
    "owner_occupied_no_vehicle_pct":               {"ADI"},
    "renter_occupied_no_vehicle_pct":              {"ADI"},
    "incomplete_plumbing_pct":                     {"ADI"},
    "owner_occupied_pct":                          {"ADI"},
    "heating_fuel_utility_gas_pct":                {"ADI"},
}

ACS_VARS = [v for v in ACS_VARS if v in INDEX_MEMBERSHIP]
_COMPOSITE_VARS = [
    'lep_total_pct','single_parent_pct','crowded_housing_pct','multi_unit_housing_pct',
    'housing_cost_burdened_pct','minority_pct','pov_total_pct','educ_lt9th_grade_pct',
    'educ_hs_or_higher_pct','educ_lt12th_grade_pct','white_collar_pct',
    'no_vehicle_b25044_pct','crowded_b25014_pct','income_disparity_ratio',
]
print(
    f"Index-relevant ACS variables: {len(ACS_VARS)}  "
    f"({sum(1 for v in ACS_VARS if v in _COMPOSITE_VARS)} composite, "
    f"{sum(1 for v in ACS_VARS if v not in _COMPOSITE_VARS)} sub-components)"
)

# ── Merge embeddings + PLACES + ACS ───────────────────────────────────────────
df = emb_tract.join(places_wide, on="tract_fips", how="inner")

if acs is not None:
    df = df.join(acs, on="tract_fips", how="left")

_readi_extras = (
    pl.read_csv(SRI_DIR / "ReADI_CT_2022.csv", infer_schema_length=5_000)
    .select(["GEOID", "MEDMORT", "NOINT"])
    .with_columns(pl.col("GEOID").cast(pl.Utf8).str.zfill(11).alias("tract_fips"))
    .drop("GEOID")
    .rename({"MEDMORT": "median_owner_cost_mortgage", "NOINT": "pct_no_telephone"})
)
df = df.join(_readi_extras, on="tract_fips", how="left")
ACS_VARS = ACS_VARS + ["median_owner_cost_mortgage", "pct_no_telephone"]

print(f"Merged dataset: {len(df):,} tracts × {df.shape[1]} columns")

# ── Q1: ACS correlations ───────────────────────────────────────────────────────
def lgbm_cv_r2(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
) -> tuple[float, float]:
    kf = GroupKFold(n_splits=n_splits)
    scores = []
    for train_idx, val_idx in kf.split(X, y, groups=groups):
        model = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            min_child_samples=20, random_state=42, verbosity=-1,
        )
        model.fit(X[train_idx], y[train_idx])
        scores.append(r2_score(y[val_idx], model.predict(X[val_idx])))
    return float(np.mean(scores)), float(np.std(scores))


_q1_cache = OUTPUTS_DIR / "acs_correlations.csv"

if USE_CACHED and _q1_cache.exists():
    results_df = pl.read_csv(_q1_cache)
    print(f"Q1: loaded from cache ({len(results_df)} rows)")
else:
    print("Q1: fitting LightGBM models ...")
    df_pd = df.to_pandas()
    X_all = df_pd[FEATURE_COLS].to_numpy(dtype=float)
    state_all = df_pd["tract_fips"].str[:2].to_numpy()

    all_targets = [("PLACES", m) for m in PLACES_MEASURES] + [("ACS", v) for v in ACS_VARS]
    results = []

    for group, col in all_targets:
        if col not in df_pd.columns:
            continue
        y_series = df_pd[col]
        mask = y_series.notna().to_numpy()
        if mask.sum() < 100:
            continue
        y = y_series[mask].to_numpy(dtype=float)
        X = X_all[mask]
        states = state_all[mask]
        r2_mean, r2_std = lgbm_cv_r2(X, y, groups=states)
        results.append({"group": group, "variable": col, "r2_mean": r2_mean, "r2_std": r2_std})

    results_df = (
        pl.DataFrame(results)
        .with_columns(
            pl.col("variable").map_elements(
                lambda v: ", ".join(sorted(INDEX_MEMBERSHIP.get(v, set()))),
                return_dtype=pl.Utf8,
            ).alias("indices")
        )
        .sort("r2_mean", descending=True)
    )
    results_df.write_csv(_q1_cache)
    print(f"Q1: done — {len(results_df)} targets evaluated")


def plot_r2_bars(ax: plt.Axes, data: pl.DataFrame, title: str) -> None:
    data_s = data.sort("r2_mean")
    labels = data_s["variable"].to_list()
    means  = data_s["r2_mean"].to_list()
    stds   = data_s["r2_std"].to_list()
    colors = ["#2196F3" if v >= 0 else "#E53935" for v in means]
    ax.barh(labels, means, xerr=stds, color=colors, capsize=3,
            alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Cross-validated R²  (5-fold)", fontsize=11)
    ax.set_xlim(-0.15, 1.0)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.tick_params(axis="y", labelsize=9)


acs_results = results_df.filter(pl.col("group") == "ACS")
n_acs = len(acs_results)

if n_acs > 0:
    fig, ax = plt.subplots(figsize=(10, max(8, 0.38 * n_acs)))
    plot_r2_bars(ax, acs_results, "ACS tract variables")
    plt.suptitle(
        f"Variance in tract-level ACS variables explained by {args.model} embeddings  |  "
        f"n = {len(df):,} tracts (2022)",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "r2_embeddings.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Q1 plot saved — {n_acs} ACS variables")

# ── Q2: Embeddings vs. indices for PLACES ─────────────────────────────────────
FIPS_TO_ABBR = {
    "01":"AL","02":"AK","04":"AZ","05":"AR","06":"CA","08":"CO","09":"CT",
    "10":"DE","11":"DC","12":"FL","13":"GA","15":"HI","16":"ID","17":"IL",
    "18":"IN","19":"IA","20":"KS","21":"KY","22":"LA","23":"ME","24":"MD",
    "25":"MA","26":"MI","27":"MN","28":"MS","29":"MO","30":"MT","31":"NE",
    "32":"NV","33":"NH","34":"NJ","35":"NM","36":"NY","37":"NC","38":"ND",
    "39":"OH","40":"OK","41":"OR","42":"PA","44":"RI","45":"SC","46":"SD",
    "47":"TN","48":"TX","49":"UT","50":"VT","51":"VA","53":"WA","54":"WV",
    "55":"WI","56":"WY",
}

readi = (
    pl.read_csv(SRI_DIR / "ReADI_CT_2022.csv", infer_schema_length=5_000)
    .select(["GEOID", "ReADI_CT_NR"])
    .with_columns(pl.col("GEOID").cast(pl.Utf8).str.zfill(11).alias("tract_fips"))
    .drop("GEOID")
)

sdvi = (
    pl.read_csv(SRI_DIR / "sdvi_ct_2020.csv", infer_schema_length=5_000, null_values=["NA"])
    .select(["GEOID", "SVI", "SDI"])
    .with_columns(pl.col("GEOID").cast(pl.Utf8).str.zfill(11).alias("tract_fips"))
    .drop("GEOID")
    .with_columns([
        (pl.col("SVI").rank(method="average") / pl.col("SVI").count() * 100).alias("SVI_NR"),
        (pl.col("SDI").rank(method="average") / pl.col("SDI").count() * 100).alias("SDI_NR"),
    ])
    .drop(["SVI", "SDI"])
)

df2 = (
    df.join(readi, on="tract_fips", how="inner")
      .join(sdvi, on="tract_fips", how="left")
      .with_columns(pl.col("tract_fips").str.slice(0, 2).alias("state_fips"))
)

all_states = sorted(df2["state_fips"].unique().to_list())
rng = np.random.default_rng(42)
holdout_states = set(rng.choice(all_states, size=max(1, round(0.2 * len(all_states))), replace=False).tolist())
train_states   = set(all_states) - holdout_states

df2_pd     = df2.to_pandas()
train_mask = df2_pd["state_fips"].isin(train_states).to_numpy()
test_mask  = df2_pd["state_fips"].isin(holdout_states).to_numpy()

READI_FEATS     = ["ReADI_CT_NR"]
SVI_FEATS       = ["SVI_NR"]
SDI_FEATS       = ["SDI_NR"]
ALL_INDEX_FEATS = READI_FEATS + SVI_FEATS + SDI_FEATS
COMBINED_FEATS  = ALL_INDEX_FEATS + FEATURE_COLS

holdout_abbrs = sorted(FIPS_TO_ABBR.get(s, s) for s in holdout_states)
print(f"Training states : {len(train_states)}  ({train_mask.sum():,} tracts)")
print(f"Held-out states : {len(holdout_states)}  ({test_mask.sum():,} tracts)  {holdout_abbrs}")


def fit_eval(feat_cols: list[str], y: np.ndarray, train: np.ndarray, test: np.ndarray) -> float:
    X_tr = df2_pd.loc[train, feat_cols].to_numpy(dtype=float)
    X_te = df2_pd.loc[test,  feat_cols].to_numpy(dtype=float)
    model = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        min_child_samples=20, random_state=42, verbosity=-1,
    )
    model.fit(X_tr, y[train])
    return float(r2_score(y[test], model.predict(X_te)))


_q2_cache = OUTPUTS_DIR / "places_reg.csv"

if USE_CACHED and _q2_cache.exists():
    results_q2_df = pl.read_csv(_q2_cache)
    print(f"Q2: loaded from cache ({len(results_q2_df)} rows)")
else:
    print("Q2: fitting LightGBM models ...")
    results_q2 = []
    for col in PLACES_MEASURES:
        if col not in df2_pd.columns:
            continue
        y = df2_pd[col].to_numpy(dtype=float)
        valid       = ~np.isnan(y)
        train_valid = train_mask & valid
        test_valid  = test_mask  & valid
        if train_valid.sum() < 100 or test_valid.sum() < 50:
            continue

        for feat_cols, model_name in [
            (READI_FEATS,     "ReADI"),
            (SVI_FEATS,       "SVI"),
            (SDI_FEATS,       "SDI"),
            (ALL_INDEX_FEATS, "All indices"),
            (FEATURE_COLS,    "Embeddings"),
            (COMBINED_FEATS,  "All indices + Embeddings"),
        ]:
            r2 = fit_eval(feat_cols, y, train_valid, test_valid)
            results_q2.append({"outcome": col, "model": model_name, "r2": r2,
                                "n_train": int(train_valid.sum()), "n_test": int(test_valid.sum())})

    results_q2_df = pl.DataFrame(results_q2)
    results_q2_df.write_csv(_q2_cache)
    print(f"Q2: done — {len(results_q2_df)} rows")


MODEL_ORDER  = ["ReADI", "SVI", "SDI", "All indices", "Embeddings", "All indices + Embeddings"]
MODEL_COLORS = {
    "ReADI":                    "#E53935",
    "SVI":                      "#FF7043",
    "SDI":                      "#FFA726",
    "All indices":              "#AB47BC",
    "Embeddings":               "#2196F3",
    "All indices + Embeddings": "#43A047",
}

readi_r2 = (
    results_q2_df.filter(pl.col("model") == "ReADI")
    .sort("r2")
    .select("outcome")
    .to_series()
    .to_list()
)
readi_r2_labels = [PLACES_LABELS.get(o, o) for o in readi_r2]

fig, ax = plt.subplots(figsize=(9, max(4, 0.6 * len(readi_r2))))
for model_name in MODEL_ORDER:
    sub = (
        results_q2_df.filter(pl.col("model") == model_name)
        .with_columns(pl.col("outcome").cast(pl.Enum(readi_r2)))
        .sort("outcome")
    )
    ys     = [readi_r2.index(o) for o in sub["outcome"].to_list()]
    r2vals = sub["r2"].to_list()
    ax.scatter(r2vals, ys, color=MODEL_COLORS[model_name], label=model_name,
               zorder=3, s=80, edgecolors="white", linewidths=0.6)

for i, outcome in enumerate(readi_r2):
    row = results_q2_df.filter(pl.col("outcome") == outcome).sort("model")
    r2s = {r["model"]: r["r2"] for r in row.iter_rows(named=True)}
    xs  = [r2s.get(m, np.nan) for m in MODEL_ORDER]
    ax.plot(xs, [i] * len(MODEL_ORDER), color="grey", linewidth=0.8, zorder=2, alpha=0.5)

ax.set_yticks(range(len(readi_r2)))
ax.set_yticklabels(readi_r2_labels, fontsize=10)
ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xlabel("R²  on held-out states (20%)", fontsize=11)
ax.set_title(
    f"PLACES outcomes: {args.model} embeddings vs. social risk indices\n"
    f"Held-out states: {', '.join(holdout_abbrs)}",
    fontsize=11, fontweight="bold",
)
ax.legend(loc="lower right", framealpha=0.9)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "places_reg.png", dpi=150, bbox_inches="tight")
plt.close()
print("Q2 plot saved")

# ── Q3: Residual variance captured by embeddings ──────────────────────────────
def residual_r2(
    col: str,
    y: np.ndarray,
    train_valid: np.ndarray,
    test_valid: np.ndarray,
    index_feats: list[str],
    index_name: str,
) -> dict:
    lgbm_kw = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
                   min_child_samples=20, random_state=42, verbosity=-1)

    X_r_tr = df2_pd.loc[train_valid, index_feats].to_numpy(dtype=float)
    X_r_te = df2_pd.loc[test_valid,  index_feats].to_numpy(dtype=float)
    m1 = lgb.LGBMRegressor(**lgbm_kw)
    m1.fit(X_r_tr, y[train_valid])
    resid_train = y[train_valid] - m1.predict(X_r_tr)
    resid_test  = y[test_valid]  - m1.predict(X_r_te)
    r2_s1 = float(r2_score(y[test_valid], m1.predict(X_r_te)))

    X_e_tr = df2_pd.loc[train_valid, FEATURE_COLS].to_numpy(dtype=float)
    X_e_te = df2_pd.loc[test_valid,  FEATURE_COLS].to_numpy(dtype=float)
    m2 = lgb.LGBMRegressor(**lgbm_kw)
    m2.fit(X_e_tr, resid_train)
    r2_resid = float(r2_score(resid_test, m2.predict(X_e_te)))
    additional_var = max(0.0, r2_resid) * max(0.0, 1.0 - r2_s1)

    return {
        "index":          index_name,
        "outcome":        col,
        "r2_index":       r2_s1,
        "r2_residual":    r2_resid,
        "additional_var": additional_var,
        "n_train":        int(train_valid.sum()),
        "n_test":         int(test_valid.sum()),
    }


_q3_cache = OUTPUTS_DIR / "places_residual_by_index.csv"

if USE_CACHED and _q3_cache.exists():
    results_q3_df = pl.read_csv(_q3_cache)
    print(f"Q3: loaded from cache ({len(results_q3_df)} rows)")
else:
    print("Q3: fitting residual models ...")
    results_q3 = []
    for col in PLACES_MEASURES:
        if col not in df2_pd.columns:
            continue
        y = df2_pd[col].to_numpy(dtype=float)
        valid       = ~np.isnan(y)
        train_valid = train_mask & valid
        test_valid  = test_mask  & valid
        if train_valid.sum() < 100 or test_valid.sum() < 50:
            continue
        for index_feats, index_name in [
            (READI_FEATS, "ReADI"),
            (SVI_FEATS,   "SVI"),
            (SDI_FEATS,   "SDI"),
        ]:
            results_q3.append(residual_r2(col, y, train_valid, test_valid, index_feats, index_name))

    results_q3_df = pl.DataFrame(results_q3).sort(["index", "r2_residual"], descending=[False, True])
    results_q3_df.write_csv(_q3_cache)
    print(f"Q3: done — {len(results_q3_df)} rows")


INDEX_NAMES  = ["ReADI", "SVI", "SDI"]
INDEX_COLORS = {"ReADI": "#E53935", "SVI": "#FF7043", "SDI": "#FFA726"}

readi_order = results_q3_df.filter(pl.col("index") == "ReADI").sort("r2_residual")["outcome"].to_list()
svi_order   = results_q3_df.filter(pl.col("index") == "SVI").sort("r2_residual")["outcome"].to_list()
sdi_order   = results_q3_df.filter(pl.col("index") == "SDI").sort("r2_residual")["outcome"].to_list()
INDEX_ORDERS = [readi_order, svi_order, sdi_order]

fig, axes = plt.subplots(
    len(INDEX_NAMES), 2,
    figsize=(14, max(6, 0.5 * len(readi_order)) * len(INDEX_NAMES)),
    gridspec_kw={"width_ratios": [1, 1]},
)
for row_i, idx_name in enumerate(INDEX_NAMES):
    order  = INDEX_ORDERS[row_i]
    y_pos  = np.arange(len(order))
    labels = [PLACES_LABELS.get(o, o) for o in order]
    plot_df = (
        results_q3_df.filter(pl.col("index") == idx_name)
        .with_columns(pl.col("outcome").cast(pl.Enum(order)))
        .sort("outcome")
    )
    r2_resid   = plot_df["r2_residual"].to_list()
    additional = plot_df["additional_var"].to_list()
    color      = INDEX_COLORS[idx_name]

    ax = axes[row_i, 0]
    bar_colors = [color if v >= 0 else "#9E9E9E" for v in r2_resid]
    ax.barh(y_pos, r2_resid, color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel(f"R²  of embeddings on {idx_name} residuals", fontsize=10)
    ax.set_title(f"{idx_name}: share of unexplained variance\ncaptured by embeddings",
                 fontsize=10, fontweight="bold")
    ax.set_yticks(y_pos); ax.set_yticklabels(labels, fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    ax = axes[row_i, 1]
    ax.barh(y_pos, additional, color="#43A047", alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Additional variance explained\n(residual R² × unexplained fraction)", fontsize=10)
    ax.set_title(f"{idx_name}: absolute additional variance\nexplained by embeddings",
                 fontsize=10, fontweight="bold")
    ax.set_yticks(y_pos); ax.set_yticklabels(labels, fontsize=9)
    ax.grid(axis="x", alpha=0.3)

plt.suptitle(
    f"Q3: {args.model} signal beyond each index — held-out states: {', '.join(holdout_abbrs)}",
    fontsize=11, fontweight="bold", y=1.01,
)
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "places_residual_by_index.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Q3 plot saved — {len(readi_order)} outcomes × {len(INDEX_NAMES)} indices")

# ── Q4: Heterogeneity across states and tract sizes ───────────────────────────
_q4s_cache = OUTPUTS_DIR / "q4_state.csv"
_q4d_cache = OUTPUTS_DIR / "q4_decile.csv"

HAS_ALAND = "ALAND" in df2_pd.columns

if USE_CACHED and _q4s_cache.exists() and (not HAS_ALAND or _q4d_cache.exists()):
    results_q4_state_df  = pl.read_csv(_q4s_cache)
    results_q4_decile_df = pl.read_csv(_q4d_cache) if HAS_ALAND else None
    print("Q4: loaded from cache")
else:
    X_readi_all = df2_pd[READI_FEATS].to_numpy(dtype=float)
    X_svi_all   = df2_pd[SVI_FEATS].to_numpy(dtype=float)
    X_sdi_all   = df2_pd[SDI_FEATS].to_numpy(dtype=float)
    X_emb_all   = df2_pd[FEATURE_COLS].to_numpy(dtype=float)

    def cv_r2_subset(X: np.ndarray, y: np.ndarray, n_splits: int = 2) -> float:
        if len(y) < n_splits * 5:
            return np.nan
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []
        for tr, va in kf.split(X):
            m = lgb.LGBMRegressor(
                n_estimators=30, learning_rate=0.15, num_leaves=16,
                min_child_samples=20, random_state=42, verbosity=-1,
            )
            m.fit(X[tr], y[tr])
            scores.append(r2_score(y[va], m.predict(X[va])))
        return float(np.nanmean(scores))

    # Q4a: State-wise
    print("Q4a: state-wise fits ...")
    results_q4_state = []
    for state_fips in sorted(df2_pd["state_fips"].unique()):
        state_mask = (df2_pd["state_fips"] == state_fips).to_numpy()
        if state_mask.sum() < 50:
            continue
        state_abbr = FIPS_TO_ABBR.get(state_fips, state_fips)
        r2s_readi, r2s_svi, r2s_sdi, r2s_emb = [], [], [], []
        for col in PLACES_MEASURES:
            if col not in df2_pd.columns:
                continue
            valid = state_mask & df2_pd[col].notna().to_numpy()
            if valid.sum() < 50:
                continue
            idx = np.where(valid)[0]
            y   = df2_pd[col].iloc[idx].to_numpy(dtype=float)
            r2s_readi.append(cv_r2_subset(X_readi_all[idx], y))
            r2s_svi.append(cv_r2_subset(X_svi_all[idx], y))
            r2s_sdi.append(cv_r2_subset(X_sdi_all[idx], y))
            r2s_emb.append(cv_r2_subset(X_emb_all[idx], y))
        if r2s_readi:
            results_q4_state.append({
                "state":      state_abbr,
                "r2_readi":   float(np.nanmean(r2s_readi)),
                "r2_svi":     float(np.nanmean(r2s_svi)),
                "r2_sdi":     float(np.nanmean(r2s_sdi)),
                "r2_emb":     float(np.nanmean(r2s_emb)),
                "n_outcomes": len(r2s_readi),
                "n_tracts":   int(state_mask.sum()),
            })

    results_q4_state_df = pl.DataFrame(results_q4_state)
    results_q4_state_df.write_csv(_q4s_cache)

    # Q4b: Tract-size deciles (requires ALAND)
    if HAS_ALAND:
        print("Q4b: tract-size decile fits ...")
        aland = df2_pd["ALAND"].to_numpy(dtype=float)
        _edges = np.nanpercentile(aland, np.linspace(0, 100, 11))
        _edges[0] -= 1
        decile_labels = np.searchsorted(_edges[1:-1], aland)

        results_q4_decile = []
        for dec in range(10):
            dec_mask = (decile_labels == dec)
            if dec_mask.sum() < 50:
                continue
            r2s_readi, r2s_svi, r2s_sdi, r2s_emb = [], [], [], []
            for col in PLACES_MEASURES:
                if col not in df2_pd.columns:
                    continue
                valid = dec_mask & df2_pd[col].notna().to_numpy()
                if valid.sum() < 50:
                    continue
                idx = np.where(valid)[0]
                y   = df2_pd[col].iloc[idx].to_numpy(dtype=float)
                r2s_readi.append(cv_r2_subset(X_readi_all[idx], y))
                r2s_svi.append(cv_r2_subset(X_svi_all[idx], y))
                r2s_sdi.append(cv_r2_subset(X_sdi_all[idx], y))
                r2s_emb.append(cv_r2_subset(X_emb_all[idx], y))
            if r2s_readi:
                results_q4_decile.append({
                    "decile":           dec + 1,
                    "median_aland_km2": float(np.median(aland[dec_mask]) / 1e6),
                    "r2_readi":         float(np.nanmean(r2s_readi)),
                    "r2_svi":           float(np.nanmean(r2s_svi)),
                    "r2_sdi":           float(np.nanmean(r2s_sdi)),
                    "r2_emb":           float(np.nanmean(r2s_emb)),
                    "n_tracts":         int(dec_mask.sum()),
                })

        results_q4_decile_df = pl.DataFrame(results_q4_decile)
        results_q4_decile_df.write_csv(_q4d_cache)
    else:
        results_q4_decile_df = None
        print("Q4b: skipped (ALAND not in embeddings file)")

    print("Q4: done")


s_df = results_q4_state_df.to_pandas()

n_panels = 3 + (1 if results_q4_decile_df is not None else 0)
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

def state_scatter(ax, index_col, index_label, index_color):
    ax.scatter(s_df[index_col], s_df["r2_emb"],
               s=s_df["n_tracts"] / 30, color="#2196F3",
               alpha=0.7, edgecolors="white", linewidths=0.5, zorder=3)
    for _, row in s_df.iterrows():
        ax.annotate(row["state"], (row[index_col], row["r2_emb"]),
                    fontsize=6.5, ha="center", va="bottom",
                    xytext=(0, 3), textcoords="offset points")
    lim = [min(s_df[index_col].min(), s_df["r2_emb"].min()) - 0.02,
           max(s_df[index_col].max(), s_df["r2_emb"].max()) + 0.02]
    ax.plot(lim, lim, color="grey", linewidth=1, linestyle="--", zorder=2)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(f"Avg R²  — {index_label} (within-state CV)", fontsize=10)
    ax.set_ylabel("Avg R²  — Embeddings (within-state CV)", fontsize=10)
    ax.set_title(f"{index_label} vs. Embeddings by state\n(bubble size ∝ n tracts)",
                 fontsize=10, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(alpha=0.25)

state_scatter(axes[0, 0], "r2_readi", "ReADI", "#E53935")
state_scatter(axes[0, 1], "r2_svi",   "SVI",   "#FF7043")
state_scatter(axes[1, 0], "r2_sdi",   "SDI",   "#FFA726")

ax = axes[1, 1]
if results_q4_decile_df is not None:
    d_df = results_q4_decile_df.sort("decile").to_pandas()
    ax.plot(d_df["decile"], d_df["r2_readi"], color="#E53935", marker="o", linewidth=1.8, markersize=6, label="ReADI")
    ax.plot(d_df["decile"], d_df["r2_svi"],   color="#FF7043", marker="o", linewidth=1.8, markersize=6, label="SVI")
    ax.plot(d_df["decile"], d_df["r2_sdi"],   color="#FFA726", marker="o", linewidth=1.8, markersize=6, label="SDI")
    ax.plot(d_df["decile"], d_df["r2_emb"],   color="#2196F3", marker="o", linewidth=1.8, markersize=6, label="Embeddings")
    for _, row in d_df.iterrows():
        area_str = (f"{row['median_aland_km2']:.0f} km²"
                    if row["median_aland_km2"] >= 1 else f"{row['median_aland_km2']*100:.0f} ha")
        top_r2 = max(row["r2_readi"], row["r2_svi"], row["r2_sdi"], row["r2_emb"])
        ax.annotate(area_str, (row["decile"], top_r2), fontsize=6.5, ha="center", va="bottom",
                    xytext=(0, 4), textcoords="offset points", rotation=45)
    ax.set_xticks(d_df["decile"])
    ax.set_xticklabels([f"D{d}" for d in d_df["decile"]], fontsize=9)
    ax.set_xlabel("Tract-size decile  (D1 = smallest / most urban)", fontsize=10)
    ax.set_ylabel("Avg R²  across PLACES outcomes (within-decile CV)", fontsize=10)
    ax.set_title("Performance by tract size\n(proxy for urban ↔ rural)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
else:
    ax.set_visible(False)

plt.suptitle(f"Q4: Heterogeneity of {args.model} vs. index performance",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "q4_heterogeneity.png", dpi=150, bbox_inches="tight")
plt.close()
print("Q4 plot saved")

# ── Summary ───────────────────────────────────────────────────────────────────
elapsed = time.time() - _t0
print()
print(f"=== Complete: {args.model} ===")
print(f"Elapsed : {elapsed/60:.1f} min")
print(f"Outputs : {OUTPUTS_DIR}")
