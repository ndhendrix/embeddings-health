"""
get_data.py

Pull ACS 5-year (2022) variables for all US census tracts and write to data/acs.csv.
Variables cover the inputs used by SVI, SDI, and ADI deprivation indices.

Census API serves three table families from separate endpoints:
  B/C (detail)  → /acs/acs5
  DP (profile)  → /acs/acs5/profile
  S  (subject)  → /acs/acs5/subject
"""

import os
import time
from pathlib import Path

import requests
import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

CENSUS_API_KEY = os.environ.get("CENSUS_API_KEY")
if not CENSUS_API_KEY:
    raise EnvironmentError(
        "CENSUS_API_KEY not found. Add it to a .env file in the project root."
    )

BASE = "https://api.census.gov/data"
YEAR = 2022
MAX_VARS = 49  # Census API limit is 50 per request; NAME takes one slot in the first batch

# All 50 states + DC
STATE_FIPS = [
    "01","02","04","05","06","08","09","10","11","12","13",
    "15","16","17","18","19","20","21","22","23","24","25",
    "26","27","28","29","30","31","32","33","34","35","36",
    "37","38","39","40","41","42","44","45","46","47","48",
    "49","50","51","53","54","55","56",
]

# ---------------------------------------------------------------------------
# ACS variables grouped by Census API endpoint
# ---------------------------------------------------------------------------

DETAIL_VARS = [
    # Poverty (ADI)
    "B17021_001E", "B17021_002E", "B17021_003E",
    # Income (ADI)
    "B19001_002E", "B19001_017E",
    "B19113_001E",   # median family income
    "B19301_001E",   # per capita income
    # Employment (ADI)
    "B23025_003E", "B23025_005E",
    # Educational attainment (SDI / ADI)
    "B15003_001E",
    "B15003_002E",  "B15003_003E",  "B15003_004E",  "B15003_005E",
    "B15003_006E",  "B15003_007E",  "B15003_008E",  "B15003_009E",
    "B15003_010E",  "B15003_011E",  "B15003_012E",  "B15003_013E",
    "B15003_014E",  "B15003_015E",  "B15003_016E",  # < high school
    "B15003_017E",  "B15003_018E",  "B15003_019E",  "B15003_020E",
    "B15003_021E",  "B15003_022E",  "B15003_023E",  "B15003_024E",
    "B15003_025E",  # high school and above
    # Language (SVI limited English)
    "B16005_001E",
    "B16005_007E",  "B16005_008E",  "B16005_012E",  "B16005_013E",
    "B16005_017E",  "B16005_018E",  "B16005_022E",  "B16005_023E",
    "B16005_029E",  "B16005_030E",  "B16005_034E",  "B16005_035E",
    "B16005_039E",  "B16005_040E",  "B16005_044E",  "B16005_045E",
    # Occupation (ADI white-collar)
    "C24010_001E",
    "C24010_019E",  "C24010_020E",  "C24010_021E",  "C24010_022E",
    "C24010_023E",  "C24010_024E",  "C24010_025E",
    "C24010_039E",  "C24010_040E",  "C24010_041E",  "C24010_042E",
    "C24010_043E",  "C24010_044E",  "C24010_045E",
    # Housing (ADI)
    "B25064_001E",   # median gross rent
    "B25077_001E",   # median home value
    "B25003_001E",   "B25003_002E",   # tenure (owner-occupied)
    "B25014_001E",   "B25014_005E",   "B25014_006E",   "B25014_007E",  # crowding
    "B25040_001E",   "B25040_003E",   # no telephone
    "B25044_001E",   "B25044_003E",   "B25044_010E",   # no vehicle
    "B25047_001E",   "B25047_002E",   # no plumbing
    # Group quarters (SVI)
    "B26001_001E",
]

PROFILE_VARS = [
    # Employment (SVI)
    "DP03_0009PE",
    # Age / population (SVI)
    "DP05_0001E",    # total population
    "DP05_0019PE",   # age ≤17 %
    "DP05_0079E",    # NH White alone (for minority calculation)
    # Household structure / disability (SVI)
    "DP02_0001E",    # total households
    "DP02_0007PE",   # single-parent (male HH, no wife)
    "DP02_0011PE",   # single-parent (female HH, no husband)
    "DP02_0072PE",   # disability %
    # Housing characteristics (SVI / SDI)
    "DP04_0012PE",   # structures with 10–19 units
    "DP04_0013PE",   # structures with 20+ units
    "DP04_0014PE",   # mobile homes
    "DP04_0047PE",   # renter-occupied %
    "DP04_0058PE",   # no vehicle
    "DP04_0078PE",   # overcrowded (owner)
    "DP04_0079PE",   # overcrowded (renter)
]

SUBJECT_VARS = [
    # Poverty (SVI / SDI)
    "S1701_C01_001E",   # poverty determination universe
    "S1701_C01_040E",   # below 150% poverty (SVI)
    "S1701_C02_001E",   # below 100% poverty (SDI)
    # Education (SVI)
    "S0601_C01_001E",   # total population 25+
    "S0601_C01_033E",   # no high school diploma %
    # Housing cost burden (SVI)
    "S2503_C01_001E",   # renter-occupied unit universe
    "S2503_C01_028E",   # rent 30–34.9% of income
    "S2503_C01_032E",   # rent 35–39.9%
    "S2503_C01_036E",   # rent 40–49.9%
    "S2503_C01_040E",   # rent ≥50%
    # Health insurance (SVI)
    "S2701_C05_001E",   # uninsured %
    # Age (SVI)
    "S0101_C02_030E",   # age 65+ %
    # Employment (SDI nonemployed proxy)
    "S2301_C04_001E",
]

# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

def _fetch_batch(
    vars_batch: list[str],
    state: str,
    dataset: str,
    include_name: bool = False,
) -> pd.DataFrame:
    """Make one Census API request and return a DataFrame keyed by GEOID."""
    get_vars = (["NAME"] + vars_batch) if include_name else vars_batch
    params = {
        "get": ",".join(get_vars),
        "for": "tract:*",
        "in": f"state:{state} county:*",
        "key": CENSUS_API_KEY,
    }
    r = requests.get(f"{BASE}/{YEAR}/{dataset}", params=params, timeout=60)
    r.raise_for_status()
    rows = r.json()
    df = pd.DataFrame(rows[1:], columns=rows[0])

    for col in df.columns:
        if col not in {"NAME", "state", "county", "tract"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["GEOID"] = df["state"] + df["county"] + df["tract"]
    return df.drop(columns=["state", "county", "tract"])


def _fetch_table(
    vars_list: list[str],
    state: str,
    dataset: str,
    include_name: bool = False,
) -> pd.DataFrame | None:
    """Fetch all variables in vars_list, chunking to respect the 50-var API limit."""
    if not vars_list:
        return None

    chunks = [vars_list[i : i + MAX_VARS] for i in range(0, len(vars_list), MAX_VARS)]
    frames = []
    for i, chunk in enumerate(chunks):
        df = _fetch_batch(chunk, state, dataset, include_name=(include_name and i == 0))
        frames.append(df)
        if i < len(chunks) - 1:
            time.sleep(0.1)

    result = frames[0]
    for df in frames[1:]:
        result = result.merge(df, on="GEOID", how="outer")
    return result


def fetch_state(state: str) -> pd.DataFrame:
    """Pull all ACS variables for all census tracts in one state."""
    endpoints = [
        ("acs/acs5",         DETAIL_VARS,  True),   # NAME in first batch of detail
        ("acs/acs5/profile", PROFILE_VARS, False),
        ("acs/acs5/subject", SUBJECT_VARS, False),
    ]

    frames = [
        _fetch_table(vars_list, state, dataset, include_name=with_name)
        for dataset, vars_list, with_name in endpoints
    ]
    frames = [f for f in frames if f is not None]

    result = frames[0]
    for df in frames[1:]:
        result = result.merge(df, on="GEOID", how="outer")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def pull_all_tracts() -> pd.DataFrame:
    """Pull ACS data for all census tracts in the US (50 states + DC)."""
    all_frames = []
    for state in STATE_FIPS:
        print(f"  state {state}...", flush=True)
        all_frames.append(fetch_state(state))
    return pd.concat(all_frames, ignore_index=True)


if __name__ == "__main__":
    out_path = Path(__file__).parents[2] / "data" / "acs.csv"
    print(f"Pulling ACS {YEAR} 5-year data for all US census tracts...")
    df = pull_all_tracts()
    df.to_csv(out_path, index=False)
    print(f"Done. {len(df):,} tracts → {out_path}")
