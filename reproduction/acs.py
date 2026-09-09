"""Paper ACS transformations, extracted from analyses_sherlock.py. No index values."""
import polars as pl

def prepare_acs(ACS_CSV):
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
            pl.read_parquet(ACS_CSV).with_columns(pl.exclude(["GEOID", "NAME"]).cast(pl.Float64, strict=True))
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
    
    return acs
