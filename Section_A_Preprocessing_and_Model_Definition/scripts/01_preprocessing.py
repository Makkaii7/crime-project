"""
Task 1 — Preprocessing the UCI Communities and Crime dataset.

What this script does, step by step:
    1. Load the raw dataset (data/communities_with_headers.csv).
    2. Drop 23 police-related columns. 22 are ~84% missing; the 23rd
       (`LemasPctOfficDrugUn`) has no missing values but is zero for all
       communities without LEMAS data, so its variation mostly encodes the
       missingness pattern — a subtle form of leakage we avoid by dropping it.
    3. Impute the single missing value in `OtherPerCap` with the column median.
    4. Save a MASTER dataset (data/communities_master.csv) that keeps the 5
       ID / administrative columns (state, county, community, communityname,
       fold). These are NOT predictors but are needed for later work:
         - `fold` drives the pre-made 10-fold CV in Section C (Task 5).
         - `communityname`, `county`, `community` let us trace outliers back
           to specific communities in diagnostics (Task 6).
         - `state` could be used as fixed effects in a robustness extension.
    5. Save a MODELLING dataset (data/communities_clean.csv) = master minus
       the 5 ID columns. This is what feeds Task 2.
    6. Verify no missing values remain, master shape is (1994, 105) and
       clean shape is (1994, 100).
    7. Print a short human-readable summary.

Notes on the master file:
    - It contains the imputed `OtherPerCap`, so it is NOT a verbatim copy of
      the raw CSV — it is the cleaned data with IDs retained.
    - The raw UCI data has ~1,175 missing values in each of `county` and
      `community` (withheld for confidentiality). We deliberately do NOT
      impute these — they are not predictors, and `communityname` + `state`
      together already identify a row for outlier traceback. So the master
      file has missing values only in those two columns.
    - The modelling (`clean`) file has ZERO missing values — that's the
      load-bearing invariant.

Run from project root:
    python scripts/01_preprocessing.py
"""

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Paths — resolve relative to this file so the script works from any cwd.
# parents[2] climbs: script file -> scripts/ -> Section_A_.../  -> repo root.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "communities_with_headers.csv"
MASTER_PATH = PROJECT_ROOT / "data" / "communities_master.csv"
CLEAN_PATH = PROJECT_ROOT / "data" / "communities_clean.csv"


# ---------------------------------------------------------------------------
# Column groups.
# ---------------------------------------------------------------------------
# 23 police / law-enforcement columns.
#
# 22 of these are ~84% missing because the underlying LEMAS survey only covered
# a subset of communities. Dropping them is justified in the report (non-MCAR
# with informative missingness, 10-20 obs-per-predictor rule, imputing would
# fabricate data for ~84% of rows).
#
# The 23rd column, `LemasPctOfficDrugUn`, has NO missing values but is zero
# for all 1,675 communities without LEMAS data. Its variation therefore
# largely encodes "did this community have police data?" rather than a real
# police-behaviour measurement. Leaving it in would be a subtle form of
# leakage through the missingness pattern, so we drop it with the other 22.
POLICE_COLS = [
    "LemasSwornFT",
    "LemasSwFTPerPop",
    "LemasSwFTFieldOps",
    "LemasSwFTFieldPerPop",
    "LemasTotalReq",
    "LemasTotReqPerPop",
    "PolicReqPerOffic",
    "PolicPerPop",
    "RacialMatchCommPol",  # note: UCI column name is RacialMatchCommPol
    "PctPolicWhite",
    "PctPolicBlack",
    "PctPolicHisp",
    "PctPolicAsian",
    "PctPolicMinor",
    "OfficAssgnDrugUnits",
    "NumKindsDrugsSeiz",
    "PolicAveOTWorked",
    "PolicCars",
    "PolicOperBudg",
    "LemasPctPolicOnPatr",
    "LemasGangUnitDeploy",
    "LemasPctOfficDrugUn",
    "PolicBudgPerPop",
]

# 5 identifier / administrative columns.
# Kept in the MASTER dataset (for diagnostics, CV, robustness extensions)
# but removed from the MODELLING dataset (they are not predictors).
ID_COLS = ["state", "county", "community", "communityname", "fold"]


def main() -> None:
    # -----------------------------------------------------------------------
    # Step 1 — Load the raw CSV.
    # -----------------------------------------------------------------------
    print(f"Loading raw data from: {RAW_PATH}")
    df = pd.read_csv(RAW_PATH)
    raw_shape = df.shape
    print(f"  Raw shape: {raw_shape}  (rows, columns)")

    # -----------------------------------------------------------------------
    # Step 2 — Drop the 23 police-related columns.
    # -----------------------------------------------------------------------
    # Sanity check: every listed police column should actually be in the file.
    # If any are missing, that's a schema mismatch worth knowing about.
    missing_police = [c for c in POLICE_COLS if c not in df.columns]
    assert not missing_police, f"Expected police cols not found: {missing_police}"
    df = df.drop(columns=POLICE_COLS)
    print(f"  Dropped {len(POLICE_COLS)} police columns -> shape {df.shape}")

    # -----------------------------------------------------------------------
    # Step 3 — Impute the single missing value in OtherPerCap with the median.
    # -----------------------------------------------------------------------
    # Median is chosen over the mean because it is robust to outliers, and
    # with only one missing value the choice barely moves the distribution.
    n_missing_other = int(df["OtherPerCap"].isna().sum())
    print(f"  Missing values in OtherPerCap before imputation: {n_missing_other}")
    other_median = df["OtherPerCap"].median()
    df["OtherPerCap"] = df["OtherPerCap"].fillna(other_median)
    print(f"  Imputed OtherPerCap with median = {other_median:.4f}")

    # -----------------------------------------------------------------------
    # Step 4 — Save the MASTER dataset (keeps the 5 ID columns).
    # -----------------------------------------------------------------------
    # The master file is the cleaned data with IDs retained. It reflects the
    # imputed OtherPerCap, so it is NOT a copy of the raw CSV. Missing values
    # are permitted ONLY in `county` and `community` (withheld for
    # confidentiality in the raw UCI data) — not in any predictor or target.
    assert all(c in df.columns for c in ID_COLS), "ID columns missing before master save."
    assert df.shape == (1994, 105), f"Expected master shape (1994, 105), got {df.shape}."
    allowed_missing = {"county", "community"}
    cols_with_missing = set(df.columns[df.isna().any()].tolist())
    unexpected = cols_with_missing - allowed_missing
    assert not unexpected, f"Unexpected missing values in: {sorted(unexpected)}"
    master_missing = int(df.isna().sum().sum())
    print(
        f"  Master has {master_missing} missing values "
        f"(only in county/community — retained as-is)."
    )
    df.to_csv(MASTER_PATH, index=False)
    print(f"  Saved master (incl. IDs) -> {MASTER_PATH}  shape {df.shape}")

    # -----------------------------------------------------------------------
    # Step 5 — Build and save the MODELLING dataset (drops the 5 ID columns).
    # -----------------------------------------------------------------------
    df_clean = df.drop(columns=ID_COLS)
    print(f"  Dropped {len(ID_COLS)} ID columns for modelling set -> shape {df_clean.shape}")

    # -----------------------------------------------------------------------
    # Step 6 — Verify no missing values remain and shapes are as expected.
    # -----------------------------------------------------------------------
    total_missing = int(df_clean.isna().sum().sum())
    print(f"  Total missing values after cleaning: {total_missing}")
    assert total_missing == 0, "Expected zero missing values after cleaning."
    assert df_clean.shape == (1994, 100), (
        f"Expected clean shape (1994, 100), got {df_clean.shape}."
    )

    df_clean.to_csv(CLEAN_PATH, index=False)
    print(f"  Saved clean (modelling set) -> {CLEAN_PATH}  shape {df_clean.shape}")

    # -----------------------------------------------------------------------
    # Step 7 — Short summary.
    # -----------------------------------------------------------------------
    print("\n=== Preprocessing summary ===")
    print(f"Raw shape:                {raw_shape}")
    print(f"Dropped police columns:   {len(POLICE_COLS)}  (incl. LemasPctOfficDrugUn)")
    print(f"Imputed (median):         OtherPerCap ({n_missing_other} value)")
    print(f"Master shape (w/ IDs):    {df.shape}")
    print(f"Clean shape (no IDs):     {df_clean.shape}")
    print(f"Missing values (clean):   {total_missing}")
    print(f"Master file:              {MASTER_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Clean file:               {CLEAN_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
