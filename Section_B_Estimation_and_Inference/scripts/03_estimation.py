"""
Task 3 — Parameter estimation (Section B).

Fits the baseline OLS model for the Communities and Crime project using the
shared baseline dataset produced in Section A.

What this script does:
    1. Load ../../data/communities_baseline_model.csv.
    2. Verify the 7 baseline predictors and target exist.
    3. Fit an OLS model with an intercept using statsmodels.
    4. Save a coefficient table (coef, SE, t, p, 95% CI) to
       outputs/tables/coefficients.csv.
    5. Save a compact model-fit table to outputs/tables/model_fit_summary.csv.
    6. Save the full statsmodels text summary to outputs/tables/ols_summary.txt.

Run from project root:
    python Section_B_Estimation_and_Inference/scripts/03_estimation.py
"""

from pathlib import Path

import pandas as pd
import statsmodels.api as sm


# ---------------------------------------------------------------------------
# Paths.
# Data lives at the repo root (shared across sections). Outputs live inside
# this section's folder.
#   parents[0] = .../scripts
#   parents[1] = .../Section_B_Estimation_and_Inference
#   parents[2] = .../<repo root>
# ---------------------------------------------------------------------------
SECTION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "communities_baseline_model.csv"
TABLES_DIR = SECTION_ROOT / "outputs" / "tables"
COEFFICIENTS_PATH = TABLES_DIR / "coefficients.csv"
MODEL_FIT_PATH = TABLES_DIR / "model_fit_summary.csv"
SUMMARY_TEXT_PATH = TABLES_DIR / "ols_summary.txt"


# ---------------------------------------------------------------------------
# Baseline model definition inherited from Section A.
# ---------------------------------------------------------------------------
PREDICTORS = [
    "PctPopUnderPov",
    "PctNotHSGrad",
    "PctUnemployed",
    "PctKids2Par",
    "PctVacantBoarded",
    "PopDens",
    "population",
]
TARGET = "ViolentCrimesPerPop"


def load_baseline_data() -> pd.DataFrame:
    """Load the shared baseline modelling dataset."""
    print(f"Loading baseline data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"  Raw shape: {df.shape}")

    needed = PREDICTORS + [TARGET]
    missing_columns = [col for col in needed if col not in df.columns]
    assert not missing_columns, f"Expected columns not found in baseline data: {missing_columns}"

    missing_rows = df[needed].isna().any(axis=1).sum()
    if missing_rows:
        print(f"  WARNING: dropping {missing_rows} row(s) with missing values in model columns.")

    model_df = df[needed].dropna().copy()
    print(f"  Modelling shape after NA check: {model_df.shape}")
    return model_df


def fit_ols(model_df: pd.DataFrame):
    """Fit the baseline OLS model with an intercept."""
    X = sm.add_constant(model_df[PREDICTORS], has_constant="add")
    y = model_df[TARGET]
    model = sm.OLS(y, X).fit()
    return model


def build_coefficient_table(model) -> pd.DataFrame:
    """Return coefficient estimates and standard inferential quantities."""
    conf_int = model.conf_int(alpha=0.05)
    table = pd.DataFrame(
        {
            "term": model.params.index,
            "coef": model.params.values,
            "std_err": model.bse.values,
            "t_stat": model.tvalues.values,
            "p_value": model.pvalues.values,
            "ci_lower_95": conf_int.iloc[:, 0].values,
            "ci_upper_95": conf_int.iloc[:, 1].values,
        }
    )
    return table


def build_model_fit_table(model) -> pd.DataFrame:
    """Return a compact one-row summary of overall model fit."""
    fit_table = pd.DataFrame(
        [
            {
                "n_obs": int(model.nobs),
                "n_parameters_including_intercept": int(len(model.params)),
                "df_model": float(model.df_model),
                "df_resid": float(model.df_resid),
                "r_squared": float(model.rsquared),
                "adj_r_squared": float(model.rsquared_adj),
                "f_statistic": float(model.fvalue),
                "f_pvalue": float(model.f_pvalue),
                "aic": float(model.aic),
                "bic": float(model.bic),
                "rmse": float(model.mse_resid ** 0.5),
            }
        ]
    )
    return fit_table


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    model_df = load_baseline_data()
    model = fit_ols(model_df)

    coefficient_table = build_coefficient_table(model)
    fit_table = build_model_fit_table(model)

    coefficient_table.to_csv(COEFFICIENTS_PATH, index=False)
    fit_table.to_csv(MODEL_FIT_PATH, index=False)
    SUMMARY_TEXT_PATH.write_text(model.summary().as_text(), encoding="utf-8")

    print(f"\nSaved coefficient table -> {COEFFICIENTS_PATH.relative_to(REPO_ROOT)}")
    print(f"Saved fit summary      -> {MODEL_FIT_PATH.relative_to(REPO_ROOT)}")
    print(f"Saved OLS text summary -> {SUMMARY_TEXT_PATH.relative_to(REPO_ROOT)}")

    print("\n=== Baseline OLS estimation summary ===")
    print(coefficient_table.round(4).to_string(index=False))
    print("\nOverall fit:")
    print(fit_table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
