"""
Task 4 — Hypothesis Testing and Confidence Intervals (Section B).

This script is designed to minimise overlap with Task 3.

What it does:
    1. Reads Task 3's outputs/tables/coefficients.csv for:
       - individual t-test results
       - 95% confidence intervals
    2. Fits the same baseline OLS model only to run joint F-tests,
       since joint F-tests cannot be recovered from coefficients.csv alone.
    3. Saves Task 4-specific inference outputs to outputs/tables/:
       - t_test_decisions.csv
       - confidence_interval_interpretations.csv
       - joint_f_tests.csv
       - task4_inference_summary.txt

Run from project root:
    python Section_B_Estimation_and_Inference/scripts/04_inference.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SECTION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = REPO_ROOT / "data" / "communities_baseline_model.csv"
TABLES_DIR = SECTION_ROOT / "outputs" / "tables"

# Task 3 output reused here to avoid overlap.
COEFFICIENTS_PATH = TABLES_DIR / "coefficients.csv"

# Task 4 outputs.
T_TESTS_PATH = TABLES_DIR / "t_test_decisions.csv"
CI_PATH = TABLES_DIR / "confidence_interval_interpretations.csv"
F_TESTS_PATH = TABLES_DIR / "joint_f_tests.csv"
SUMMARY_PATH = TABLES_DIR / "task4_inference_summary.txt"


# ---------------------------------------------------------------------------
# Baseline model specification
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

READABLE_NAMES = {
    "const": "intercept",
    "PctPopUnderPov": "poverty rate",
    "PctNotHSGrad": "share without a high-school diploma",
    "PctUnemployed": "unemployment rate",
    "PctKids2Par": "share of kids in two-parent households",
    "PctVacantBoarded": "share of vacant boarded housing",
    "PopDens": "population density",
    "population": "population size",
}


def load_task3_coefficients() -> pd.DataFrame:
    """Load Task 3 coefficient output so Task 4 does not duplicate that work."""
    if not COEFFICIENTS_PATH.exists():
        raise FileNotFoundError(
            f"Task 3 output not found: {COEFFICIENTS_PATH}\n"
            "Run 03_estimation.py first so Task 4 can reuse coefficients.csv."
        )

    coeffs = pd.read_csv(COEFFICIENTS_PATH)

    required_cols = {
        "term",
        "coef",
        "std_err",
        "t_stat",
        "p_value",
        "ci_lower_95",
        "ci_upper_95",
    }
    missing = required_cols.difference(coeffs.columns)
    if missing:
        raise ValueError(
            f"coefficients.csv is missing required columns: {sorted(missing)}"
        )

    return coeffs.copy()


def load_baseline_data() -> pd.DataFrame:
    """Load the shared baseline dataset for joint F-tests."""
    df = pd.read_csv(DATA_PATH)

    needed = PREDICTORS + [TARGET]
    missing_columns = [col for col in needed if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Expected columns not found in baseline data: {missing_columns}")

    df[needed] = df[needed].apply(pd.to_numeric, errors="coerce")
    model_df = df[needed].dropna().copy()
    return model_df


def fit_ols_for_f_tests(model_df: pd.DataFrame):
    """Fit the same OLS model used in Task 3, but only for joint F-tests."""
    X = sm.add_constant(model_df[PREDICTORS], has_constant="add")
    y = model_df[TARGET]
    return sm.OLS(y, X).fit()


def build_t_test_table(coeffs: pd.DataFrame) -> pd.DataFrame:
    """Create a Task 4 table focused only on individual t-test decisions."""
    out = coeffs[["term", "coef", "t_stat", "p_value"]].copy()
    out["readable_name"] = out["term"].map(READABLE_NAMES).fillna(out["term"])
    out["null_hypothesis"] = "H0: coefficient = 0"
    out["significant_at_5pct"] = out["p_value"] < 0.05
    out["decision"] = np.where(
        out["significant_at_5pct"],
        "Reject H0",
        "Fail to reject H0",
    )

    interpretations = []
    for _, row in out.iterrows():
        if row["term"] == "const":
            text = (
                f"The intercept has t = {row['t_stat']:.4f} and p = {row['p_value']:.4f}. "
                f"At the 5% level, we {row['decision'].lower()} that the intercept equals zero."
            )
        else:
            direction = "positive" if row["coef"] > 0 else "negative"
            text = (
                f"The coefficient on {row['readable_name']} is {direction} "
                f"(t = {row['t_stat']:.4f}, p = {row['p_value']:.4f}). "
                f"At the 5% level, we {row['decision'].lower()} that this coefficient equals zero."
            )
        interpretations.append(text)

    out["interpretation"] = interpretations

    cols = [
        "term",
        "readable_name",
        "coef",
        "t_stat",
        "p_value",
        "null_hypothesis",
        "significant_at_5pct",
        "decision",
        "interpretation",
    ]
    return out[cols]


def build_ci_table(coeffs: pd.DataFrame) -> pd.DataFrame:
    """Create a Task 4 table focused only on 95% CI interpretation."""
    out = coeffs[["term", "coef", "ci_lower_95", "ci_upper_95"]].copy()
    out["readable_name"] = out["term"].map(READABLE_NAMES).fillna(out["term"])
    out["contains_zero"] = (
        (out["ci_lower_95"] <= 0) & (out["ci_upper_95"] >= 0)
    )
    out["excludes_zero"] = ~out["contains_zero"]

    interpretations = []
    for _, row in out.iterrows():
        lower = row["ci_lower_95"]
        upper = row["ci_upper_95"]

        if row["contains_zero"]:
            meaning = (
                "Because 0 lies inside the interval, the coefficient is not statistically "
                "different from 0 at the 5% level."
            )
        else:
            meaning = (
                "Because 0 is outside the interval, the coefficient is statistically "
                "different from 0 at the 5% level."
            )

        if row["term"] == "const":
            text = (
                f"The 95% confidence interval for the intercept is "
                f"[{lower:.4f}, {upper:.4f}]. {meaning}"
            )
        else:
            text = (
                f"The 95% confidence interval for the coefficient on "
                f"{row['readable_name']} is [{lower:.4f}, {upper:.4f}]. {meaning}"
            )
        interpretations.append(text)

    out["interpretation"] = interpretations

    cols = [
        "term",
        "readable_name",
        "coef",
        "ci_lower_95",
        "ci_upper_95",
        "contains_zero",
        "excludes_zero",
        "interpretation",
    ]
    return out[cols]


def run_single_f_test(model, test_name: str, restriction: str, null_hypothesis: str) -> dict:
    """Run one joint F-test and return a clean result dictionary."""
    test = model.f_test(restriction)

    f_value = float(np.asarray(test.fvalue).squeeze())
    p_value = float(np.asarray(test.pvalue).squeeze())
    df_num = float(np.asarray(test.df_num).squeeze())
    df_denom = float(np.asarray(test.df_denom).squeeze())

    reject = p_value < 0.05
    decision = "Reject H0" if reject else "Fail to reject H0"
    interpretation = (
        f"{decision} at the 5% level. The variables in this group are jointly significant."
        if reject
        else f"{decision} at the 5% level. The variables in this group are not jointly significant."
    )

    return {
        "test_name": test_name,
        "null_hypothesis": null_hypothesis,
        "restriction": restriction,
        "f_statistic": f_value,
        "p_value": p_value,
        "df_num": df_num,
        "df_denom": df_denom,
        "reject_at_5pct": reject,
        "decision": decision,
        "interpretation": interpretation,
    }


def build_joint_f_tests_table(model) -> pd.DataFrame:
    """Run meaningful grouped F-tests with minimal overlap with Task 3."""
    tests = [
        (
            "disadvantage_block",
            "PctPopUnderPov = 0, PctNotHSGrad = 0, PctUnemployed = 0, PctKids2Par = 0",
            "H0: the disadvantage/family-structure coefficients are jointly equal to zero."
        ),
        (
            "housing_urban_scale_block",
            "PctVacantBoarded = 0, PopDens = 0, population = 0",
            "H0: the housing/urban-scale coefficients are jointly equal to zero."
        ),
    ]

    rows = [
        run_single_f_test(model, test_name, restriction, null_hypothesis)
        for test_name, restriction, null_hypothesis in tests
    ]
    return pd.DataFrame(rows)


def build_text_summary(t_table: pd.DataFrame, ci_table: pd.DataFrame, f_table: pd.DataFrame) -> str:
    """Create a compact text summary for the report."""
    lines = []
    lines.append("SECTION B - TASK 4: INFERENCE SUMMARY")
    lines.append("=" * 60)
    lines.append("")
    lines.append("INDIVIDUAL t-TESTS")
    lines.append("-" * 60)
    for _, row in t_table.iterrows():
        lines.append(
            f"{row['term']}: t = {row['t_stat']:.4f}, p = {row['p_value']:.4f} -> {row['decision']}"
        )

    lines.append("")
    lines.append("95% CONFIDENCE INTERVALS")
    lines.append("-" * 60)
    for _, row in ci_table.iterrows():
        zero_text = "contains 0" if row["contains_zero"] else "excludes 0"
        lines.append(
            f"{row['term']}: [{row['ci_lower_95']:.4f}, {row['ci_upper_95']:.4f}] -> {zero_text}"
        )

    lines.append("")
    lines.append("JOINT F-TESTS")
    lines.append("-" * 60)
    for _, row in f_table.iterrows():
        lines.append(
            f"{row['test_name']}: F = {row['f_statistic']:.4f}, "
            f"p = {row['p_value']:.4f}, df = ({row['df_num']:.0f}, {row['df_denom']:.0f}) "
            f"-> {row['decision']}"
        )

    return "\n".join(lines)


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Reuse Task 3 outputs for individual inference to minimise overlap.
    coeffs = load_task3_coefficients()
    t_table = build_t_test_table(coeffs)
    ci_table = build_ci_table(coeffs)

    # Refit only because grouped F-tests require the actual model object.
    model_df = load_baseline_data()
    model = fit_ols_for_f_tests(model_df)
    f_table = build_joint_f_tests_table(model)

    summary_text = build_text_summary(t_table, ci_table, f_table)

    t_table.to_csv(T_TESTS_PATH, index=False)
    ci_table.to_csv(CI_PATH, index=False)
    f_table.to_csv(F_TESTS_PATH, index=False)
    SUMMARY_PATH.write_text(summary_text, encoding="utf-8")

    print(f"Saved t-test decisions         -> {T_TESTS_PATH.relative_to(REPO_ROOT)}")
    print(f"Saved CI interpretations       -> {CI_PATH.relative_to(REPO_ROOT)}")
    print(f"Saved joint F-tests            -> {F_TESTS_PATH.relative_to(REPO_ROOT)}")
    print(f"Saved Task 4 text summary      -> {SUMMARY_PATH.relative_to(REPO_ROOT)}")

    print("\n=== Task 4: Individual t-test decisions ===")
    print(
        t_table[
            ["term", "t_stat", "p_value", "significant_at_5pct", "decision"]
        ].round(4).to_string(index=False)
    )

    print("\n=== Task 4: 95% confidence intervals ===")
    print(
        ci_table[
            ["term", "ci_lower_95", "ci_upper_95", "contains_zero", "excludes_zero"]
        ].round(4).to_string(index=False)
    )

    print("\n=== Task 4: Joint F-tests ===")
    print(
        f_table[
            ["test_name", "f_statistic", "p_value", "df_num", "df_denom", "decision"]
        ].round(4).to_string(index=False)
    )


if __name__ == "__main__":
    main()