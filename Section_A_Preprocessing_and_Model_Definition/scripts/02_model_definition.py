"""
Task 2 — Model definition (BASELINE shortlist).

IMPORTANT framing: this script picks a BASELINE SHORTLIST of 7 predictors,
one per major construct (theme). It is NOT the final model. Formal model
selection using BIC / AIC / cross-validation is performed in Task 5
(Section C). Raw univariate correlation is not a sufficient final-selection
criterion because it ignores multivariate redundancy and collinearity.

What this script does:
    1. Load data/communities_clean.csv (output of Task 1).
    2. Define the 7 baseline predictors and the target.
    3. Build the 8-column modelling subset (7 predictors + target).
    4. Compute univariate Pearson correlations with the target and compare
       them against the values asserted in the report; flag any material
       disagreement.
    5. Save a correlation heatmap (8x8) to outputs/plots/correlation_heatmap.png.
    6. Save pairwise scatter plots (predictor vs target) to
       outputs/plots/scatter_plots.png.
    7. Save summary statistics (mean, std, min, max) to
       outputs/tables/summary_stats.csv.
    8. Save the baseline modelling dataset to
       data/communities_baseline_model.csv.

Run from project root:
    python scripts/02_model_definition.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Paths.
# Data lives at the repo root (shared across sections). Outputs live inside
# this section's folder.
#   parents[0] = .../scripts
#   parents[1] = .../Section_A_Preprocessing_and_Model_Definition
#   parents[2] = .../<repo root>
# ---------------------------------------------------------------------------
SECTION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
CLEAN_PATH = REPO_ROOT / "data" / "communities_clean.csv"
BASELINE_PATH = REPO_ROOT / "data" / "communities_baseline_model.csv"
HEATMAP_PATH = SECTION_ROOT / "outputs" / "plots" / "correlation_heatmap.png"
SCATTER_PATH = SECTION_ROOT / "outputs" / "plots" / "scatter_plots.png"
SUMMARY_PATH = SECTION_ROOT / "outputs" / "tables" / "summary_stats.csv"


# ---------------------------------------------------------------------------
# Baseline shortlist: one predictor per construct.
# ---------------------------------------------------------------------------
# The constructs are chosen to cover the main drivers hypothesised in the
# crime-and-communities literature without duplicating information. Within
# each construct we picked the variable with the strongest target correlation
# while sanity-checking for redundancy against variables already in the list.
CONSTRUCTS = [
    # (construct label,         variable name,       sign note)
    ("Socioeconomic disadvantage", "PctPopUnderPov",   "positive"),
    ("Education",                  "PctNotHSGrad",     "positive"),
    ("Employment",                 "PctUnemployed",    "positive"),
    ("Family disruption",          "PctKids2Par",      "negative"),
    ("Housing disorder",           "PctVacantBoarded", "positive"),
    ("Urban scale",                "PopDens",          "positive"),
    ("Community size",             "population",       "positive"),
]

PREDICTORS = [v for _, v, _ in CONSTRUCTS]
TARGET = "ViolentCrimesPerPop"

# Correlations asserted in the report (for a sanity check against the data).
# If any observed |diff| > 0.01 we print a warning so we can reconcile the
# report numbers with the actual data rather than letting stale numbers ship.
REPORTED_CORRELATIONS = {
    "PctPopUnderPov":   +0.522,
    "PctNotHSGrad":     +0.483,
    "PctUnemployed":    +0.504,
    "PctKids2Par":      -0.738,
    "PctVacantBoarded": +0.483,
    "PopDens":          +0.281,
    "population":       +0.367,
}


def main() -> None:
    # -----------------------------------------------------------------------
    # Step 1 — Load the cleaned dataset.
    # -----------------------------------------------------------------------
    print(f"Loading clean data from: {CLEAN_PATH}")
    df = pd.read_csv(CLEAN_PATH)
    print(f"  Clean shape: {df.shape}")

    # -----------------------------------------------------------------------
    # Step 2 — Verify all baseline columns exist before subsetting.
    # -----------------------------------------------------------------------
    needed = PREDICTORS + [TARGET]
    missing = [c for c in needed if c not in df.columns]
    assert not missing, f"Expected columns not found in clean data: {missing}"

    # -----------------------------------------------------------------------
    # Step 3 — Build the 8-column modelling subset.
    # -----------------------------------------------------------------------
    model_df = df[needed].copy()
    print(f"  Baseline subset shape: {model_df.shape}  (7 predictors + 1 target)")

    # -----------------------------------------------------------------------
    # Step 4 — Sanity-check correlations against the report.
    # -----------------------------------------------------------------------
    observed = model_df.corr(method="pearson")[TARGET].drop(TARGET)
    print("\n  Univariate Pearson correlations with target:")
    flagged = []
    for var in PREDICTORS:
        obs = observed[var]
        rep = REPORTED_CORRELATIONS[var]
        diff = obs - rep
        marker = ""
        if abs(diff) > 0.01:
            marker = "  <-- REPORT VALUE DIFFERS BY >0.01"
            flagged.append((var, obs, rep, diff))
        print(f"    {var:<18s}  observed={obs:+.3f}   report={rep:+.3f}   diff={diff:+.3f}{marker}")
    if flagged:
        print(
            "\n  WARNING: the report's correlation values disagree with the data "
            "for the variables flagged above. Reconcile before the final write-up."
        )

    # -----------------------------------------------------------------------
    # Step 5 — Correlation heatmap (8x8: 7 predictors + target).
    # -----------------------------------------------------------------------
    HEATMAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    corr = model_df.corr(method="pearson")

    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Correlation matrix — 7 baseline predictors + target", pad=10)
    plt.tight_layout()
    fig.savefig(HEATMAP_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved heatmap -> {HEATMAP_PATH.relative_to(REPO_ROOT)}")

    # -----------------------------------------------------------------------
    # Step 6 — Pairwise scatter plots: each predictor vs target.
    # -----------------------------------------------------------------------
    SCATTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    # 7 predictors -> 2 rows x 4 cols grid; last cell is left blank.
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharey=True)
    axes_flat = axes.flatten()
    for ax, var in zip(axes_flat, PREDICTORS):
        ax.scatter(model_df[var], model_df[TARGET], alpha=0.35, s=12, edgecolor="none")
        r = observed[var]
        ax.set_title(f"{var}  (r = {r:+.2f})", fontsize=10)
        ax.set_xlabel(var)
        ax.set_ylabel(TARGET)
    # Hide the unused 8th subplot.
    for ax in axes_flat[len(PREDICTORS):]:
        ax.axis("off")
    fig.suptitle("Baseline predictors vs ViolentCrimesPerPop", y=1.02, fontsize=12)
    plt.tight_layout()
    fig.savefig(SCATTER_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved scatter plots -> {SCATTER_PATH.relative_to(REPO_ROOT)}")

    # -----------------------------------------------------------------------
    # Step 7 — Summary statistics (mean, std, min, max) -> CSV.
    # -----------------------------------------------------------------------
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary = model_df.agg(["mean", "std", "min", "max"]).T
    summary.index.name = "variable"
    summary.to_csv(SUMMARY_PATH)
    print(f"  Saved summary stats -> {SUMMARY_PATH.relative_to(REPO_ROOT)}")

    # -----------------------------------------------------------------------
    # Step 8 — Save the baseline modelling dataset.
    # -----------------------------------------------------------------------
    model_df.to_csv(BASELINE_PATH, index=False)
    print(f"  Saved baseline dataset -> {BASELINE_PATH.relative_to(REPO_ROOT)}")

    # -----------------------------------------------------------------------
    # Summary.
    # -----------------------------------------------------------------------
    print("\n=== Model definition summary (baseline shortlist) ===")
    print(f"Predictors ({len(PREDICTORS)}):")
    for label, var, _ in CONSTRUCTS:
        print(f"  - {var:<18s}  [{label}]")
    print(f"Target: {TARGET}")
    print(f"Baseline subset shape: {model_df.shape}")
    print("NOTE: final model selection (BIC/AIC/CV) is Task 5 in Section C.")


if __name__ == "__main__":
    main()
