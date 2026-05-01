"""
Task 6 — Model Diagnostics and Remedies (Section C).

Reads the outputs of Task 5 (05_model_selection.py) and the shared data files
from Section A.  Does NOT refit model selection — only diagnostics and remedy.

Checks performed:
  1. Multicollinearity  — VIF table, condition number
  2. Heteroscedasticity — Breusch-Pagan test, Residuals-vs-Fitted, Scale-Location
  3. Normality          — Jarque-Bera test, Normal Q-Q plot
  4. Functional form    — Ramsey RESET test (powers 2 and 3)
  5. Autocorrelation    — Durbin-Watson statistic
  6. Influential points — Cook's distance, Leverage; top-5 named via
                          communities_master.csv (communityname + state)

Remedy implemented:
  - Log transformation of the dependent variable: log(Y + c) where c is a
    small positive shift chosen as half the smallest positive Y value.
  - Same final predictors as the baseline model so diagnostic changes are
    attributable to the transformation alone.
  - Side-by-side comparison of baseline vs. log-model diagnostics.

Outputs (in outputs/tables/ and outputs/plots/):
  tables/vif_table.csv
  tables/diagnostic_tests.csv
  tables/influential_points_baseline.csv
  tables/influential_points_logmodel.csv
  tables/log_model_coefficients.csv
  tables/diagnostic_comparison.csv
  plots/residuals_vs_fitted.png
  plots/qq_plot.png
  plots/scale_location.png
  plots/leverage_plot.png
  plots/cooks_distance.png
  plots/logy_residuals_vs_fitted.png
  plots/logy_qq_plot.png
  plots/logy_scale_location.png
  plots/logy_cooks_distance.png
  plots/vif_barplot.png
  plots/diagnostic_panel_baseline.png
  plots/diagnostic_panel_logmodel.png

Run from project root:
    python scripts/06_diagnostics.py

Dependencies: pandas, numpy, scipy, matplotlib  (no statsmodels / sklearn)
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import scipy.stats as st

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE       = Path(__file__).resolve().parent
DATA_DIR   = HERE.parent.parent / "data"
OUTPUTS    = HERE / "outputs"
TABLES_DIR = OUTPUTS / "tables"
PLOTS_DIR  = OUTPUTS / "plots"

CLEAN_PATH   = DATA_DIR / "communities_clean.csv"
MASTER_PATH  = DATA_DIR / "communities_master.csv"
SEL_PATH     = OUTPUTS / "task5_selections.json"

TARGET = "ViolentCrimesPerPop"


# ===========================================================================
# OLS helpers  (identical to 05_model_selection.py — self-contained)
# ===========================================================================

def add_intercept(X_df: pd.DataFrame) -> np.ndarray:
    X = X_df.to_numpy(dtype=float)
    return np.column_stack([np.ones(len(X)), X])


def ols_fit(X: np.ndarray, y: np.ndarray) -> dict:
    n, k = X.shape
    coef, _, _, sv = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ coef
    resid  = y - fitted
    sse    = float(resid @ resid)
    df_resid = n - k
    sigma2   = sse / df_resid

    XtX     = X.T @ X
    XtXinv  = np.linalg.pinv(XtX)
    vcov    = sigma2 * XtXinv

    h       = np.einsum("ij,jk,ik->i", X, XtXinv, X)
    e_adj   = resid / np.clip(1.0 - h, 1e-8, None)
    Xe      = X * e_adj[:, None]
    meat    = Xe.T @ Xe
    vcov_HC3 = XtXinv @ meat @ XtXinv

    se       = np.sqrt(np.diag(vcov))
    se_HC3   = np.sqrt(np.diag(vcov_HC3))
    t_vals   = coef / se
    t_HC3    = coef / se_HC3
    pval     = 2 * st.t.sf(np.abs(t_vals), df=df_resid)
    pval_HC3 = 2 * st.t.sf(np.abs(t_HC3),  df=df_resid)
    crit     = st.t.ppf(0.975, df=df_resid)
    ci_lo    = coef - crit * se;   ci_hi    = coef + crit * se
    ci_lo_HC3 = coef - crit * se_HC3; ci_hi_HC3 = coef + crit * se_HC3

    sst = float(np.sum((y - y.mean()) ** 2))
    r2  = 1.0 - sse / sst
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / df_resid

    sigma2_mle = sse / n
    ll   = -0.5 * n * (np.log(2 * np.pi * sigma2_mle) + 1.0)
    aic  = -2.0 * ll + 2.0 * k
    bic  = -2.0 * ll + np.log(n) * k

    df_model = k - 1
    sst_reg  = sst - sse
    fstat = (sst_reg / df_model) / (sse / df_resid) if df_model > 0 else np.nan
    fpval = float(st.f.sf(fstat, df_model, df_resid)) if not np.isnan(fstat) else np.nan

    cond_num = float(sv[0] / sv[-1]) if len(sv) > 0 and sv[-1] > 0 else np.nan

    return dict(
        coef=coef, fitted=fitted, resid=resid, sse=sse, h=h,
        df_resid=df_resid, sigma2=sigma2, XtXinv=XtXinv,
        vcov=vcov, vcov_HC3=vcov_HC3,
        se=se, t=t_vals, pval=pval, ci_lo=ci_lo, ci_hi=ci_hi,
        se_HC3=se_HC3, t_HC3=t_HC3, pval_HC3=pval_HC3,
        ci_lo_HC3=ci_lo_HC3, ci_hi_HC3=ci_hi_HC3,
        r2=r2, adj_r2=adj_r2, aic=aic, bic=bic,
        fstat=fstat, fpval=fpval, condition_number=cond_num,
        n=n, k=k,
    )


# ===========================================================================
# Diagnostic tests  (pure numpy / scipy)
# ===========================================================================

def vif_table(X_df: pd.DataFrame) -> pd.DataFrame:
    """VIF for each predictor (no intercept column expected in X_df)."""
    cols  = X_df.columns.tolist()
    X_arr = X_df.to_numpy(dtype=float)
    vifs  = []
    for j, col in enumerate(cols):
        y_j = X_arr[:, j]
        X_j = np.delete(X_arr, j, axis=1)
        X_j_c = np.column_stack([np.ones(len(y_j)), X_j])
        coef, _, _, _ = np.linalg.lstsq(X_j_c, y_j, rcond=None)
        fitted = X_j_c @ coef
        ss_res = float(np.sum((y_j - fitted) ** 2))
        ss_tot = float(np.sum((y_j - y_j.mean()) ** 2))
        r2_j   = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vif    = 1.0 / (1.0 - r2_j) if r2_j < 1.0 else np.inf
        vifs.append({"variable": col, "VIF": vif})
    return pd.DataFrame(vifs).sort_values("VIF", ascending=False)


def breusch_pagan(resid: np.ndarray, X_with_intercept: np.ndarray) -> dict:
    """
    Breusch-Pagan test for heteroscedasticity.
    Auxiliary regression: resid² ~ X (with intercept already in X).
    Returns LM statistic, LM p-value, F statistic, F p-value.
    """
    n, k = X_with_intercept.shape
    u2   = resid ** 2
    # auxiliary OLS
    coef_aux, _, _, _ = np.linalg.lstsq(X_with_intercept, u2, rcond=None)
    fitted_aux = X_with_intercept @ coef_aux
    ss_reg = float(np.sum((fitted_aux - u2.mean()) ** 2))
    ss_tot = float(np.sum((u2 - u2.mean()) ** 2))
    r2_aux = ss_reg / ss_tot if ss_tot > 0 else 0.0

    lm     = n * r2_aux
    lm_p   = float(st.chi2.sf(lm, df=k - 1))
    df1    = k - 1
    df2    = n - k
    fstat  = (r2_aux / df1) / ((1 - r2_aux) / df2) if df2 > 0 else np.nan
    fp     = float(st.f.sf(fstat, df1, df2)) if not np.isnan(fstat) else np.nan
    return {"LM": lm, "LM_pval": lm_p, "F": fstat, "F_pval": fp}


def jarque_bera_test(resid: np.ndarray) -> dict:
    """Jarque-Bera normality test."""
    result = st.jarque_bera(resid)
    return {"JB_stat": float(result.statistic), "JB_pval": float(result.pvalue)}


def durbin_watson(resid: np.ndarray) -> float:
    """Durbin-Watson statistic."""
    diff = np.diff(resid)
    return float(np.sum(diff ** 2) / np.sum(resid ** 2))


def ramsey_reset(X_with_intercept: np.ndarray, y: np.ndarray,
                 powers: tuple[int, ...] = (2, 3)) -> dict:
    """
    Ramsey RESET test.  Augments the model with powers of the fitted values
    and tests whether the augmentation coefficients are jointly zero.
    """
    n, k = X_with_intercept.shape
    coef0, _, _, _ = np.linalg.lstsq(X_with_intercept, y, rcond=None)
    yhat = X_with_intercept @ coef0
    sse0 = float(np.sum((y - yhat) ** 2))

    aug_cols = [yhat ** p for p in powers]
    X_aug    = np.column_stack([X_with_intercept] + aug_cols)
    coef1, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
    yhat1 = X_aug @ coef1
    sse1  = float(np.sum((y - yhat1) ** 2))

    q     = len(powers)
    df2   = n - k - q
    fstat = ((sse0 - sse1) / q) / (sse1 / df2) if df2 > 0 else np.nan
    pval  = float(st.f.sf(fstat, q, df2)) if not np.isnan(fstat) else np.nan
    return {"RESET_F": fstat, "RESET_pval": pval, "df1": q, "df2": df2}


def cooks_distance(resid: np.ndarray, h: np.ndarray, k: int, sigma2: float
                   ) -> np.ndarray:
    """Cook's distance Di = (e_i² / (k * sigma²)) * h_i / (1 - h_i)²"""
    denom = (1.0 - h) ** 2
    denom = np.where(denom < 1e-12, 1e-12, denom)
    return (resid ** 2 * h) / (k * sigma2 * denom)


def standardised_residuals(resid: np.ndarray, sigma2: float,
                            h: np.ndarray) -> np.ndarray:
    denom = np.sqrt(sigma2 * (1.0 - h))
    denom = np.where(denom < 1e-12, 1e-12, denom)
    return resid / denom


def influential_table(fit: dict, index: pd.Index, master: pd.DataFrame
                      ) -> pd.DataFrame:
    """Build an influential-points table with community names."""
    cook  = cooks_distance(fit["resid"], fit["h"], fit["k"], fit["sigma2"])
    std_r = standardised_residuals(fit["resid"], fit["sigma2"], fit["h"])
    lev   = fit["h"]

    df = pd.DataFrame({
        "row_index":            np.arange(len(cook)),
        "cooks_distance":       cook,
        "leverage":             lev,
        "standardized_residual":std_r,
        "communityname":        master["communityname"].values,
        "state":                master["state"].values,
    }).sort_values("cooks_distance", ascending=False)
    return df


# ===========================================================================
# Plotting helpers
# ===========================================================================

def plot_residuals_vs_fitted(fitted: np.ndarray, resid: np.ndarray,
                             title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(fitted, resid, s=10, alpha=0.5, color="#4878d0")
    ax.axhline(0, linestyle="--", linewidth=1, color="red")
    ax.set_xlabel("Fitted values"); ax.set_ylabel("Residuals")
    ax.set_title(title)
    plt.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def plot_qq(resid: np.ndarray, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    (osm, osr), (slope, intercept, _) = st.probplot(resid, dist="norm")
    ax.plot(osm, osr, "o", ms=3, alpha=0.5, color="#4878d0")
    ax.plot(osm, slope * np.array(osm) + intercept, "r-", linewidth=1.5)
    ax.set_xlabel("Theoretical Quantiles"); ax.set_ylabel("Sample Quantiles")
    ax.set_title(title)
    plt.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def plot_scale_location(fitted: np.ndarray, resid: np.ndarray,
                        sigma2: float, h: np.ndarray,
                        title: str, path: Path) -> None:
    std_r = standardised_residuals(resid, sigma2, h)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(fitted, np.sqrt(np.abs(std_r)), s=10, alpha=0.5, color="#ee854a")
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("√|Standardized residuals|")
    ax.set_title(title)
    plt.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def plot_cooks(cook: np.ndarray, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    markerline, stemlines, baseline = ax.stem(
        np.arange(len(cook)), cook, markerfmt=" ", basefmt="k-")
    plt.setp(stemlines, linewidth=0.6, color="#4878d0")
    ax.set_xlabel("Observation index"); ax.set_ylabel("Cook's distance")
    ax.set_title(title)
    plt.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def plot_leverage(lev: np.ndarray, std_r: np.ndarray,
                  title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(lev, std_r ** 2, s=10, alpha=0.5, color="#6acc65")
    ax.set_xlabel("Leverage"); ax.set_ylabel("Standardized residuals²")
    ax.set_title(title)
    plt.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def plot_vif_bar(vif_df: pd.DataFrame, path: Path) -> None:
    vif_sorted = vif_df.sort_values("VIF")
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#d65f5f" if v > 5 else "#ee854a" if v > 2.5 else "#6acc65"
              for v in vif_sorted["VIF"]]
    ax.barh(vif_sorted["variable"], vif_sorted["VIF"],
            color=colors, edgecolor="white")
    ax.axvline(5, linestyle="--", linewidth=1, color="red", label="VIF = 5")
    ax.axvline(10, linestyle=":", linewidth=1, color="darkred", label="VIF = 10")
    ax.set_xlabel("VIF"); ax.set_title("Variance Inflation Factors — Final Model")
    ax.legend(fontsize=8)
    plt.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def plot_diagnostic_panel(fit: dict, title_prefix: str, path: Path) -> None:
    """2×2 panel: Residuals vs Fitted, Q-Q, Scale-Location, Cook's Distance."""
    fitted = fit["fitted"];  resid = fit["resid"]
    std_r  = standardised_residuals(resid, fit["sigma2"], fit["h"])
    cook   = cooks_distance(resid, fit["h"], fit["k"], fit["sigma2"])

    fig = plt.figure(figsize=(12, 9))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    # 1. Residuals vs Fitted
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(fitted, resid, s=8, alpha=0.45, color="#4878d0")
    ax1.axhline(0, linestyle="--", linewidth=1, color="red")
    ax1.set_xlabel("Fitted values"); ax1.set_ylabel("Residuals")
    ax1.set_title(f"{title_prefix}\nResiduals vs Fitted")

    # 2. Normal Q-Q
    ax2 = fig.add_subplot(gs[0, 1])
    (osm, osr), (slope, intercept, _) = st.probplot(resid, dist="norm")
    ax2.plot(osm, osr, "o", ms=3, alpha=0.45, color="#4878d0")
    ax2.plot(osm, slope * np.array(osm) + intercept, "r-", linewidth=1.5)
    ax2.set_xlabel("Theoretical Quantiles"); ax2.set_ylabel("Sample Quantiles")
    ax2.set_title("Normal Q-Q Plot")

    # 3. Scale-Location
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(fitted, np.sqrt(np.abs(std_r)), s=8, alpha=0.45, color="#ee854a")
    ax3.set_xlabel("Fitted values")
    ax3.set_ylabel("√|Standardized residuals|")
    ax3.set_title("Scale-Location")

    # 4. Cook's Distance
    ax4 = fig.add_subplot(gs[1, 1])
    markerline, stemlines, _ = ax4.stem(
        np.arange(len(cook)), cook, markerfmt=" ", basefmt="k-")
    plt.setp(stemlines, linewidth=0.5, color="#4878d0")
    ax4.set_xlabel("Observation index"); ax4.set_ylabel("Cook's distance")
    ax4.set_title("Cook's Distance")

    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load data and Task 5 selections
    # -----------------------------------------------------------------------
    print("Loading data …")
    clean  = pd.read_csv(CLEAN_PATH)
    master = pd.read_csv(MASTER_PATH)

    if SEL_PATH.exists():
        with open(SEL_PATH) as f:
            sel = json.load(f)
        final_selected = sel["final_selected"]
        print(f"  Loaded Task 5 selections: {len(final_selected)} final variables")
    else:
        # Fallback: use the known good list from the original script
        final_selected = [
            "racepctblack", "agePct12t29", "pctUrban", "OtherPerCap",
            "MalePctDivorce", "PctKids2Par", "PctWorkMom", "PctIlleg",
            "PctPersDenseHous", "HousVacant", "PctHousOccup",
            "MedOwnCostPctIncNoMtg", "NumStreet",
        ]
        print("  WARNING: task5_selections.json not found — using hardcoded fallback list.")

    y = clean[TARGET].to_numpy(dtype=float)
    X_final_df = clean[final_selected]
    X_final    = add_intercept(X_final_df)
    col_names  = ["const"] + final_selected

    # -----------------------------------------------------------------------
    # Fit baseline final model
    # -----------------------------------------------------------------------
    print("Fitting final OLS model …")
    fit = ols_fit(X_final, y)

    # -----------------------------------------------------------------------
    # 1. VIF
    # -----------------------------------------------------------------------
    print("Computing VIFs …")
    vif_df = vif_table(X_final_df)
    vif_df.to_csv(TABLES_DIR / "vif_table.csv", index=False)
    print(vif_df.round(3).to_string(index=False))

    # -----------------------------------------------------------------------
    # 2. Heteroscedasticity — Breusch-Pagan
    # -----------------------------------------------------------------------
    bp = breusch_pagan(fit["resid"], X_final)
    print(f"\nBreusch-Pagan:  F = {bp['F']:.4f},  p = {bp['F_pval']:.3e}")

    # -----------------------------------------------------------------------
    # 3. Normality — Jarque-Bera
    # -----------------------------------------------------------------------
    jb = jarque_bera_test(fit["resid"])
    print(f"Jarque-Bera:    stat = {jb['JB_stat']:.2f},  p = {jb['JB_pval']:.3e}")

    # -----------------------------------------------------------------------
    # 4. Functional form — Ramsey RESET
    # -----------------------------------------------------------------------
    reset = ramsey_reset(X_final, y)
    print(f"RESET:          F({reset['df1']},{reset['df2']}) = "
          f"{reset['RESET_F']:.4f},  p = {reset['RESET_pval']:.4f}")

    # -----------------------------------------------------------------------
    # 5. Autocorrelation — Durbin-Watson
    # -----------------------------------------------------------------------
    dw = durbin_watson(fit["resid"])
    print(f"Durbin-Watson:  {dw:.4f}")

    # -----------------------------------------------------------------------
    # 6. Influential observations
    # -----------------------------------------------------------------------
    cook  = cooks_distance(fit["resid"], fit["h"], fit["k"], fit["sigma2"])
    std_r = standardised_residuals(fit["resid"], fit["sigma2"], fit["h"])
    inf_df = influential_table(fit, clean.index, master)
    inf_df.to_csv(TABLES_DIR / "influential_points_baseline.csv", index=False)
    print(f"\nTop 5 influential observations (Cook's distance):")
    print(inf_df.head(5)[
        ["communityname", "state", "cooks_distance",
         "leverage", "standardized_residual"]].round(4).to_string(index=False))

    # -----------------------------------------------------------------------
    # Save baseline diagnostic summary
    # -----------------------------------------------------------------------
    diag_baseline = {
        "model": "baseline_OLS",
        "adj_r2": fit["adj_r2"],
        "aic": fit["aic"],
        "bic": fit["bic"],
        "condition_number": fit["condition_number"],
        "max_vif": float(vif_df["VIF"].max()),
        "bp_F": bp["F"], "bp_F_pval": bp["F_pval"],
        "jb_stat": jb["JB_stat"], "jb_pval": jb["JB_pval"],
        "dw": dw,
        "reset_F": reset["RESET_F"], "reset_pval": reset["RESET_pval"],
        "max_cooks": float(cook.max()),
        "n_cooks_above_4n": int((cook > 4 / fit["n"]).sum()),
    }

    # -----------------------------------------------------------------------
    # Baseline coefficient table
    # -----------------------------------------------------------------------
    coef_df = pd.DataFrame({
        "variable":    col_names,
        "coef":        fit["coef"],
        "std_err":     fit["se"],
        "t":           fit["t"],
        "p_value":     fit["pval"],
        "ci_low":      fit["ci_lo"],
        "ci_high":     fit["ci_hi"],
        "std_err_HC3": fit["se_HC3"],
        "t_HC3":       fit["t_HC3"],
        "p_value_HC3": fit["pval_HC3"],
        "ci_low_HC3":  fit["ci_lo_HC3"],
        "ci_high_HC3": fit["ci_hi_HC3"],
    })
    coef_df.to_csv(TABLES_DIR / "baseline_model_coefficients.csv", index=False)

    # -----------------------------------------------------------------------
    # Baseline plots
    # -----------------------------------------------------------------------
    print("\nSaving baseline diagnostic plots …")
    plot_residuals_vs_fitted(
        fit["fitted"], fit["resid"],
        "Residuals vs Fitted — Baseline OLS",
        PLOTS_DIR / "residuals_vs_fitted.png")

    plot_qq(fit["resid"], "Normal Q-Q Plot — Baseline OLS",
            PLOTS_DIR / "qq_plot.png")

    plot_scale_location(
        fit["fitted"], fit["resid"], fit["sigma2"], fit["h"],
        "Scale-Location — Baseline OLS",
        PLOTS_DIR / "scale_location.png")

    plot_cooks(cook, "Cook's Distance — Baseline OLS",
               PLOTS_DIR / "cooks_distance.png")

    plot_leverage(fit["h"], std_r,
                  "Leverage Plot — Baseline OLS",
                  PLOTS_DIR / "leverage_plot.png")

    plot_vif_bar(vif_df, PLOTS_DIR / "vif_barplot.png")

    plot_diagnostic_panel(fit, "Baseline OLS",
                          PLOTS_DIR / "diagnostic_panel_baseline.png")

    # -----------------------------------------------------------------------
    # REMEDY: Log-transform the dependent variable
    # -----------------------------------------------------------------------
    print("\n--- Remedy: log(Y + c) transformation ---")
    positive_y = y[y > 0]
    log_shift  = float(min(1e-3, 0.5 * positive_y.min())) if len(positive_y) > 0 else 1e-6
    y_log      = np.log(y + log_shift)
    print(f"  Log shift c = {log_shift:.6g}")

    fit_log = ols_fit(X_final, y_log)
    print(f"  Log-model Adj-R² = {fit_log['adj_r2']:.4f}  "
          f"(baseline: {fit['adj_r2']:.4f})")

    bp_log    = breusch_pagan(fit_log["resid"], X_final)
    jb_log    = jarque_bera_test(fit_log["resid"])
    reset_log = ramsey_reset(X_final, y_log)
    dw_log    = durbin_watson(fit_log["resid"])
    cook_log  = cooks_distance(fit_log["resid"], fit_log["h"],
                                fit_log["k"], fit_log["sigma2"])
    std_r_log = standardised_residuals(fit_log["resid"], fit_log["sigma2"], fit_log["h"])

    print(f"  BP F-pval:    {bp_log['F_pval']:.3e}  (baseline: {bp['F_pval']:.3e})")
    print(f"  JB stat:      {jb_log['JB_stat']:.2f}  (baseline: {jb['JB_stat']:.2f})")
    print(f"  RESET p:      {reset_log['RESET_pval']:.4f}  "
          f"(baseline: {reset['RESET_pval']:.4f})")
    print(f"  Durbin-Watson:{dw_log:.4f}  (baseline: {dw:.4f})")
    print(f"  Max Cook's D: {cook_log.max():.4f}  (baseline: {cook.max():.4f})")
    print(f"  N above 4/n:  {int((cook_log > 4 / fit_log['n']).sum())}  "
          f"(baseline: {int((cook > 4 / fit['n']).sum())})")

    # Log-model influential points
    inf_log_df = influential_table(fit_log, clean.index, master)
    inf_log_df.to_csv(TABLES_DIR / "influential_points_logmodel.csv", index=False)

    # Log-model coefficient table
    log_coef_df = pd.DataFrame({
        "variable":    col_names,
        "coef":        fit_log["coef"],
        "std_err":     fit_log["se"],
        "t":           fit_log["t"],
        "p_value":     fit_log["pval"],
        "ci_low":      fit_log["ci_lo"],
        "ci_high":     fit_log["ci_hi"],
        "std_err_HC3": fit_log["se_HC3"],
        "t_HC3":       fit_log["t_HC3"],
        "p_value_HC3": fit_log["pval_HC3"],
        "ci_low_HC3":  fit_log["ci_lo_HC3"],
        "ci_high_HC3": fit_log["ci_hi_HC3"],
    })
    log_coef_df.to_csv(TABLES_DIR / "log_model_coefficients.csv", index=False)

    # -----------------------------------------------------------------------
    # Log-model plots
    # -----------------------------------------------------------------------
    print("Saving log-model diagnostic plots …")
    plot_residuals_vs_fitted(
        fit_log["fitted"], fit_log["resid"],
        "Residuals vs Fitted — log(Y+c) Model",
        PLOTS_DIR / "logy_residuals_vs_fitted.png")

    plot_qq(fit_log["resid"],
            "Normal Q-Q Plot — log(Y+c) Model",
            PLOTS_DIR / "logy_qq_plot.png")

    plot_scale_location(
        fit_log["fitted"], fit_log["resid"],
        fit_log["sigma2"], fit_log["h"],
        "Scale-Location — log(Y+c) Model",
        PLOTS_DIR / "logy_scale_location.png")

    plot_cooks(cook_log, "Cook's Distance — log(Y+c) Model",
               PLOTS_DIR / "logy_cooks_distance.png")

    plot_diagnostic_panel(fit_log, "log(Y+c) Remedy",
                          PLOTS_DIR / "diagnostic_panel_logmodel.png")

    # -----------------------------------------------------------------------
    # Diagnostic comparison table
    # -----------------------------------------------------------------------
    diag_log = {
        "model": "log_OLS",
        "adj_r2": fit_log["adj_r2"],
        "aic": fit_log["aic"],
        "bic": fit_log["bic"],
        "condition_number": fit_log["condition_number"],
        "max_vif": float(vif_df["VIF"].max()),       # same X, same VIF
        "bp_F": bp_log["F"], "bp_F_pval": bp_log["F_pval"],
        "jb_stat": jb_log["JB_stat"], "jb_pval": jb_log["JB_pval"],
        "dw": dw_log,
        "reset_F": reset_log["RESET_F"], "reset_pval": reset_log["RESET_pval"],
        "max_cooks": float(cook_log.max()),
        "n_cooks_above_4n": int((cook_log > 4 / fit_log["n"]).sum()),
    }

    diag_comp = pd.DataFrame([diag_baseline, diag_log])
    diag_comp.to_csv(TABLES_DIR / "diagnostic_comparison.csv", index=False)

    # -----------------------------------------------------------------------
    # Save diagnostic tests summary
    # -----------------------------------------------------------------------
    test_rows = [
        {"test": "Breusch-Pagan (F)", "model": "baseline",
         "stat": bp["F"], "pval": bp["F_pval"],
         "conclusion": "Heteroscedasticity present" if bp["F_pval"] < 0.05
                       else "No heteroscedasticity"},
        {"test": "Breusch-Pagan (F)", "model": "log_remedy",
         "stat": bp_log["F"], "pval": bp_log["F_pval"],
         "conclusion": "Heteroscedasticity present" if bp_log["F_pval"] < 0.05
                       else "No heteroscedasticity"},
        {"test": "Jarque-Bera", "model": "baseline",
         "stat": jb["JB_stat"], "pval": jb["JB_pval"],
         "conclusion": "Non-normal residuals" if jb["JB_pval"] < 0.05
                       else "Normality not rejected"},
        {"test": "Jarque-Bera", "model": "log_remedy",
         "stat": jb_log["JB_stat"], "pval": jb_log["JB_pval"],
         "conclusion": "Non-normal residuals" if jb_log["JB_pval"] < 0.05
                       else "Normality not rejected"},
        {"test": "RESET", "model": "baseline",
         "stat": reset["RESET_F"], "pval": reset["RESET_pval"],
         "conclusion": "Misspecification detected" if reset["RESET_pval"] < 0.05
                       else "No misspecification"},
        {"test": "RESET", "model": "log_remedy",
         "stat": reset_log["RESET_F"], "pval": reset_log["RESET_pval"],
         "conclusion": "Misspecification detected" if reset_log["RESET_pval"] < 0.05
                       else "No misspecification"},
        {"test": "Durbin-Watson", "model": "baseline",
         "stat": dw, "pval": None,
         "conclusion": f"DW ≈ {dw:.3f}; no serial dependence concern"},
        {"test": "Durbin-Watson", "model": "log_remedy",
         "stat": dw_log, "pval": None,
         "conclusion": f"DW ≈ {dw_log:.3f}; no serial dependence concern"},
    ]
    test_df = pd.DataFrame(test_rows)
    test_df.to_csv(TABLES_DIR / "diagnostic_tests.csv", index=False)

    # -----------------------------------------------------------------------
    # Final print summary
    # -----------------------------------------------------------------------
    print("\n=== Task 6 complete ===")
    print(f"\nDiagnostic comparison (baseline vs log-remedy):")
    print(diag_comp[[
        "model", "adj_r2", "bp_F_pval", "jb_stat",
        "reset_pval", "dw", "max_cooks", "n_cooks_above_4n"
    ]].round(4).to_string(index=False))
    print(f"\nAll outputs saved to: {OUTPUTS}")


if __name__ == "__main__":
    main()
