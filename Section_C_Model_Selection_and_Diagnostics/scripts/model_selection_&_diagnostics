from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
from matplotlib import pyplot as plt
from scipy.stats import jarque_bera
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LassoLarsIC, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor


DATASET_ZIP_URL = "https://archive.ics.uci.edu/static/public/183/communities%2Band%2Bcrime.zip"
TARGET = "ViolentCrimesPerPop"
NONPREDICTIVE = ["state", "county", "community", "communityname", "fold"]

COLUMNS = [
    "state", "county", "community", "communityname", "fold", "population",
    "householdsize", "racepctblack", "racePctWhite", "racePctAsian",
    "racePctHisp", "agePct12t21", "agePct12t29", "agePct16t24",
    "agePct65up", "numbUrban", "pctUrban", "medIncome", "pctWWage",
    "pctWFarmSelf", "pctWInvInc", "pctWSocSec", "pctWPubAsst",
    "pctWRetire", "medFamInc", "perCapInc", "whitePerCap", "blackPerCap",
    "indianPerCap", "AsianPerCap", "OtherPerCap", "HispPerCap",
    "NumUnderPov", "PctPopUnderPov", "PctLess9thGrade", "PctNotHSGrad",
    "PctBSorMore", "PctUnemployed", "PctEmploy", "PctEmplManu",
    "PctEmplProfServ", "PctOccupManu", "PctOccupMgmtProf",
    "MalePctDivorce", "MalePctNevMarr", "FemalePctDiv", "TotalPctDiv",
    "PersPerFam", "PctFam2Par", "PctKids2Par", "PctYoungKids2Par",
    "PctTeen2Par", "PctWorkMomYoungKids", "PctWorkMom", "NumIlleg",
    "PctIlleg", "NumImmig", "PctImmigRecent", "PctImmigRec5",
    "PctImmigRec8", "PctImmigRec10", "PctRecentImmig", "PctRecImmig5",
    "PctRecImmig8", "PctRecImmig10", "PctSpeakEnglOnly",
    "PctNotSpeakEnglWell", "PctLargHouseFam", "PctLargHouseOccup",
    "PersPerOccupHous", "PersPerOwnOccHous", "PersPerRentOccHous",
    "PctPersOwnOccup", "PctPersDenseHous", "PctHousLess3BR", "MedNumBR",
    "HousVacant", "PctHousOccup", "PctHousOwnOcc", "PctVacantBoarded",
    "PctVacMore6Mos", "MedYrHousBuilt", "PctHousNoPhone",
    "PctWOFullPlumb", "OwnOccLowQuart", "OwnOccMedVal", "OwnOccHiQuart",
    "RentLowQ", "RentMedian", "RentHighQ", "MedRent",
    "MedRentPctHousInc", "MedOwnCostPctInc", "MedOwnCostPctIncNoMtg",
    "NumInShelters", "NumStreet", "PctForeignBorn", "PctBornSameState",
    "PctSameHouse85", "PctSameCity85", "PctSameState85", "LemasSwornFT",
    "LemasSwFTPerPop", "LemasSwFTFieldOps", "LemasSwFTFieldPerPop",
    "LemasTotalReq", "LemasTotReqPerPop", "PolicReqPerOffic",
    "PolicPerPop", "RacialMatchCommPol", "PctPolicWhite", "PctPolicBlack",
    "PctPolicHisp", "PctPolicAsian", "PctPolicMinor",
    "OfficAssgnDrugUnits", "NumKindsDrugsSeiz", "PolicAveOTWorked",
    "LandArea", "PopDens", "PctUsePubTrans", "PolicCars", "PolicOperBudg",
    "LemasPctPolicOnPatr", "LemasGangUnitDeploy", "LemasPctOfficDrugUn",
    "PolicBudgPerPop", "ViolentCrimesPerPop",
]


def ensure_data(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / "communities.data"
    if data_path.exists():
        return data_path

    response = requests.get(DATASET_ZIP_URL, timeout=60)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        with zf.open("communities.data") as src, open(data_path, "wb") as dst:
            dst.write(src.read())

    return data_path


def load_data(data_dir: Path) -> pd.DataFrame:
    data_path = ensure_data(data_dir)
    return pd.read_csv(data_path, header=None, names=COLUMNS, na_values="?")


def fit_ols(y: np.ndarray, X_df: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    return sm.OLS(np.asarray(y), sm.add_constant(X_df)).fit()


def cv_metrics(
    X_df: pd.DataFrame,
    y: np.ndarray,
    folds: np.ndarray,
    model_factory,
    scale: bool = False,
) -> dict[str, float]:
    preds = np.zeros_like(y, dtype=float)

    for f in np.unique(folds):
        train = folds != f
        test = folds == f

        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_df.loc[train])
        X_test = imputer.transform(X_df.loc[test])

        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        model = model_factory()
        model.fit(X_train, y[train])
        preds[test] = model.predict(X_test)

    return {
        "RMSE": float(np.sqrt(mean_squared_error(y, preds))),
        "MAE": float(mean_absolute_error(y, preds)),
        "CV_R2": float(r2_score(y, preds)),
    }


def cv_metrics_log_target(
    X_df: pd.DataFrame,
    y: np.ndarray,
    folds: np.ndarray,
    model_factory,
    shift: float,
    scale: bool = False,
) -> dict[str, float]:
    """
    Cross-validation for a log-transformed dependent variable.
    Predictions are mapped back to the original y scale using Duan's smearing.
    """
    preds = np.zeros_like(y, dtype=float)

    for f in np.unique(folds):
        train = folds != f
        test = folds == f

        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_df.loc[train])
        X_test = imputer.transform(X_df.loc[test])

        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        y_train_log = np.log(y[train] + shift)
        model = model_factory()
        model.fit(X_train, y_train_log)

        train_pred_log = model.predict(X_train)
        test_pred_log = model.predict(X_test)
        smear = float(np.mean(np.exp(y_train_log - train_pred_log)))

        preds[test] = np.clip(np.exp(test_pred_log) * smear - shift, 0.0, 1.0)

    return {
        "RMSE": float(np.sqrt(mean_squared_error(y, preds))),
        "MAE": float(mean_absolute_error(y, preds)),
        "CV_R2": float(r2_score(y, preds)),
    }


def get_log_shift(y: np.ndarray) -> float:
    positive = np.asarray(y)[np.asarray(y) > 0]
    if positive.size == 0:
        return 1e-6
    return float(min(1e-3, 0.5 * positive.min()))


def safe_reset(y: np.ndarray, X_df: pd.DataFrame, power: int = 2):
    """
    Avoid the pandas/statsmodels fittedvalues[:, None] issue by fitting an
    equivalent OLS model on pure NumPy arrays for RESET.
    """
    X_np = sm.add_constant(X_df).to_numpy(dtype=float)
    y_np = np.asarray(y, dtype=float)
    aux_model = sm.OLS(y_np, X_np).fit()
    return linear_reset(aux_model, power=power, use_f=True)


def make_coef_table(model, robust_model) -> pd.DataFrame:
    conf_int = model.conf_int()
    robust_conf = np.asarray(robust_model.conf_int())

    return pd.DataFrame(
        {
            "variable": model.params.index,
            "coef": model.params.values,
            "std_err": model.bse.values,
            "t": model.tvalues.values,
            "p_value": model.pvalues.values,
            "coef_HC3": np.asarray(robust_model.params),
            "std_err_HC3": np.asarray(robust_model.bse),
            "t_HC3": np.asarray(robust_model.tvalues),
            "p_value_HC3": np.asarray(robust_model.pvalues),
            "ci_low": conf_int.iloc[:, 0].values,
            "ci_high": conf_int.iloc[:, 1].values,
            "ci_low_HC3": robust_conf[:, 0],
            "ci_high_HC3": robust_conf[:, 1],
        }
    )


def save_plots(model, out_dir: Path, prefix: str = "") -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)

    influence = model.get_influence()
    frame = influence.summary_frame()
    fitted = np.asarray(model.fittedvalues)
    resid = np.asarray(model.resid)
    stud_resid = np.asarray(frame["standard_resid"])
    leverage = np.asarray(frame["hat_diag"])
    cooks = np.asarray(frame["cooks_d"])

    plt.figure(figsize=(7, 5))
    plt.scatter(fitted, resid, s=12, alpha=0.6)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Fitted")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}residuals_vs_fitted.png", dpi=180)
    plt.close()

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    sm.qqplot(resid, line="45", fit=True, ax=ax)
    ax.set_title("Normal Q-Q Plot")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}qq_plot.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(fitted, np.sqrt(np.abs(stud_resid)), s=12, alpha=0.6)
    plt.xlabel("Fitted values")
    plt.ylabel("Sqrt(|Standardized residuals|)")
    plt.title("Scale-Location Plot")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}scale_location.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(leverage, stud_resid ** 2, s=12, alpha=0.6)
    plt.xlabel("Leverage")
    plt.ylabel("Standardized residuals squared")
    plt.title("Leverage Plot")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}leverage_plot.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    markerline, stemlines, baseline = plt.stem(np.arange(len(cooks)), cooks)
    plt.setp(markerline, marker="None")
    plt.xlabel("Observation index")
    plt.ylabel("Cook's distance")
    plt.title("Cook's Distance")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}cooks_distance.png", dpi=180)
    plt.close()

    influential = pd.DataFrame(
        {
            "index": np.arange(len(cooks)),
            "cooks_distance": cooks,
            "leverage": leverage,
            "standardized_residual": stud_resid,
        }
    ).sort_values("cooks_distance", ascending=False)
    influential.to_csv(out_dir / f"{prefix}influential_points.csv", index=False)
    return influential


def main() -> None:
    here = Path(__file__).resolve().parent
    data_dir = here / "data"
    outputs = here / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)

    df = load_data(data_dir)

    missing = df.isna().mean().sort_values(ascending=False)
    high_missing = missing[missing > 0.8].index.tolist()
    features = [c for c in df.columns if c not in high_missing + NONPREDICTIVE + [TARGET]]

    X_all = df[features]
    y = df[TARGET].to_numpy(dtype=float)
    folds = (df["fold"].astype(int) - 1).to_numpy()

    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X_all), columns=features)

    X_scaled = StandardScaler().fit_transform(X_imp)
    lasso = LassoLarsIC(criterion="bic")
    lasso.fit(X_scaled, y)
    lasso_coef = pd.Series(lasso.coef_, index=features)
    selected20 = lasso_coef[lasso_coef.abs() > 1e-8].index.tolist()

    current = selected20.copy()
    current_model = fit_ols(y, X_imp[current])

    while True:
        current_bic = current_model.bic
        candidates = []
        for col in current:
            candidate_cols = [c for c in current if c != col]
            model = fit_ols(y, X_imp[candidate_cols])
            candidates.append((model.bic, col, candidate_cols, model))
        best_bic, removed, best_cols, best_model = sorted(candidates, key=lambda t: t[0])[0]
        if best_bic < current_bic - 1e-8:
            current = best_cols
            current_model = best_model
        else:
            break

    selected13 = current
    final_model = current_model
    final_model_hc3 = final_model.get_robustcov_results(cov_type="HC3")

    full_model = fit_ols(y, X_imp)
    post20_model = fit_ols(y, X_imp[selected20])

    metrics = {
        "OLS_all100": cv_metrics(X_all, y, folds, lambda: LinearRegression(), scale=False),
        "OLS_postLasso20": cv_metrics(df[selected20], y, folds, lambda: LinearRegression(), scale=False),
        "OLS_final13": cv_metrics(df[selected13], y, folds, lambda: LinearRegression(), scale=False),
        "ElasticNet100": cv_metrics(
            X_all,
            y,
            folds,
            lambda: ElasticNetCV(
                l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                cv=5,
                random_state=0,
                max_iter=100000,
            ),
            scale=True,
        ),
    }

    comparison_rows = [
        {
            "Model": "OLS_all100",
            "p": len(features),
            "Adj_R2": full_model.rsquared_adj,
            "AIC": full_model.aic,
            "BIC": full_model.bic,
            **metrics["OLS_all100"],
        },
        {
            "Model": "OLS_postLasso20",
            "p": len(selected20),
            "Adj_R2": post20_model.rsquared_adj,
            "AIC": post20_model.aic,
            "BIC": post20_model.bic,
            **metrics["OLS_postLasso20"],
        },
        {
            "Model": "OLS_final13",
            "p": len(selected13),
            "Adj_R2": final_model.rsquared_adj,
            "AIC": final_model.aic,
            "BIC": final_model.bic,
            **metrics["OLS_final13"],
        },
        {
            "Model": "ElasticNet100",
            "p": int((lasso_coef.abs() > 1e-8).sum()),
            "Adj_R2": np.nan,
            "AIC": np.nan,
            "BIC": np.nan,
            **metrics["ElasticNet100"],
        },
    ]

    coef_table = make_coef_table(final_model, final_model_hc3)
    coef_table.to_csv(outputs / "final_model_coefficients.csv", index=False)

    X_final = sm.add_constant(X_imp[selected13])
    bp = het_breuschpagan(final_model.resid, X_final)
    reset = safe_reset(y, X_imp[selected13], power=2)
    jb = jarque_bera(final_model.resid)
    dw = sm.stats.durbin_watson(final_model.resid)

    vif_table = pd.DataFrame(
        {
            "variable": selected13,
            "VIF": [variance_inflation_factor(X_final.values, i) for i in range(1, X_final.shape[1])],
        }
    ).sort_values("VIF", ascending=False)
    vif_table.to_csv(outputs / "final_model_vif.csv", index=False)

    family_test = final_model.f_test("PctKids2Par = 0, PctIlleg = 0, MalePctDivorce = 0, PctWorkMom = 0")
    housing_test = final_model.f_test(
        "PctPersDenseHous = 0, HousVacant = 0, PctHousOccup = 0, "
        "MedOwnCostPctIncNoMtg = 0, OtherPerCap = 0"
    )

    influential = save_plots(final_model, outputs)

    # Remedy: log-transform the dependent variable and refit the final specification.
    log_shift = get_log_shift(y)
    y_log = np.log(y + log_shift)
    log_model = fit_ols(y_log, X_imp[selected13])
    log_model_hc3 = log_model.get_robustcov_results(cov_type="HC3")

    metrics["OLS_final13_logY"] = cv_metrics_log_target(
        df[selected13],
        y,
        folds,
        lambda: LinearRegression(),
        shift=log_shift,
        scale=False,
    )

    comparison_rows.append(
        {
            "Model": "OLS_final13_logY",
            "p": len(selected13),
            "Adj_R2": log_model.rsquared_adj,
            "AIC": log_model.aic,
            "BIC": log_model.bic,
            **metrics["OLS_final13_logY"],
        }
    )

    log_coef_table = make_coef_table(log_model, log_model_hc3)
    log_coef_table.to_csv(outputs / "final_logy_model_coefficients.csv", index=False)

    log_bp = het_breuschpagan(log_model.resid, X_final)
    log_reset = safe_reset(y_log, X_imp[selected13], power=2)
    log_jb = jarque_bera(log_model.resid)
    log_dw = sm.stats.durbin_watson(log_model.resid)

    log_influential = save_plots(log_model, outputs, prefix="logy_")

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(outputs / "model_comparison.csv", index=False)

    summary = {
        "n_obs": int(df.shape[0]),
        "n_features_total": int(len(df.columns) - 1),
        "n_features_after_dropping_nonpredictive_and_high_missing": int(len(features)),
        "n_high_missing_columns_dropped": int(len(high_missing)),
        "high_missing_columns": high_missing,
        "remaining_missing_cells_after_drop": int(X_all.isna().sum().sum()),
        "selected20": selected20,
        "selected13": selected13,
        "full_adj_r2": float(full_model.rsquared_adj),
        "post20_adj_r2": float(post20_model.rsquared_adj),
        "final13_adj_r2": float(final_model.rsquared_adj),
        "final13_aic": float(final_model.aic),
        "final13_bic": float(final_model.bic),
        "final13_rmse": metrics["OLS_final13"]["RMSE"],
        "final13_mae": metrics["OLS_final13"]["MAE"],
        "final13_cv_r2": metrics["OLS_final13"]["CV_R2"],
        "bp_lm_stat": float(bp[0]),
        "bp_lm_pvalue": float(bp[1]),
        "bp_f_stat": float(bp[2]),
        "bp_f_pvalue": float(bp[3]),
        "jb_stat": float(jb.statistic),
        "jb_pvalue": float(jb.pvalue),
        "durbin_watson": float(dw),
        "reset_fvalue": float(np.asarray(reset.fvalue).item()),
        "reset_pvalue": float(np.asarray(reset.pvalue).item()),
        "family_fvalue": float(np.asarray(family_test.fvalue).item()),
        "family_pvalue": float(np.asarray(family_test.pvalue).item()),
        "housing_fvalue": float(np.asarray(housing_test.fvalue).item()),
        "housing_pvalue": float(np.asarray(housing_test.pvalue).item()),
        "final_condition_number": float(final_model.condition_number),
        "full_condition_number": float(full_model.condition_number),
        "max_cooks_distance": float(influential["cooks_distance"].max()),
        "log_shift": float(log_shift),
        "logy_adj_r2": float(log_model.rsquared_adj),
        "logy_aic": float(log_model.aic),
        "logy_bic": float(log_model.bic),
        "logy_rmse_original_scale": metrics["OLS_final13_logY"]["RMSE"],
        "logy_mae_original_scale": metrics["OLS_final13_logY"]["MAE"],
        "logy_cv_r2_original_scale": metrics["OLS_final13_logY"]["CV_R2"],
        "logy_bp_lm_stat": float(log_bp[0]),
        "logy_bp_lm_pvalue": float(log_bp[1]),
        "logy_bp_f_stat": float(log_bp[2]),
        "logy_bp_f_pvalue": float(log_bp[3]),
        "logy_jb_stat": float(log_jb.statistic),
        "logy_jb_pvalue": float(log_jb.pvalue),
        "logy_durbin_watson": float(log_dw),
        "logy_reset_fvalue": float(np.asarray(log_reset.fvalue).item()),
        "logy_reset_pvalue": float(np.asarray(log_reset.pvalue).item()),
        "logy_condition_number": float(log_model.condition_number),
        "logy_max_cooks_distance": float(log_influential["cooks_distance"].max()),
    }
    with open(outputs / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Analysis complete.")
    print("\nSelected variables (final 13):")
    print(", ".join(selected13))
    print("\nModel comparison:")
    print(comparison.round(4).to_string(index=False))
    print("\nOriginal-model diagnostics:")
    print(f"Breusch-Pagan p-value: {bp[3]:.3e}")
    print(f"Jarque-Bera p-value:   {jb.pvalue:.3e}")
    print(f"Durbin-Watson:         {dw:.4f}")
    print(f"RESET p-value:         {float(np.asarray(reset.pvalue).item()):.4f}")
    print("\nLog-remedy diagnostics:")
    print(f"Log shift used:        {log_shift:.6g}")
    print(f"Breusch-Pagan p-value: {log_bp[3]:.3e}")
    print(f"Jarque-Bera p-value:   {log_jb.pvalue:.3e}")
    print(f"Durbin-Watson:         {log_dw:.4f}")
    print(f"RESET p-value:         {float(np.asarray(log_reset.pvalue).item()):.4f}")
    print(f"Outputs saved to:      {outputs}")


if __name__ == "__main__":
    main()
