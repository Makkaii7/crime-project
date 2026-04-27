from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, ElasticNetCV, LassoLarsIC
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


TARGET = "ViolentCrimesPerPop"
NONPREDICTIVE = ["state", "county", "community", "communityname", "fold"]


def fit_ols(y, X_df):
    return sm.OLS(y, sm.add_constant(X_df)).fit()


def cv_metrics(X_df, y, folds, model_factory, scale=False):
    preds = np.zeros_like(y)

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
        "RMSE": np.sqrt(mean_squared_error(y, preds)),
        "MAE": mean_absolute_error(y, preds),
        "CV_R2": r2_score(y, preds),
    }


def main():
    here = Path(__file__).resolve().parent.parent
    df = pd.read_csv(here / "data/communities_baseline_model.csv")

    TARGET = "ViolentCrimesPerPop"

    missing = df.isna().mean()
    high_missing = missing[missing > 0.8].index

    features = [c for c in df.columns if c not in list(high_missing) + NONPREDICTIVE + [TARGET]]

    X = df[features]
    y = df[TARGET].values
    folds = df["fold"].values

    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=features)

    # LASSO selection
    X_scaled = StandardScaler().fit_transform(X_imp)
    lasso = LassoLarsIC(criterion="bic")
    lasso.fit(X_scaled, y)
    selected20 = list(pd.Series(lasso.coef_, index=features)[lambda x: x != 0].index)

    # BIC reduction
    current = selected20.copy()
    model = fit_ols(y, X_imp[current])

    while True:
        bic_current = model.bic
        candidates = []
        for col in current:
            cols = [c for c in current if c != col]
            m = fit_ols(y, X_imp[cols])
            candidates.append((m.bic, cols, m))

        best_bic, best_cols, best_model = sorted(candidates)[0]

        if best_bic < bic_current:
            current = best_cols
            model = best_model
        else:
            break

    selected13 = current

    full_model = fit_ols(y, X_imp)
    final_model = fit_ols(y, X_imp[selected13])

    metrics = {
        "OLS_full": cv_metrics(X, y, folds, lambda: LinearRegression()),
        "OLS_final": cv_metrics(df[selected13], y, folds, lambda: LinearRegression()),
        "ElasticNet": cv_metrics(
            X, y, folds,
            lambda: ElasticNetCV(l1_ratio=[0.1,0.5,1.0], cv=5),
            scale=True
        ),
    }

    comparison = pd.DataFrame([
        {
            "Model": "OLS_full",
            "Adj_R2": full_model.rsquared_adj,
            "AIC": full_model.aic,
            "BIC": full_model.bic,
            **metrics["OLS_full"],
        },
        {
            "Model": "OLS_final",
            "Adj_R2": final_model.rsquared_adj,
            "AIC": final_model.aic,
            "BIC": final_model.bic,
            **metrics["OLS_final"],
        },
        {
            "Model": "ElasticNet",
            **metrics["ElasticNet"],
        }
    ])

    out = here / "outputs/tables"
    out.mkdir(parents=True, exist_ok=True)

    comparison.to_csv(out / "model_comparison.csv", index=False)

    # Save selected features for next step
    pd.Series(selected13).to_csv(here / "outputs/selected_features.csv", index=False)

    print("Model selection complete.")


if __name__ == "__main__":
    main()
