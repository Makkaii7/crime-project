from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor


def main():
    here = Path(__file__).resolve().parent.parent

    df = pd.read_csv(here / "data/communities_baseline_model.csv")
    selected = pd.read_csv(here / "outputs/selected_features.csv", header=None)[0].tolist()

    TARGET = "ViolentCrimesPerPop"

    X = df[selected]
    y = df[TARGET]

    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    out_tables = here / "outputs/tables"
    out_plots = here / "outputs/plots"

    out_tables.mkdir(parents=True, exist_ok=True)
    out_plots.mkdir(parents=True, exist_ok=True)

    # VIF
    vif = pd.DataFrame({
        "variable": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    })
    vif.to_csv(out_tables / "vif.csv", index=False)

    # Breusch-Pagan
    bp = het_breuschpagan(model.resid, X)

    # Jarque-Bera
    jb = jarque_bera(model.resid)

    # Residual plot
    plt.scatter(model.fittedvalues, model.resid)
    plt.axhline(0)
    plt.savefig(out_plots / "residuals.png")
    plt.close()

    # QQ plot
    sm.qqplot(model.resid, line="45")
    plt.savefig(out_plots / "qqplot.png")
    plt.close()

    print("Diagnostics complete.")
    print("BP p-value:", bp[1])
    print("JB p-value:", jb.pvalue)


if __name__ == "__main__":
    main()
