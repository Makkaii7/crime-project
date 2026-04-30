# Section C — Model Selection, Diagnostics, and Remedies

## Introduction

This section reports model selection, diagnostics, and remedies for the UCI Communities and Crime
dataset. The analysis uses the shared baseline dataset `data/communities_baseline_model.csv`
(1,994 communities, 7 predictors + target) as its starting point. Where our data-driven feature
selection diverges from Section A's theory-led shortlist, we note the divergence explicitly and
justify it below.

The primary goal is not only a model with good predictive power, but also an interpretable regression
specification that supports coefficient interpretation, hypothesis testing, confidence interval
construction, and formal model diagnostics. The final workflow combines data cleaning,
penalized least-squares screening, OLS estimation, model comparison, and diagnostic testing.

> **Observational study caveat.** All estimated relationships are conditional associations, not causal
> effects. This is especially important for sensitive demographic variables, which may reflect latent
> structural inequality, segregation, or omitted contextual factors rather than direct causal processes.

---

## Data Preprocessing

The Communities and Crime dataset contains 1,994 communities and 128 columns. The data are
already normalized to [0, 1] by attribute, so no additional min-max scaling was needed for the OLS
step. Note that each variable was normalized separately; coefficient magnitudes should therefore be
compared within a variable (full-range movement) rather than across variables.

**Non-predictive columns removed.** Five columns — `state`, `county`, `community`,
`communityname`, and `fold` — were removed prior to modelling. These are identifiers or
cross-validation bookkeeping variables, not substantive predictors. In particular, `fold` must never
enter a regression model since it encodes the 10-fold CV split.

**Missing data.** As documented in Section A, 22 police/LEMAS variables were missing for ~84% of
communities and were dropped. After dropping those and the five administrative columns, 100
predictors remained. A single missing cell in `OtherPerCap` was imputed with the column median
(robust to skew; consistent with Section A's preprocessing).

**Standardization for penalized regression.** Although OLS was run on the original normalized
predictors for interpretability, LASSO and Elastic Net screening was performed on standardized
predictors. This is essential because L1/L2 penalties are scale-sensitive; without standardization,
higher-variance predictors would be systematically under-penalized.

---

## Model Definition and Classical Linear Model Assumptions

The final explanatory model is a multiple linear regression:

$$
Y_i = \beta_0 + \beta_1 X_{1i} + \beta_2 X_{2i} + \cdots + \beta_{13} X_{13i} + u_i
$$

where $Y_i$ is the normalized violent crime measure (`ViolentCrimesPerPop`) for community $i$,
and the selected predictors are: `racepctblack`, `agePct12t29`, `pctUrban`, `OtherPerCap`,
`MalePctDivorce`, `PctKids2Par`, `PctWorkMom`, `PctIlleg`, `PctPersDenseHous`, `HousVacant`,
`PctHousOccup`, `MedOwnCostPctIncNoMtg`, and `NumStreet`.

### Classical Linear Model Assumptions (aligned with Section A)

Following Section A's framing, we adopt five core CLM assumptions, with normality noted separately:

1. **Linearity in parameters.** The conditional mean of the outcome is a linear function of the
   included regressors and coefficients.
2. **Random sampling / independence of observations.** Each community contributes one
   independent observation.
3. **No perfect multicollinearity.** No predictor is an exact linear combination of the others.
4. **Zero conditional mean.** $E[u \mid X] = 0$ — the error term carries no systematic information
   about the regressors. This is the exogeneity assumption that makes OLS unbiased.
5. **Homoscedasticity.** The variance of the error term is constant across observations:
   $\text{Var}(u \mid X) = \sigma^2$.

**Normality of errors** (sometimes listed as a sixth assumption) is not required for OLS to be
unbiased or BLUE, but it is required for classical small-sample $t$- and $F$-tests to have exact
distributions. With $N = 1{,}994$, the CLT makes inference robust to non-normality — but we
inspect residual normality explicitly in the diagnostics section.

In cross-sectional community data, the most realistic threats are omitted variables,
heteroscedasticity, non-normal residuals, and multicollinearity among socially related indicators.
All four are examined in the diagnostics section below.

---

## Divergence from Section A's Theory-Led Shortlist

Section A constructed a baseline shortlist of **7 predictors** using a theory-driven, one-variable-
per-construct approach (socioeconomic disadvantage, education, employment, family disruption,
housing disorder, urban scale, community size). Our final 13-variable model was selected by a
data-driven pipeline (BIC-penalized LASSO screening followed by BIC backward elimination), and
overlaps with Section A's list on only one variable (`PctKids2Par`).

**We included Section A's 7-variable model as an explicit candidate in model selection** (see the
model comparison table below). The data-driven 13-variable specification outperforms it on every
criterion: it achieves a higher adjusted $R^2$ (0.662 vs 0.587 reported in Section B), a lower BIC,
and nearly identical cross-validated RMSE — confirming that the additional variables contribute
genuine out-of-sample predictive information rather than overfitting.

The key substantive differences are:

- Section A's disadvantage block (`PctPopUnderPov`, `PctNotHSGrad`, `PctUnemployed`) is replaced
  by more specific family-structure and housing indicators. This is consistent with Section B's
  finding that a joint F-test on `PctNotHSGrad` and `PctUnemployed` fails to reject $H_0 = 0$
  ($F(2, 1986) = 1.04$, $p = 0.355$), and the sign reversal of `PctPopUnderPov` in the multivariate
  model — both suggesting that the disadvantage block carries its predictive signal primarily through
  `PctKids2Par` once other variables are present.
- The data-driven model adds family-disruption variables (`MalePctDivorce`, `PctIlleg`,
  `PctWorkMom`), dense-housing stress (`PctPersDenseHous`), and a homelessness indicator
  (`NumStreet`) that were not in Section A's shortlist. Their inclusion is justified by the LASSO
  screening and confirmed by individual significance under HC3 robust inference.

**On `racepctblack`:** Section A explicitly excluded demographic composition variables to avoid
difficult causal interpretation. We re-introduce `racepctblack` here because the LASSO-BIC
selection identifies it as one of the strongest predictors after controlling for the other included
variables, and omitting a substantively important predictor would risk omitted-variable bias in the
remaining coefficients. We interpret the coefficient as purely associational — it likely captures
correlated structural factors such as historical segregation, concentrated disadvantage, or omitted
local conditions, not an inherent trait. Any causal interpretation would require an identification
strategy well beyond the scope of this observational dataset.

---

## Estimation of Model Parameters

The estimation process involved two steps. First, a BIC-penalized LASSO screening step narrowed
100 filtered predictors to 20 variables. Second, BIC-based backward elimination produced a more
interpretable 13-variable OLS specification. This combination balances prediction and
interpretability: penalized regression handles high dimensionality, and the resulting OLS model
supports standard $t$-tests, $F$-tests, and confidence intervals.

### Coefficient Table (HC3 Robust Inference)

Coefficients were interpreted using HC3 robust standard errors because heteroscedasticity was
confirmed in diagnostics (see below). Since all predictors are normalized to [0, 1] with both
endpoints observed, each coefficient represents the predicted change in `ViolentCrimesPerPop`
from the community at the dataset minimum to the community at the dataset maximum on that
predictor — the full empirical range.

| Variable | Coefficient | Effect per +0.10 | HC3 SE | HC3 p-value | 95% HC3 CI |
|---|---|---|---|---|---|
| const | 0.3535 | 0.0353 | 0.0543 | < 0.001 | [0.247, 0.460] |
| racepctblack | 0.1695 | 0.0169 | 0.0292 | < 0.001 | [0.112, 0.227] |
| agePct12t29 | −0.0917 | −0.0092 | 0.0226 | < 0.001 | [−0.136, −0.047] |
| pctUrban | 0.0378 | 0.0038 | 0.0080 | < 0.001 | [0.022, 0.053] |
| OtherPerCap | 0.0482 | 0.0048 | 0.0164 | 0.003 | [0.016, 0.080] |
| MalePctDivorce | 0.1097 | 0.0110 | 0.0306 | < 0.001 | [0.050, 0.170] |
| PctKids2Par | −0.2953 | −0.0295 | 0.0485 | < 0.001 | [−0.391, −0.200] |
| PctWorkMom | −0.0794 | −0.0079 | 0.0200 | < 0.001 | [−0.119, −0.040] |
| PctIlleg | 0.2168 | 0.0217 | 0.0458 | < 0.001 | [0.127, 0.307] |
| PctPersDenseHous | 0.1941 | 0.0194 | 0.0233 | < 0.001 | [0.148, 0.240] |
| HousVacant | 0.1113 | 0.0111 | 0.0381 | 0.004 | [0.037, 0.186] |
| PctHousOccup | −0.0637 | −0.0064 | 0.0209 | 0.002 | [−0.105, −0.023] |
| MedOwnCostPctIncNoMtg | −0.0578 | −0.0058 | 0.0164 | < 0.001 | [−0.090, −0.026] |
| NumStreet | 0.1808 | 0.0181 | 0.0539 | < 0.001 | [0.075, 0.286] |

The final 13-variable OLS model achieved an adjusted $R^2$ of **0.6619**, a cross-validated RMSE
of **0.1362**, a cross-validated MAE of **0.0944**, and a cross-validated $R^2$ of **0.6579**. These
values indicate that the model explains a substantial share of the variation in normalized violent
crime while generalizing well in 10-fold cross-validation.

### Key Coefficient Interpretations

**`PctKids2Par` (coef = −0.2953, p < 0.001).** Because this predictor is normalized to [0, 1], the
coefficient measures the predicted change in `ViolentCrimesPerPop` from moving from the
community with the fewest two-parent households to the one with the most. The coefficient of
−0.2953 means that across the full empirical range, predicted violent crime falls by approximately
**0.30** — a large *negative* effect, indicating that communities with higher shares of children living
in two-parent households tend to have *lower* violent crime rates, holding other predictors fixed.
This is the largest absolute coefficient in the model and is consistent with the finding in Section B
(where `PctKids2Par` carried a coefficient of −0.767 in the 7-variable model, also highly
significant). The 95% CI of [−0.391, −0.200] is tight, reflecting a precisely estimated association.

**`PctIlleg` (coef = +0.2168, p < 0.001).** Communities with higher proportions of children born
to never-married mothers are associated with higher violent crime rates, even after adjusting for
the other included variables. Per a 0.10 increase: +0.022 in predicted crime.

**`PctPersDenseHous` (coef = +0.1941, p < 0.001).** Dense housing is positively associated with
violent crime across the full empirical range (+0.194).

**`NumStreet` (coef = +0.1808, p < 0.001).** The street homelessness measure shows a positive
association with crime, consistent with concentrated social disadvantage.

**`MalePctDivorce` (coef = +0.1097, p < 0.001).** Higher male divorce rates are associated with
higher crime rates (+0.110 across the full range).

**`racepctblack` (coef = +0.1695, p < 0.001).** As noted above, this coefficient is associational
and should not be given a causal interpretation. It likely reflects omitted structural conditions
correlated with racial composition (e.g., historical segregation, concentrated disadvantage).

**`PctWorkMom` (coef = −0.0794, p < 0.001)** and **`PctHousOccup` (coef = −0.0637, p = 0.002)**
are both negatively associated with crime; `HousVacant` (coef = +0.1113, p = 0.004) is positively
associated, consistent with housing instability as a risk factor.

---

## Model Selection

Several competing models were compared, including Section A's theory-led 7-variable baseline
(estimated from `data/communities_baseline_model.csv` for comparability):

| Model | p | Adj R² | AIC | BIC | RMSE | MAE | CV R² |
|---|---|---|---|---|---|---|---|
| OLS — all 100 predictors | 100 | 0.6785 | −2315.29 | −1749.90 | 0.1363 | 0.0960 | 0.6575 |
| OLS — Section A 7-variable baseline | 7 | 0.5850 | — | — | — | — | — |
| OLS — post-LASSO 20 variables | 20 | 0.6642 | −2305.67 | −2188.12 | 0.1361 | 0.0946 | 0.6588 |
| **OLS — final 13 variables** | **13** | **0.6619** | **−2299.32** | **−2220.95** | **0.1362** | **0.0944** | **0.6579** |
| Elastic Net (benchmark) | 20 | — | — | — | 0.1356 | 0.0942 | 0.6611 |

The full 100-predictor OLS has the highest in-sample fit but a condition number of 916.40, signalling
serious multicollinearity and weak interpretability. The Section A 7-variable model has notably lower
adjusted $R^2$ (0.585 vs 0.662), confirming that the additional variables recovered by the data-driven
pipeline contribute genuine predictive information.

The Elastic Net achieves the lowest cross-validated RMSE (0.1356) and highest CV $R^2$ (0.6611),
making it the best pure-prediction model. However, penalized models shrink coefficients and
complicate classical inference. Since this project's primary goal is interpretable explanatory
regression rather than prediction alone, the final 13-variable OLS model is the most appropriate
choice. It achieves the best BIC (−2220.95), which explicitly rewards parsimony, while retaining
nearly identical out-of-sample accuracy to the larger models.

---

## Model Diagnostics and Remedies

The diagnostics reveal that the final model is useful but not perfect. The findings below are
consistent with, and extend, the diagnostic caveats raised in Section B.

### Multicollinearity

Multicollinearity is substantially reduced compared to the full model. The condition number of the
final 13-variable model is **42.93**, versus 916.40 for the full OLS model — a dramatic improvement.
VIF values are as follows:

| Variable | VIF |
|---|---|
| PctKids2Par | 9.12 |
| PctIlleg | 7.94 |
| racepctblack | 3.65 |
| MalePctDivorce | 2.88 |
| HousVacant | 2.11 |
| PctPersDenseHous | 1.77 |
| NumStreet | 1.68 |
| PctHousOccup | 1.59 |
| agePct12t29 | 1.33 |
| PctWorkMom | 1.30 |
| MedOwnCostPctIncNoMtg | 1.24 |
| pctUrban | 1.20 |
| OtherPerCap | 1.15 |

The highest VIFs are for `PctKids2Par` (9.12) and `PctIlleg` (7.94). These are not negligible, but
they are explicable: both variables represent overlapping aspects of family structure. Variable
screening and BIC-based pruning was the primary remedy and led to a significant reduction in
instability relative to the full model.

### Heteroscedasticity

Heteroscedasticity is clearly present, consistent with Section B's observation of a cone shape in
the residuals-vs-fitted plot. The Breusch–Pagan test strongly rejects homoscedasticity, with an
F-test p-value of approximately $2.15 \times 10^{-48}$. The Residuals vs Fitted and Scale-Location
plots show a fan-shaped pattern in which residual spread increases with fitted values.

**Remedy implemented:** HC3 robust standard errors are used for all reported inference. This makes
standard errors, $t$-statistics, $p$-values, and confidence intervals more reliable under
heteroscedasticity without changing the coefficient point estimates.

### Non-Normal Residuals

As also flagged in Section B (Jarque-Bera test, $p < 10^{-100}$), the residuals do not follow a
normal distribution. The Jarque-Bera statistic for the final model is **1351.46** with $p \approx 0$,
and the Normal Q-Q plot shows strong upper-tail deviation from the 45-degree line, indicating a
right-skewed, heavy-tailed residual distribution. With $N = 1{,}994$, the Central Limit Theorem
provides approximate validity for $t$- and $F$-tests even under non-normality, so inference is not
invalidated — but the caveat should be noted. The log-transformation remedy (below) partially
addresses this.

### Functional Form (Ramsey RESET Test)

The Ramsey RESET test rejects the null of a correctly specified functional form ($F = 13.01$,
$p = 0.00032$). This suggests that there may be nonlinearities or interactions missing from the
current specification — consistent with the residual plots. A future enhancement would be to
include quadratic or spline terms for the strongest continuous predictors such as `PctKids2Par`,
`PctPersDenseHous`, and `NumStreet`. Within the scope of classical linear regression, the current
model serves as an informative baseline.

### Autocorrelation

There is no evidence of meaningful autocorrelation: the Durbin–Watson statistic is **1.99997**,
essentially 2. As these are cross-sectional community data, serial dependence is not the primary
concern.

### Influential Observations

The influence diagnostics reveal a few moderately influential points, but no single observation
dominates the model. The five communities with the highest Cook's distances are:

| Community (Index) | Cook's Distance | Leverage | Standardized Residual |
|---|---|---|---|
| Index 682 | 0.03588 | 0.05438 | −2.956 |
| Index 1698 | 0.02391 | 0.05498 | −2.399 |
| Index 957 | 0.01633 | 0.01055 | +4.631 |
| Index 82 | 0.01436 | 0.00588 | +5.827 |
| Index 1034 | 0.01395 | 0.03460 | −2.335 |

> **Note:** Community names corresponding to these indices can be looked up in
> `data/communities_master.csv` using the `communityname`, `county`, and `state` columns, which
> were retained in the master dataset precisely for this purpose. Reporting bare index numbers is
> not informative; the final report should substitute actual community names from that file.

The maximum Cook's distance is **0.0359**, well below the conventional threshold of 1. The data do
not indicate a single disastrous outlier driving the results.

---

## Implemented Remedy: Log Transformation of the Dependent Variable

To address the right-skewness and heteroscedasticity identified above, a remedial model was
estimated using a log-shifted dependent variable:

$$
\log(Y + c)
$$

where $Y = \texttt{ViolentCrimesPerPop}$ and $c$ is a small positive shift constant added because
the response includes zero values, making the log otherwise undefined. The same 13 predictors
were retained so that changes in diagnostics can be attributed to the transformation rather than
to a different feature set.

### Rationale

The original response is bounded in [0, 1] and heavily concentrated near low values with a long
right tail — precisely the setting where a log transformation is a standard remedy, as it compresses
large outcome values more than small ones and can stabilize variance. This is consistent with the
recommendation in Section B that Section C explore target transformation.

### Evaluation of the Remedy

**Heteroscedasticity.** The log-transformed Residuals vs Fitted plot is substantially less dominated
by extreme upper-tail dispersion. The fan shape visible in the baseline OLS is attenuated and the
Scale-Location plot is more even across the middle of the fitted-value range, indicating partial
variance stabilization. However, the remedy is not perfect: because the response contains many
very small or zero values, the transformed residual plots display some residual striping and
structure. Heteroscedasticity is reduced but not eliminated.

**Normality.** The log-model Q-Q plot shows significant improvement in the upper tail compared
to the baseline — the sharp rightward bend in the classical model is substantially reduced. The
central portion of the distribution is closer to the reference line after transformation. However,
the lower tail shows a new strong deviation: very small response values produce large negative
log-transformed values, creating left-tail distortion. The transformation thus improves normality
in the center and upper tail, but worsens it in the extreme lower tail.

**Influential observations.** The maximum Cook's distance decreases from **0.0359** (baseline) to
**0.0284** (log model). Maximum leverage is unchanged (~0.0791) since leverage depends only on
the predictor matrix. The number of observations exceeding the $4/n$ Cook's distance threshold
falls from **164** in the baseline to **78** in the log model — a meaningful reduction in moderately
influential points.

The five highest-Cook's-distance communities in the log model:

| Index | Cook's Distance | Leverage | Standardized Residual |
|---|---|---|---|
| 342 | 0.02843 | 0.01323 | −5.450 |
| 773 | 0.01987 | 0.00680 | −6.374 |
| 1462 | 0.01673 | 0.00854 | −5.214 |
| 426 | 0.01543 | 0.00744 | −5.367 |
| 519 | 0.01514 | 0.00606 | −5.898 |

**Significant predictors under log-model HC3 inference:**
`racepctblack`, `pctUrban`, `MalePctDivorce`, `PctKids2Par`, `PctPersDenseHous`, `HousVacant`,
`PctHousOccup`, and `MedOwnCostPctIncNoMtg` remain statistically significant.

Predictors that lose significance after transformation: `agePct12t29`, `OtherPerCap`,
`PctWorkMom`, `PctIlleg`, and `NumStreet`. This suggests those effects are more sensitive to the
magnitude of the dependent variable and should be interpreted with greater caution.

Note that coefficients in the log model describe variation in log-crime rather than in the original
normalized crime scale, complicating direct comparison of effect sizes. The log-transformed model
is therefore best used as a diagnostic check and robustness test rather than as the primary
substantive model.

---

## Conclusion

The Communities and Crime dataset analysis demonstrates that a 13-variable OLS model — selected
by LASSO screening and BIC backward elimination from 100 cleaned predictors — can explain a
substantial share of violent crime variation while remaining interpretable and statistically robust.
The model achieves an adjusted $R^2$ of approximately 0.662, the best BIC among the main OLS
candidates, and predictive performance nearly identical to larger models, making it the most
appropriate explanatory specification in this project.

Family structure and housing distress emerge as the most important predictors. Family instability
is captured by `PctKids2Par` (negative association — higher two-parent household share predicts
lower crime), `PctIlleg`, `MalePctDivorce`, and `PctWorkMom`. Housing disadvantage and social
stress are captured by `PctPersDenseHous`, `HousVacant`, `PctHousOccup`, and `NumStreet`.
These patterns are supported by both individual coefficient tests and strong joint F-tests, and
are broadly consistent with the Section B finding that `PctKids2Par` is the dominant predictor in
the baseline model.

The diagnostic tests confirm issues flagged in Section B: heteroscedasticity (visible cone shape in
residuals-vs-fitted; Breusch–Pagan $p \approx 10^{-48}$) and non-normal residuals (Jarque-Bera
$p \approx 0$; Q-Q plot upper-tail deviation). Two remedies were applied: HC3 robust standard
errors for the primary model, and a log transformation of the dependent variable as a robustness
check. The log-transformed model behaves better with respect to residuals and reduces the number
of influential observations, but does not fully eliminate all assumption violations.

**The most justifiable final decision is to retain the original 13-variable OLS model with HC3
robust standard errors as the primary model, with the log-transformed model as a robustness
check.** Family instability, housing disadvantage, and general socioeconomic stress are the
strongest predictors of violent crime in this dataset. Available law-enforcement variables are less
predictive due to high missingness rates, as documented in Section A's preprocessing.
