# Analyzing a Real-World Dataset Using Linear Regression
## UCI Communities and Crime Dataset — Full Project Report

> **Observational-study caveat.** All estimated relationships are conditional associations, not causal effects. This applies especially to sensitive demographic variables, which may reflect latent structural inequality, historical segregation, or omitted contextual factors rather than direct causal processes.

---

## Introduction

This project applies linear regression to investigate the relationship between socio-economic, demographic, and housing variables and violent crime rates across U.S. communities, using the UCI Communities and Crime dataset. The dataset integrates socio-economic data from the 1990 US Census, law-enforcement data from the 1990 US LEMAS survey, and crime data from the 1995 FBI UCR. It comprises **1,994 communities** with **128 variables**, including demographic details, law-enforcement statistics, and various crime rates. The response variable is `ViolentCrimesPerPop`, a normalized per-capita violent crime rate bounded in [0, 1].

The primary goal is not only to fit a model with good predictive power, but to construct an interpretable regression specification that supports coefficient interpretation, hypothesis testing, confidence interval construction, and formal model diagnostics. The workflow combines data cleaning, theory-led model definition, OLS estimation, formal inference, data-driven model selection, and diagnostic testing with implemented remedies.

The analysis is organized into three sections:

- **Section A (Tasks 1–2):** Data preprocessing and baseline model definition
- **Section B (Tasks 3–4):** OLS estimation, coefficient interpretation, hypothesis testing, and confidence intervals
- **Section C (Tasks 5–6):** Model selection and model diagnostics with remedies

---

## Section A — Preprocessing and Model Definition

### Task 1 — Data Preprocessing

#### The Missing-Data Problem

The raw UCI dataset contains 1,994 communities and 128 columns. Missing values are concentrated in two places:

1. **22 police-related variables** (from the LEMAS survey) are missing for 1,675 communities — approximately 84% of the sample. These are not scattered missing values; they are systematically absent for communities that were not surveyed.
2. **`OtherPerCap`** has exactly one missing value.

A third, more subtle issue was also identified: `LemasPctOfficDrugUn`.

#### Handling the 22 LEMAS Police Columns

Three options were considered:

| Option | Description | Decision |
|---|---|---|
| Drop columns | Remove the 22 variables; keep all 1,994 communities. | **Chosen** |
| Drop rows | Restrict sample to the 319 communities with LEMAS data. | Rejected — reduces N by 84% and biases the sample toward larger, more urban, higher-crime communities. The rule of thumb of 10–20 observations per predictor is still satisfied without these columns. |
| Impute | Fill the 84% missing entries with a statistical method. | Rejected — requires fabricating values for the overwhelming majority of observations; any imputation would be driven almost entirely by the imputation model, not data. |

The missingness is clearly **not MCAR** — communities with police data are systematically larger, more urban, and have higher crime rates than those without. Dropping the columns (not the rows) preserves the full sample while avoiding a heavily biased subsample.

#### `LemasPctOfficDrugUn` — A Hidden Concern

Although `LemasPctOfficDrugUn` had no missing values, its value is 0 for all 1,675 communities without police data. Its variation therefore mostly reflects police-data availability rather than substantive differences in policing — a subtle form of leakage through the missingness pattern. It was excluded from the model along with the other 22 police-related variables.

#### Dropping the 5 ID / Administrative Columns

`state`, `county`, `community`, `communityname`, and `fold` are identifiers, not predictors. They were excluded from the modelling dataset but **retained in a separate master dataset** (`communities_master.csv`) for later use:

- `fold` encodes the pre-made 10-fold cross-validation groups used in Task 5.
- `communityname` / `county` / `community` allow tracing outlier observations back to specific communities in Task 6 diagnostics.
- `state` could be used as fixed effects in a robustness extension.

#### `OtherPerCap` — Median Imputation

The single missing value in `OtherPerCap` was imputed with the column median. With only one missing observation the imputation barely moves the distribution, and the median is preferred over the mean for its robustness to the right tail typical of per-capita income variables.

#### Normalization

The creators' normalization (all predictors scaled to [0, 1]) is retained as-is. This makes coefficient magnitudes comparable across variables. However, normalization does not eliminate skewness, nonlinearity, or heteroscedasticity — these are assessed in Task 6.

#### Verification

After preprocessing:

- **Modelling dataset** (`communities_clean.csv`): **(1,994 × 100)** with **zero missing values**
- **Master dataset** (`communities_master.csv`): **(1,994 × 105)**; missing values confined to `county` and `community` (withheld for confidentiality in the raw UCI data) and deliberately retained

---

### Task 2 — Model Definition (Baseline Shortlist)

#### The Problem

After preprocessing, 99 candidate predictors remain. Using all of them in a classical linear model would be statistically and pedagogically unhelpful: collinearity dominates, interpretation breaks down, and most coefficients would not be individually significant.

#### The Approach — One Variable Per Construct

A theme-based correlation screen was used to construct a baseline shortlist of **7 predictors** — one per major theoretical construct. This is a preliminary selection; formal model selection using BIC, AIC, and cross-validation is performed in Task 5. Raw univariate correlation is not sufficient as a final selection criterion because it ignores multivariate redundancy and collinearity.

#### Chosen Variables

| Construct | Variable | Correlation with Target |
|---|---|---|
| Socioeconomic disadvantage | `PctPopUnderPov` | +0.522 |
| Education | `PctNotHSGrad` | +0.483 |
| Employment | `PctUnemployed` | +0.504 |
| Family disruption | `PctKids2Par` | −0.738 |
| Housing disorder | `PctVacantBoarded` | +0.483 |
| Urban scale | `PopDens` | +0.281 |
| Community size | `population` | +0.367 |

`racePctWhite` (correlation −0.685, the strongest demographic predictor) was excluded from the baseline to keep the model interpretable and to avoid difficult causal questions about the demographic–crime relationship. Immigration-related variables (`PctForeignBorn`, `PctImmigRec5`, `PctNotSpeakEnglWell`) were also dropped — their univariate correlations were all below ±0.30 and several were collinear with `PopDens`.

#### The Baseline Regression Equation

$$
\text{ViolentCrimesPerPop}_i = \beta_0 + \beta_1\,\text{PctPopUnderPov}_i + \beta_2\,\text{PctNotHSGrad}_i + \beta_3\,\text{PctUnemployed}_i + \beta_4\,\text{PctKids2Par}_i + \beta_5\,\text{PctVacantBoarded}_i + \beta_6\,\text{PopDens}_i + \beta_7\,\text{population}_i + \varepsilon_i
$$

#### The Classical Linear Model Assumptions

The following five core assumptions underlie OLS estimation, with normality noted separately:

1. **Linearity in parameters.** The conditional mean of the target is a linear function of the regression coefficients.
2. **Random sample / independent observations.** The 1,994 communities are treated as independent draws from the population of U.S. communities.
3. **No perfect multicollinearity.** No predictor is an exact linear combination of the others; the design matrix has full column rank so $\hat{\beta}$ is uniquely identified.
4. **Zero conditional mean of the error.** $E[\varepsilon \mid X] = 0$ — predictors carry no systematic information about the error. This is the exogeneity assumption that makes OLS unbiased.
5. **Homoscedasticity.** The error variance is constant across observations: $\text{Var}(\varepsilon \mid X) = \sigma^2$. This is what makes OLS efficient (BLUE) rather than merely unbiased.

**Normality of errors** is sometimes listed as a sixth assumption. It is not required for OLS to be unbiased or BLUE, but it is required for classical small-sample $t$- and $F$-tests to have exact distributions. With N = 1,994 the Central Limit Theorem makes the inference robust to non-normality, but residual normality is explicitly inspected in Task 6.

#### Limitations and Deferred Decisions

- **Outlier detection** is deferred to Task 6. ID columns are preserved in the master dataset so any flagged observation can be traced back to a specific community.
- **Target transformation** (e.g. $\sqrt{y}$, $\log y$) is not applied here. Task 6 diagnostics determine whether skewness or heteroscedasticity warrants transformation.
- A **disadvantage block** was identified: `PctPopUnderPov`, `PctNotHSGrad`, `PctUnemployed`, and `PctKids2Par` all correlate strongly with each other (|r| between 0.67 and 0.78). All four are retained in the baseline shortlist for Task 5 to evaluate, but model selection criteria and VIF analysis are anticipated to lead to dropping one or more.
- **State fixed effects** are not included in the baseline but could be added as a robustness extension.

---

## Section B — Estimation and Inference

> **A note on units.** The UCI dataset is normalized so each predictor lies in [0, 1], and every predictor has at least one community at 0.00 and at least one at 1.00 in the sample. A "one-unit increase in a normalized predictor" therefore corresponds to **moving from the community with the lowest observed value to the community with the highest observed value** — the *full empirical range*. Coefficients below represent large changes in predicted `ViolentCrimesPerPop`, not changes per percentage point.

### Task 3 — OLS Estimation and Coefficient Interpretation

The baseline OLS model was fitted with an intercept using `statsmodels`. Results are saved to `outputs/tables/coefficients.csv`, `outputs/tables/model_fit_summary.csv`, and `outputs/tables/ols_summary.txt`.

**Overall fit: R² = 0.587, Adj R² = 0.585, F(7, 1986) = 402.6, N = 1,994.**

#### Coefficient Table — Baseline 7-Variable OLS

| Term | Coef | p-value | 95% CI lower | 95% CI upper |
|---|---|---|---|---|
| const | 0.6752 | < 0.001 | — | — |
| PctPopUnderPov | −0.0918 | 0.001 | — | — |
| PctNotHSGrad | 0.0041 | 0.879 | −0.049 | +0.057 |
| PctUnemployed | 0.0383 | 0.217 | — | — |
| PctKids2Par | −0.7674 | < 0.001 | −0.824 | −0.710 |
| PctVacantBoarded | 0.0898 | < 0.001 | — | — |
| PopDens | 0.0744 | < 0.001 | — | — |
| population | 0.2724 | < 0.001 | — | — |

#### Coefficient Interpretations

**Intercept (`const = 0.6752`).** The predicted `ViolentCrimesPerPop` when every predictor equals 0 — a community at the dataset minimum on all seven predictors simultaneously, including `PctKids2Par = 0` (no kids in two-parent households). This is an unrealistic combination; the intercept is a regression anchor and should not be over-interpreted.

**`PctPopUnderPov` (coef = −0.0918, p = 0.001).** Univariate correlation with the target was +0.522, suggesting more poverty → more crime. However, in the multivariate model the partial coefficient is **negative** (−0.092). This sign reversal is a textbook multicollinearity symptom: within the disadvantage block, the variables are highly correlated (|r| between 0.67 and 0.78), and most of the shared "disadvantage" signal is absorbed by `PctKids2Par`, leaving `PctPopUnderPov` to estimate an unstable residual effect. Task 6 VIF analysis quantifies this formally.

**`PctNotHSGrad` (coef = +0.0041, p = 0.879).** The estimated effect is essentially zero across the full empirical range and is not statistically significant. The data does not support an independent effect of educational attainment once the other regressors are included.

**`PctUnemployed` (coef = +0.0383, p = 0.217).** A positive but statistically insignificant partial effect. We cannot reject the null hypothesis that this variable's contribution is zero in the presence of the other predictors.

**`PctKids2Par` (coef = −0.7674, p < 0.001).** The largest absolute coefficient in the model. Moving from the community with the fewest two-parent households to the community with the most is associated with a predicted **fall of 0.77** in `ViolentCrimesPerPop` — a very large *negative* effect, precisely estimated (95% CI: [−0.824, −0.710]).

**`PctVacantBoarded` (coef = +0.0898, p < 0.001).** Moving across the full empirical range is associated with a predicted increase of **+0.090** in violent crime — statistically significant, indicating a positive association with housing disorder.

**`PopDens` (coef = +0.0744, p < 0.001).** A predicted increase of **+0.074** from lowest to highest population density — statistically significant, showing a positive association between density and violent crime.

**`population` (coef = +0.2724, p < 0.001).** Moving from the smallest to the largest community is associated with a predicted increase of **+0.272** — large and statistically significant.

#### Diagnostic Note on Out-of-Range Predictions

OLS produces unbounded predictions, but the target is bounded in [0, 1]. Inspecting fitted values, **107 of 1,994 communities (5.4%) receive predictions below 0**, and 8 receive predictions above 1. This is a known limitation of OLS on a bounded outcome; Section C explores remedies.

---

### Task 4 — Hypothesis Testing and Confidence Intervals

> **Caveat on standard errors.** The residuals-vs-fitted plot shows a clear cone shape (visible heteroscedasticity). All standard errors, t-statistics, p-values, and confidence intervals in this section are based on the default non-robust covariance estimator which assumes homoscedasticity. Section C revisits inference using HC3 robust standard errors. The conclusions below should be read as conditional on the homoscedasticity assumption. The Jarque-Bera test (p < 1×10⁻¹⁰⁰) and Q-Q plot additionally indicate non-normal residuals, though the CLT makes asymptotic inference approximately valid at N = 1,994.

#### Individual t-Tests (5% Level)

- **Reject H₀** (statistically significant): `PctPopUnderPov` (negative — see multicollinearity discussion above), `PctKids2Par`, `PctVacantBoarded`, `PopDens`, `population`, and the intercept.
- **Fail to reject H₀**: `PctNotHSGrad` (p = 0.879) and `PctUnemployed` (p = 0.217).

#### 95% Confidence Intervals — Notable Cases

- **`PctKids2Par` CI: [−0.824, −0.710]** — extremely tight, entirely below zero. Signals a precisely estimated large negative effect.
- **`PctNotHSGrad` CI: [−0.049, +0.057]** — almost symmetric around zero. The data rules out a large effect in either direction; this is informative non-significance, not just noise.

#### Joint F-Tests

| Test | Restriction | F | df | p-value | Decision |
|---|---|---|---|---|---|
| Disadvantage block | `PctPopUnderPov = PctNotHSGrad = PctUnemployed = PctKids2Par = 0` | 326.63 | (4, 1986) | < 1×10⁻²⁰⁰ | Reject H₀ |
| Non-disadvantage controls | `PctVacantBoarded = PopDens = population = 0` | 56.95 | (3, 1986) | < 1×10⁻³⁰ | Reject H₀ |
| Non-significant pair | `PctNotHSGrad = PctUnemployed = 0` | 1.04 | (2, 1986) | 0.355 | Fail to reject H₀ |

The joint test on the two individually non-significant predictors (F(2, 1986) = 1.04, p = 0.355) fails to reject the null that both coefficients are zero. This directly informs Task 5: the data is consistent with a more parsimonious specification that drops `PctNotHSGrad` and `PctUnemployed`, a conclusion confirmed by the data-driven selection in Section C.

---

## Section C — Model Selection, Diagnostics, and Remedies

### Task 5 — Model Selection

#### 5.1 Selection Strategy

Starting from the 99 candidate predictors in `communities_clean.csv`, a two-stage pipeline is applied. First, a **BIC-penalised LASSO** (coordinate-descent, implemented in pure numpy/scipy — no sklearn dependency) narrows the field to 19 variables by choosing the regularisation strength that minimises the Bayesian Information Criterion. Second, **BIC backward elimination** removes one predictor at a time as long as BIC improves, yielding the final parsimonious 13-variable specification.

Section A's theory-led 7-variable shortlist is included as an explicit comparison candidate so the objective justification for the data-driven model can be verified directly. The Section B finding that `PctNotHSGrad` and `PctUnemployed` are jointly insignificant (F(2, 1986) = 1.04, p = 0.355) further supports replacing the theory-led disadvantage block with the data-driven selection.

#### 5.2 Candidate Model Comparison

| Model | p | Adj R² | AIC | BIC | Cond. No. | CV-RMSE | CV-MAE | CV-R² |
|---|---|---|---|---|---|---|---|---|
| OLS — all 99 predictors | 99 | 0.6782 | −2314.38 | −1754.59 | 915.34 | 0.1363 | 0.0960 | 0.6575 |
| OLS — Section A 7-variable | 7 | 0.5852 | −1897.37 | −1852.59 | 18.38 | 0.1505 | 0.1082 | 0.5825 |
| OLS — post-LASSO 19 variables | 19 | 0.6638 | −2304.82 | −2192.86 | 65.46 | 0.1361 | 0.0947 | 0.6587 |
| **OLS — final 13 variables** | **13** | **0.6619** | **−2299.32** | **−2220.95** | **42.93** | **0.1362** | **0.0944** | **0.6579** |

The full 99-predictor OLS achieves the highest in-sample fit but has a condition number of 915.3, indicating severe multicollinearity and weak interpretability. The final 13-variable model achieves the **best BIC (−2220.95)** — which explicitly penalises complexity — while maintaining nearly identical out-of-sample predictive performance to the post-LASSO 19-variable model.

#### 5.3 Divergence from Section A and Final Model Justification

Section A selected 7 predictors on theoretical grounds (one per construct). The data-driven pipeline selects 13 variables, overlapping with Section A only on `PctKids2Par`. The 7-variable model achieves Adj-R² = 0.585 and CV-R² = 0.583, versus **0.662 and 0.658** for the final 13-variable model — a meaningful improvement at the cost of 6 additional predictors. The BIC of the 7-variable model (−1852.6) is substantially worse than the final model's BIC (−2221.0), confirming the additional variables contribute genuine information rather than overfitting.

The key substantive additions over Section A's shortlist are: family-disruption indicators (`MalePctDivorce`, `PctIlleg`, `PctWorkMom`), dense-housing stress (`PctPersDenseHous`), housing vacancy/occupancy (`HousVacant`, `PctHousOccup`), homeownership cost burden (`MedOwnCostPctIncNoMtg`), and a street homelessness measure (`NumStreet`). Each survives BIC elimination, indicating genuine marginal predictive value.

**On `racepctblack`:** Section A excluded racial composition variables to avoid difficult causal interpretation. `racepctblack` is re-introduced here because LASSO-BIC selects it and omitting a substantively important predictor risks omitted-variable bias in remaining coefficients. The coefficient is interpreted as **purely associational** — it likely captures structural factors such as historical segregation or concentrated disadvantage, not an inherent trait. No causal interpretation is warranted without a proper identification strategy.

#### 5.4 Selected Variables

| LASSO-BIC (19 variables) | Final BIC Backward Elimination (13 variables) |
|---|---|
| racepctblack | racepctblack |
| racePctWhite | agePct12t29 |
| agePct12t29 | pctUrban |
| pctUrban | OtherPerCap |
| pctWInvInc | MalePctDivorce |
| AsianPerCap | PctKids2Par |
| OtherPerCap | PctWorkMom |
| MalePctDivorce | PctIlleg |
| PctKids2Par | PctPersDenseHous |
| PctWorkMom | HousVacant |
| PctIlleg | PctHousOccup |
| PctPersDenseHous | MedOwnCostPctIncNoMtg |
| HousVacant | NumStreet |
| PctHousOccup | |
| PctVacantBoarded | |
| MedRentPctHousInc | |
| MedOwnCostPctIncNoMtg | |
| NumStreet | |
| PctForeignBorn | |

#### 5.5 Final Model Equation

$$
\text{ViolentCrimesPerPop}_i = \beta_0 + \beta_1\,\text{racepctblack}_i + \beta_2\,\text{agePct12t29}_i + \beta_3\,\text{pctUrban}_i + \beta_4\,\text{OtherPerCap}_i + \beta_5\,\text{MalePctDivorce}_i + \beta_6\,\text{PctKids2Par}_i + \beta_7\,\text{PctWorkMom}_i + \beta_8\,\text{PctIlleg}_i + \beta_9\,\text{PctPersDenseHous}_i + \beta_{10}\,\text{HousVacant}_i + \beta_{11}\,\text{PctHousOccup}_i + \beta_{12}\,\text{MedOwnCostPctIncNoMtg}_i + \beta_{13}\,\text{NumStreet}_i + \varepsilon_i
$$

#### 5.6 Final Model — OLS Coefficients with HC3 Robust Inference

Heteroscedasticity is confirmed in Task 6 (Breusch-Pagan p ≈ 2.15×10⁻⁴⁸), so HC3 heteroscedasticity-consistent standard errors are used for all reported inference. Since all predictors are normalised to [0, 1], each coefficient represents the predicted change in `ViolentCrimesPerPop` from the community at the dataset minimum to the maximum on that predictor — the full empirical range.

| Variable | Coef | HC3 SE | t (HC3) | p (HC3) | 95% CI low | 95% CI high |
|---|---|---|---|---|---|---|
| const | 0.3535 | 0.0543 | 6.5155 | < 0.001 | 0.2471 | 0.4599 |
| racepctblack | 0.1695 | 0.0292 | 5.8001 | < 0.001 | 0.1122 | 0.2267 |
| agePct12t29 | −0.0917 | 0.0226 | −4.0653 | < 0.001 | −0.1360 | −0.0475 |
| pctUrban | 0.0378 | 0.0080 | 4.7259 | < 0.001 | 0.0221 | 0.0534 |
| OtherPerCap | 0.0482 | 0.0164 | 2.9405 | 0.003 | 0.0160 | 0.0803 |
| MalePctDivorce | 0.1097 | 0.0306 | 3.5886 | < 0.001 | 0.0497 | 0.1696 |
| PctKids2Par | −0.2953 | 0.0485 | −6.0829 | < 0.001 | −0.3905 | −0.2001 |
| PctWorkMom | −0.0794 | 0.0200 | −3.9643 | < 0.001 | −0.1186 | −0.0401 |
| PctIlleg | 0.2168 | 0.0458 | 4.7281 | < 0.001 | 0.1269 | 0.3067 |
| PctPersDenseHous | 0.1941 | 0.0233 | 8.3214 | < 0.001 | 0.1484 | 0.2399 |
| HousVacant | 0.1113 | 0.0381 | 2.9185 | 0.004 | 0.0365 | 0.1861 |
| PctHousOccup | −0.0637 | 0.0209 | −3.0417 | 0.002 | −0.1047 | −0.0226 |
| MedOwnCostPctIncNoMtg | −0.0578 | 0.0164 | −3.5166 | < 0.001 | −0.0901 | −0.0256 |
| NumStreet | 0.1808 | 0.0539 | 3.3567 | < 0.001 | 0.0752 | 0.2865 |

#### 5.7 Key Coefficient Interpretations

**`PctKids2Par` (coef = −0.2953, p < 0.001).** The coefficient is **negative** — communities with a higher share of children in two-parent households are associated with **lower** violent crime rates. Moving across the full empirical range reduces predicted crime by approximately 0.30. This is the largest absolute coefficient in the model and is precisely estimated (95% HC3 CI: [−0.391, −0.200]). This finding is consistent with Section B, where `PctKids2Par` carried a coefficient of −0.767 — also highly significant and the dominant predictor in the 7-variable model.

**`PctIlleg` (coef = +0.2168, p < 0.001).** Higher proportions of children born to never-married mothers are associated with higher violent crime (+0.217 across the full range), consistent with the broader family-instability literature.

**`PctPersDenseHous` (coef = +0.1941, p < 0.001).** Dense housing is positively associated with crime (+0.194). Combined with `HousVacant` (+0.111) and `PctHousOccup` (−0.064), this reveals that both housing crowding and vacancy/instability independently elevate predicted crime.

**`NumStreet` (coef = +0.1808, p < 0.001).** Street homelessness is positively associated with crime (+0.181), capturing concentrated social disadvantage not covered by the other housing variables.

**`MalePctDivorce` (coef = +0.1097, p < 0.001).** Higher male divorce rates are positively associated with crime (+0.110), consistent with family-disruption theories.

**`racepctblack` (coef = +0.1695, p < 0.001).** Interpreted as purely associational — likely reflects omitted structural conditions (historical segregation, concentrated disadvantage, institutional disinvestment) rather than any inherent characteristic.

**`agePct12t29` (coef = −0.0917, p < 0.001).** A negative partial association — after controlling for family structure, housing density, and other variables, a higher share of the 12–29 age group is associated with slightly lower predicted crime.

---

### Task 6 — Model Diagnostics and Remedies

#### 6.1 Multicollinearity — VIF and Condition Number

The full 99-predictor OLS has a condition number of 915.3, indicating severe multicollinearity. The final 13-variable model reduces this to **42.9** — a major improvement achieved by the LASSO-BIC screening and BIC backward elimination.

**Variance Inflation Factors — Final 13-Variable Model:**

| Variable | VIF |
|---|---|
| PctKids2Par | 9.121 |
| PctIlleg | 7.939 |
| racepctblack | 3.647 |
| MalePctDivorce | 2.882 |
| HousVacant | 2.115 |
| PctPersDenseHous | 1.775 |
| NumStreet | 1.678 |
| PctHousOccup | 1.591 |
| agePct12t29 | 1.330 |
| PctWorkMom | 1.303 |
| MedOwnCostPctIncNoMtg | 1.239 |
| pctUrban | 1.203 |
| OtherPerCap | 1.154 |

Most predictors have low-to-moderate VIFs. The two highest — `PctKids2Par` (9.12) and `PctIlleg` (7.94) — are elevated because both variables measure overlapping aspects of family structure. Both remain below the conventional threshold of 10 and are retained because their coefficients are substantively interpretable and precisely estimated.

#### 6.2 Heteroscedasticity — Breusch-Pagan Test

The **Breusch-Pagan F-test** strongly rejects homoscedasticity (F = 21.48, p ≈ 2.15×10⁻⁴⁸). This is consistent with Section B's observation of a cone-shaped residuals-vs-fitted plot. Residual variance increases with fitted values, visible as a fan shape in the diagnostic plots.

**Remedy applied:** HC3 heteroscedasticity-consistent standard errors are used for all reported coefficient-level inference. This corrects standard errors, t-statistics, p-values, and confidence intervals without changing point estimates, and is the appropriate implemented remedy for detected heteroscedasticity.

#### 6.3 Non-Normal Residuals — Jarque-Bera Test

The **Jarque-Bera statistic is 1351.46** (p ≈ 3.42×10⁻²⁹⁴), indicating strong departure from normality — consistent with Section B's findings (JB p < 1×10⁻¹⁰⁰). The Q-Q plot shows a sharp upper-tail deviation from the 45-degree reference line, indicating a right-skewed, heavy-tailed residual distribution. With N = 1,994, the Central Limit Theorem provides asymptotic validity for t- and F-tests despite non-normality, but the caveat is duly noted.

#### 6.4 Functional Form — Ramsey RESET Test

The **Ramsey RESET test** rejects the null of correct functional form (F(2, 1978) = 25.52, p < 0.001). This suggests possible nonlinear relationships or missing interactions. Future work could add quadratic or spline terms for the strongest continuous predictors — `PctKids2Par`, `PctPersDenseHous`, and `NumStreet`. Within the scope of classical linear regression, the current model serves as an informative baseline.

#### 6.5 Autocorrelation — Durbin-Watson

The **Durbin-Watson statistic is 2.000** — essentially the ideal value. Since these are cross-sectional community data, serial dependence is not a primary concern and is confirmed absent.

#### 6.6 Influential Observations — Cook's Distance and Leverage

Cook's distances are small throughout — the maximum is **0.0359**, well below the conventional threshold of 1. Influential-point community names are looked up from `communities_master.csv` (retained in Section A for exactly this purpose):

| Community | State | Cook's D | Leverage | Std. Residual |
|---|---|---|---|---|
| Philadelphiacity | PA (42) | 0.0359 | 0.0544 | −2.956 |
| FortLauderdalecity | FL (12) | 0.0239 | 0.0550 | −2.399 |
| WestHollywoodcity | CA (6) | 0.0163 | 0.0105 | +4.631 |
| Vernoncity | TX (48) | 0.0144 | 0.0059 | +5.827 |
| Gatesvillecity | TX (48) | 0.0140 | 0.0346 | −2.335 |

None of these observations dominates the model — no Cook's distance approaches 1. The data do not indicate any single disastrous outlier driving the overall findings.

#### 6.7 Diagnostic Tests Summary

| Test | Model | Statistic | p-value | Conclusion |
|---|---|---|---|---|
| Breusch-Pagan (F) | Baseline OLS | 21.48 | 2.15×10⁻⁴⁸ | Heteroscedasticity present |
| Breusch-Pagan (F) | log(Y+c) remedy | 5.22 | 2.91×10⁻⁹ | Heteroscedasticity present (reduced) |
| Jarque-Bera | Baseline OLS | 1351.46 | 3.42×10⁻²⁹⁴ | Non-normal residuals |
| Jarque-Bera | log(Y+c) remedy | 2433.66 | ≈ 0 | Non-normal residuals (worsened in lower tail) |
| RESET | Baseline OLS | 25.52 | 1.14×10⁻¹¹ | Misspecification detected |
| RESET | log(Y+c) remedy | 70.78 | 2.04×10⁻³⁰ | Misspecification detected (worsened) |
| Durbin-Watson | Baseline OLS | 2.000 | — | No serial dependence |
| Durbin-Watson | log(Y+c) remedy | 1.982 | — | No serial dependence |

#### 6.8 Remedy — Log Transformation of the Dependent Variable

To address the right-skewness and heteroscedasticity identified above, a remedial model is estimated using:

$$
\log(Y + c)
$$

where $Y = \texttt{ViolentCrimesPerPop}$ and $c = 0.001$ (half the smallest positive Y value, added because the response includes zeros). The same 13 predictors are retained so diagnostic changes are attributable to the transformation rather than to a different feature set.

**Rationale:** The original response is bounded in [0, 1] and heavily concentrated near lower values with a long right tail — precisely the setting where a log transformation is a standard remedy, as it compresses large outcome values more than small ones. Section B flagged both heteroscedasticity and non-normality and deferred this exploration to Section C.

#### 6.9 Evaluation of the Remedy

**Diagnostic Comparison — Baseline OLS vs log(Y+c) Remedy:**

| Model | Adj R² | BP F-pval | JB stat | RESET p | DW | Max Cook | N Cook > 4/n |
|---|---|---|---|---|---|---|---|
| Baseline OLS | 0.6619 | 2.15×10⁻⁴⁸ | 1351.46 | < 0.001 | 2.000 | 0.0359 | 164 |
| log(Y+c) OLS | 0.6091 | 2.91×10⁻⁹ | 2433.66 | < 0.001 | 1.982 | 0.0284 | 78 |

**Heteroscedasticity:** BP F-pval improves from 2.15×10⁻⁴⁸ to 2.91×10⁻⁹ — substantially reduced but not eliminated. The fan shape in the residuals-vs-fitted plot is attenuated, and the Scale-Location plot is more even across the middle of the fitted-value range.

**Normality:** Jarque-Bera statistic worsens (1,351 → 2,434) because the log-transformation of a response that includes many near-zero values creates a new strong left-tail distortion. The Q-Q plot improves in the centre and upper tail (the sharp rightward bend in the baseline model is substantially reduced) but degrades at the extreme lower tail.

**Influential observations:** Maximum Cook's distance falls from 0.0359 to 0.0284. Maximum leverage is unchanged (~0.079) since leverage depends only on the predictor matrix. The number of observations above the 4/n Cook's distance threshold falls from **164 to 78** — a meaningful reduction in moderately influential points.

**Functional form:** The RESET F-statistic worsens considerably (25.52 → 70.78), indicating the log transformation introduces additional misspecification, likely due to the discrete mass of near-zero values in the response.

**Conclusion on the remedy:** The log model is useful as a robustness check and partially addresses heteroscedasticity and influential observations, but it does not fully resolve all assumption violations, worsens normality in the lower tail, complicates interpretation (coefficients describe variation in log-crime rather than original-scale crime), and fails the RESET test more severely. The **primary model remains the 13-variable OLS with HC3 robust standard errors**, and the log-transformed model serves as a diagnostic robustness check.

---

## Overall Conclusion

The 13-variable OLS model selected by LASSO-BIC screening and BIC backward elimination from 99 cleaned predictors explains approximately **66% of the variance** in normalized violent crime (Adj-R² = 0.662) and generalises well in 10-fold cross-validation (CV-R² = 0.658, CV-RMSE = 0.136). It achieves the **best BIC (−2221.0)** among all OLS candidates, clearly outperforms Section A's theory-led 7-variable shortlist (Adj-R² = 0.585, BIC = −1852.6), and substantially reduces multicollinearity (condition number: 42.9 vs. 915.3 for the full model).

**Family structure and housing distress are the strongest predictors of violent crime in this dataset.** The key substantive findings are:

- `PctKids2Par` carries the largest coefficient (−0.295) with a **negative** relationship — communities with higher shares of children in two-parent households have significantly lower predicted violent crime. This finding is robust across both the 7-variable baseline (coef = −0.767) and the final 13-variable model.
- `PctIlleg` (+0.217), `PctPersDenseHous` (+0.194), `NumStreet` (+0.181), and `MalePctDivorce` (+0.110) all show strong positive associations, reflecting the roles of family instability, housing crowding, homelessness, and community social stress.
- Housing instability is further captured by `HousVacant` (+0.111) and `PctHousOccup` (−0.064) acting jointly, showing that both crowding and vacancy elevate predicted crime.

**Diagnostics confirmed three assumption violations**, each addressed with an appropriate remedy:

1. **Heteroscedasticity** (Breusch-Pagan p ≈ 10⁻⁴⁸) → **Remedy:** HC3 robust standard errors applied throughout
2. **Non-normal residuals** (Jarque-Bera stat = 1,351) → Mitigated by large N via CLT; log-transformation explored as robustness check
3. **Functional form misspecification** (RESET p < 0.001) → Noted as a limitation; future work could add nonlinear terms for the strongest predictors

There is **no evidence of autocorrelation** (Durbin-Watson = 2.000) and **no single dominating outlier** (max Cook's D = 0.036, well below the threshold of 1).

Law-enforcement variables contribute little predictive power due to their high missingness rates (84%), consistent with Section A's preprocessing rationale and the decision to drop the 22 LEMAS police-related columns.

**The final recommended model is the 13-variable OLS with HC3 robust standard errors.** All reported coefficients, standard errors, t-statistics, p-values, and confidence intervals use HC3 inference. The log-transformed model is retained as a robustness check confirming the direction and approximate magnitude of the main substantive findings.
