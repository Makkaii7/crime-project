# Section B — Estimation and Inference (Tasks 3 and 4)

This section reports the OLS estimation of the baseline model defined in
Section A and the associated hypothesis tests and confidence intervals.
The input is the shared baseline dataset
`data/communities_baseline_model.csv` (1,994 communities, 7 predictors +
target).

A note on units used throughout. The UCI dataset is normalized so each
predictor lies in `[0, 1]`, and every predictor has at least one
community at 0.00 and at least one at 1.00 in our sample. So a
"one-unit increase in a normalized predictor" corresponds to **moving
from the community with the lowest observed value to the community with
the highest observed value** in the dataset. This is a very large
movement — the *full empirical range* — so coefficients in the table
below are large changes in predicted `ViolentCrimesPerPop`, not changes
per percentage point.

## Task 3 — Estimation and Coefficient Interpretation

We fit the baseline OLS model with an intercept using `statsmodels`.
The results are saved to
`outputs/tables/coefficients.csv`,
`outputs/tables/model_fit_summary.csv`, and
`outputs/tables/ols_summary.txt`. Overall fit: **R² = 0.587**,
**Adj R² = 0.585**, **F(7, 1986) = 402.6**, **N = 1,994**.

### Coefficient interpretations

- **Intercept (`const = 0.6752`).**
  Mathematically, this is the predicted `ViolentCrimesPerPop` when every
  predictor equals 0. In this data that corresponds to a community at
  the dataset minimum on *all seven* predictors simultaneously —
  including `PctKids2Par = 0` (no kids in two-parent households), which
  is an unrealistic combination. The intercept is therefore not a
  meaningful "baseline community"; it is a regression anchor and should
  not be over-interpreted.

- **`PctPopUnderPov` (`coef = -0.0918`, `p = 0.0010`).**
  In the univariate analysis from Section A, `PctPopUnderPov` had a
  positive correlation with violent crime (+0.522), suggesting more
  poverty is associated with more crime — the expected direction.
  However, in the multivariate model the partial coefficient is **−0.092**
  (statistically significant at the 1% level), indicating the opposite
  direction once other variables are controlled for. This is a textbook
  multicollinearity symptom. Within the disadvantage block
  (`PctPopUnderPov`, `PctNotHSGrad`, `PctUnemployed`, `PctKids2Par`),
  the four variables are highly correlated (|r| between 0.67 and 0.78).
  Most of the shared "disadvantage" signal is absorbed by `PctKids2Par`,
  which has a strong negative coefficient (−0.767) with a tight
  confidence interval. `PctPopUnderPov` ends up estimating a residual
  effect after `PctKids2Par` has soaked up the main pattern, which
  produces an unstable and counter-intuitive sign. The Section A
  Limitations section flagged this risk explicitly, and Task 6 VIF
  analysis will quantify it formally.

- **`PctNotHSGrad` (`coef = 0.0041`, `p = 0.8789`).**
  Because `PctNotHSGrad` is normalized to `[0, 1]` and both endpoints
  are observed, the coefficient measures the predicted change in
  `ViolentCrimesPerPop` from the community with the lowest share of
  non-graduates to the community with the highest. The estimated
  effect is **+0.0041 across that full range** — essentially zero —
  and is not statistically significant (p = 0.88). The data does not
  support an independent effect of `PctNotHSGrad` once the other
  regressors are included.

- **`PctUnemployed` (`coef = 0.0383`, `p = 0.2168`).**
  Because `PctUnemployed` is normalized to `[0, 1]`, the coefficient
  measures the change in predicted `ViolentCrimesPerPop` from the
  community with the lowest unemployment to the highest. The estimate
  is **+0.0383 across that full range** with p = 0.22, so we cannot
  reject the null hypothesis that this variable's partial effect is
  zero in the presence of the other predictors.

- **`PctKids2Par` (`coef = -0.7674`, `p < 0.001`).**
  Because `PctKids2Par` is normalized to `[0, 1]`, a one-unit increase
  represents moving from the community with the fewest two-parent
  households in the dataset to the community with the most. So the
  coefficient of −0.767 means that across that full range, predicted
  `ViolentCrimesPerPop` falls by **about 0.77 — a very large effect**,
  the largest in absolute value in the model and highly statistically
  significant. The 95% CI [−0.824, −0.710] is tight, signalling a
  precisely estimated negative association.

- **`PctVacantBoarded` (`coef = 0.0898`, `p < 0.001`).**
  Because `PctVacantBoarded` is normalized to `[0, 1]`, the coefficient
  measures the change in predicted `ViolentCrimesPerPop` from the
  community with the lowest share of vacant boarded housing to the
  highest. The estimate is **+0.090 across that full range**,
  statistically significant, indicating a positive association with
  housing disorder once other predictors are held fixed.

- **`PopDens` (`coef = 0.0744`, `p < 0.001`).**
  Because `PopDens` is normalized to `[0, 1]`, the coefficient is the
  predicted change in `ViolentCrimesPerPop` from the community with
  the lowest population density to the highest. The estimate is
  **+0.074 across that full range**, statistically significant,
  showing a positive association between density and violent crime
  in the model.

- **`population` (`coef = 0.2724`, `p < 0.001`).**
  Because `population` is normalized to `[0, 1]`, the coefficient is
  the predicted change in `ViolentCrimesPerPop` from the smallest
  community in the dataset to the largest. The estimate is **+0.272
  across that full range**, large and statistically significant,
  suggesting that larger communities have higher predicted violent
  crime conditional on the other variables in the model.

### Diagnostic note on out-of-range predictions

OLS produces unbounded predictions, but the target `ViolentCrimesPerPop`
is bounded in `[0, 1]`. Inspecting the fitted values, **107 of 1,994
communities (5.4%) receive predictions below 0**, and a small number
(8) receive predictions above 1. This is a known limitation of using
OLS on a bounded outcome. We retain OLS here for inferential
transparency, and Section C will explore remedies (target transformation
or alternative model forms).

## Task 4 — Hypothesis Testing and Confidence Intervals

This task uses the t-statistics, p-values, and 95% confidence intervals
already produced in Task 3 (see `outputs/tables/coefficients.csv`),
plus joint F-tests run in `04_inference.py`. The standard errors come
from the default OLS covariance estimator
(`Covariance Type: nonrobust` in the OLS summary), which assumes
homoscedasticity — see the diagnostic caveats below.

### Diagnostic caveats on the inference

The residuals-vs-fitted plot (`outputs/plots/residuals_vs_fitted.png`)
shows a clear cone shape — the residual variance increases with the
fitted value. This is visible heteroscedasticity. All standard errors,
t-statistics, p-values, and confidence intervals reported in this
section are based on the default non-robust covariance estimator
(`Covariance Type: nonrobust` in our OLS summary), which assumes
homoscedasticity. Section C will revisit this with
heteroscedasticity-consistent (HC) standard errors and may revise the
inference. For now, the inferential conclusions should be read as
conditional on this assumption.

The Q-Q plot (`outputs/plots/qq_plot_residuals.png`) and the
Jarque-Bera test (p < 1e-100 in the OLS summary) both indicate that
residuals are non-normal — there is meaningful skewness and excess
kurtosis. With N = 1994, the Central Limit Theorem makes the t-tests
and F-tests approximately valid even under non-normality, so the
inference is not invalidated. However, this caveat should be noted,
and Section C will examine whether a transformation of the target
reduces the departure from normality.

### Individual t-tests

Per-coefficient decisions at the 5% level are saved to
`outputs/tables/t_test_decisions.csv`. Summary:

- **Reject H₀** (significant at 5%): `PctPopUnderPov` (negative — see
  the multicollinearity discussion above), `PctKids2Par`,
  `PctVacantBoarded`, `PopDens`, `population`, and the intercept.
- **Fail to reject H₀**: `PctNotHSGrad` (p = 0.879) and
  `PctUnemployed` (p = 0.217).

### 95% Confidence intervals

CIs are constructed using the t-distribution with df = 1986 (saved per
coefficient in `outputs/tables/coefficients.csv` and interpreted
plain-English in `outputs/tables/confidence_interval_interpretations.csv`).
Two observations on width:

- `PctKids2Par`'s CI is **[−0.824, −0.710]** — extremely tight,
  signalling a precisely estimated effect.
- `PctNotHSGrad`'s CI is **[−0.049, +0.057]** — almost symmetric
  around zero. The data essentially rules out a large effect in
  either direction, rather than confirming a small one.

### Joint F-tests

We run three joint F-tests (full results in
`outputs/tables/joint_f_tests.csv`):

| Test | Restriction | F | df | p | Decision |
|---|---|---|---|---|---|
| Disadvantage block | `PctPopUnderPov = PctNotHSGrad = PctUnemployed = PctKids2Par = 0` | 326.63 | (4, 1986) | < 1e-200 | Reject H₀ |
| Non-disadvantage controls | `PctVacantBoarded = PopDens = population = 0` | 56.95 | (3, 1986) | < 1e-30 | Reject H₀ |
| Non-significant pair | `PctNotHSGrad = PctUnemployed = 0` | 1.04 | (2, 1986) | 0.355 | Fail to reject H₀ |

In addition to the block-level F-tests, we tested whether the two
individually non-significant predictors (`PctNotHSGrad`,
`PctUnemployed`) can be jointly excluded from the model. The result
is F(2, 1986) = 1.04, p = 0.355. We fail to reject the null hypothesis
that both coefficients are zero. This is informative for Task 5 model
selection: the data is consistent with a more parsimonious specification
that drops these two variables.
