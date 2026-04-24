# Section A — Preprocessing and Model Definition

## Task 1 — Preprocessing

### The missing-data problem

The raw UCI Communities and Crime dataset contains 1,994 communities and 128
columns. Missing values are concentrated in two places:

1. **22 police-related variables** (from the LEMAS survey) are missing for
   1,675 communities — approximately 84% of the sample. These are not
   scattered missing values; they are systematically absent for communities
   that were not surveyed.
2. **`OtherPerCap`** has exactly one missing value.

We also identified a third, more subtle issue discussed below
(`LemasPctOfficDrugUn`).

### Handling the 22 LEMAS police columns — options considered

| Option | Description | Why rejected |
|--------|-------------|--------------|
| Drop columns | Remove the 22 variables; keep all 1,994 communities. | **Chosen.** |
| Drop rows | Keep the variables; restrict the sample to the 319 communities with LEMAS data. | Reduces N by 84% and biases the sample toward communities that were surveyed. The rule of thumb of 10–20 observations per predictor is still satisfied with the retained variables, so we do not need these columns to specify a well-powered model. |
| Impute | Fill the 84% of missing entries with a statistical method. | Requires fabricating values for the overwhelming majority of observations. Any imputation here would be driven almost entirely by the imputation model, not by data. |

### Why we dropped the columns (and not the rows)

The missingness is clearly not MCAR — communities with police data are
systematically larger, more urban, and have higher crime rates than those
without. We therefore excluded these variables from the baseline model
rather than restricting the sample to 319 highly selected communities.

### `LemasPctOfficDrugUn` — a hidden concern

While reviewing the remaining police-related variables, we identified
`LemasPctOfficDrugUn` as a hidden concern. Although it had no missing
values, its value is 0 for all 1,675 communities without police data,
meaning its variation mostly reflects police-data availability rather than
substantive differences in policing. Including it would introduce subtle
bias through the missingness pattern, so we excluded it from the baseline
model.

### Dropping the 5 ID / administrative columns

`state`, `county`, `community`, `communityname`, and `fold` are identifiers,
not predictors. Including them in a regression on `ViolentCrimesPerPop`
would either be meaningless (e.g. a numeric community code) or improper
(community names as free-form strings).

However, we keep them in a **master** dataset
(`data/communities_master.csv`) separate from the modelling dataset. They
will be useful later:

- `fold` encodes the pre-made 10-fold cross-validation groups used for model
  selection in Task 5 (Section C).
- `communityname` / `county` / `community` let us trace specific rows back
  to a named community during outlier investigation (Task 6 diagnostics).
- `state` could enter a robustness check as fixed effects.

The modelling dataset (`data/communities_clean.csv`) excludes these five
columns and feeds Task 2.

### `OtherPerCap` — median imputation

A single missing value in `OtherPerCap` was imputed with the column median.
With only one missing observation, the imputation barely moves the
distribution, and the median is preferred over the mean because it is more
robust to the right tail present in per-capita income variables.

### Normalization

We retain the creators' normalization because it makes coefficient
magnitudes comparable. However, normalization does not eliminate skewness,
nonlinearity, or heteroscedasticity — these will be assessed in diagnostics
(Task 6).

### Verification

After preprocessing:

- Modelling dataset: **(1994, 100)** with **zero missing values**.
- Master dataset: **(1994, 105)**; missing values are confined to `county`
  and `community` (withheld for confidentiality in the raw UCI data) and are
  deliberately retained.

---

## Task 2 — Model Definition (Baseline Shortlist)

### The problem

After preprocessing we are left with 99 candidate predictors. Using all of
them in a classical linear model would be statistically and pedagogically
unhelpful: collinearity dominates, interpretation breaks down, and most
coefficients would not be individually significant.

### The approach — one variable per construct

We used a theme-based correlation screen to construct a baseline shortlist
of 7 predictors — one per major construct (socioeconomic disadvantage,
education, employment, family disruption, housing disorder, urban scale,
community size). This is a preliminary selection; formal model selection
using BIC, AIC, and cross-validation is performed in Task 5 (Section C).
Raw univariate correlation is not sufficient as a final selection criterion
because it ignores multivariate redundancy and collinearity.

### Chosen variables

| Construct | Variable | Correlation with target |
|-----------|----------|-------------------------|
| Socioeconomic disadvantage | `PctPopUnderPov`    | +0.522 |
| Education                  | `PctNotHSGrad`      | +0.483 |
| Employment                 | `PctUnemployed`     | +0.504 |
| Family disruption          | `PctKids2Par`       | −0.738 |
| Housing disorder           | `PctVacantBoarded`  | +0.483 |
| Urban scale                | `PopDens`           | +0.281 |
| Community size             | `population`        | +0.367 |

Correlations are computed in `scripts/02_model_definition.py` and verified
against these values at runtime.

`racePctWhite` was considered (correlation −0.685, strongest in the
demographic variables) but excluded from the baseline to keep the model
interpretable and to avoid difficult causal questions about the
demographic–crime relationship. It will be included as a sensitivity check
if time allows.

### Why the immigration construct was dropped

We initially considered an "immigration" construct (variables such as
`PctForeignBorn`, `PctImmigRec5`, `PctNotSpeakEnglWell`). All of these had
univariate correlations with the target below ±0.30 (the strongest was
about +0.30), and several were collinear with urban-scale measures already
represented by `PopDens`. We dropped the construct from the shortlist
rather than include a weak predictor that would be easily dominated by
better-motivated variables.

### The baseline regression equation

$$
\text{ViolentCrimesPerPop}_i
= \beta_0
+ \beta_1 \, \text{PctPopUnderPov}_i
+ \beta_2 \, \text{PctNotHSGrad}_i
+ \beta_3 \, \text{PctUnemployed}_i
+ \beta_4 \, \text{PctKids2Par}_i
+ \beta_5 \, \text{PctVacantBoarded}_i
+ \beta_6 \, \text{PopDens}_i
+ \beta_7 \, \text{population}_i
+ \varepsilon_i
$$

### The 5 Classical Linear Model assumptions

1. **Linearity in parameters.** The conditional mean of the target is a
   linear function of the regression coefficients. (Predictors can still
   enter via transformations, but the coefficients themselves multiply the
   covariates linearly.)
2. **Random sample / independent observations.** The 1,994 communities are
   treated as independent draws from the population of U.S. communities.
3. **No perfect multicollinearity.** No predictor is an exact linear
   combination of the others; the design matrix has full column rank so
   $\hat{\beta}$ is uniquely identified.
4. **Zero conditional mean of the error.** $E[\varepsilon \mid X] = 0$ —
   predictors carry no systematic information about the error. This is the
   exogeneity assumption that makes OLS unbiased.
5. **Homoscedasticity.** The error variance is constant across observations:
   $\text{Var}(\varepsilon \mid X) = \sigma^2$. This is what makes OLS
   efficient (BLUE) rather than just unbiased.

Normality of the errors is sometimes listed as a sixth assumption. It is
not required for OLS to be unbiased or BLUE, but it is required for the
classical small-sample $t$- and $F$-tests to have exact distributions.
With N = 1,994 the central limit theorem makes the inference robust to
non-normality, but we will still inspect residual normality in Task 6.

---

## Limitations and Deferred Decisions

- **Outlier detection** is deferred to Task 6 (diagnostics). We do not
  remove outliers mechanically at preprocessing, and we preserve ID
  columns in the master dataset so that any flagged observation can be
  traced back to a specific community.
- **Target transformation** (e.g. $\sqrt{y}$, $\log y$) is not applied here.
  Diagnostics in Task 6 will reveal whether skewness or heteroscedasticity
  warrants transformation.
- We identified a "disadvantage block" within the shortlist: PctPopUnderPov,
  PctNotHSGrad, PctUnemployed, and PctKids2Par all correlate strongly with
  each other (|r| between 0.67 and 0.78). This is expected — these
  variables are different facets of a single underlying construct
  (community disadvantage). We retained all four in the baseline shortlist
  for Task 5 to evaluate, but we anticipate that model selection criteria
  (AIC, BIC, cross-validation) and VIF analysis in Task 6 will likely lead
  to dropping one or two of them. The remaining three predictors
  (PctVacantBoarded, population, PopDens) are cleanly separated from the
  disadvantage block and from each other.
- **State fixed effects** are not included in the baseline. They could be
  added as a robustness extension using the `state` column retained in the
  master dataset.
- **Sensitivity model including `racePctWhite`** could be estimated to
  check how demographic composition affects the results; this is a
  candidate robustness check for Section C.
