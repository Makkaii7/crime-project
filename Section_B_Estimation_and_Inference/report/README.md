# Section B — Estimation and Inference
**Owner: Mohamed**
**Tasks: 3 (Parameter Estimation) and 4 (Hypothesis Testing & Confidence Intervals)**

## What to do
- **Task 3:** Fit an OLS regression using `../../data/communities_baseline_model.csv` as input. Use statsmodels so you get coefficients, standard errors, t-stats, and p-values. Interpret each coefficient in plain English.
- **Task 4:** Use the t-tests from your regression output (individual significance). Run joint F-tests on meaningful groups of variables. Construct and interpret 95% confidence intervals for each coefficient.

## Relevant lectures
- L3, L4: OLS estimation, statistical properties, Gauss-Markov
- L5: Residuals, hypothesis testing, t-tests, F-tests, confidence intervals

## Inputs
`data/communities_baseline_model.csv` — 7 predictors + target, already cleaned.

## Expected outputs in your folder
- `scripts/03_estimation.py` and `scripts/04_inference.py`
- `outputs/tables/coefficients.csv` (coefficient table with SE, t, p-values, CIs)
- `report/02_section_B.md` (your writeup)

## Important context from Section A
Read `../../Section_A_Preprocessing_and_Model_Definition/report/01_section_A.md` first. The Limitations section notes that the disadvantage-block predictors (PctPopUnderPov, PctNotHSGrad, PctUnemployed, PctKids2Par) are highly correlated with each other. Expect some coefficients in this group to have unstable signs or high standard errors — that's a multicollinearity symptom Section C will quantify with VIF.
