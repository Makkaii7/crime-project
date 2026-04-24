# Section C — Model Selection and Diagnostics
**Owner: Ayesha**
**Tasks: 5 (Model Selection) and 6 (Model Diagnostics)**

## What to do
- **Task 5:** Compare multiple candidate models (e.g., full 7-predictor model, reduced models dropping some disadvantage-block variables, maybe a model with an interaction). Compare using Adjusted R², AIC, BIC, and cross-validation (you can use the `fold` column in `communities_master.csv` for reproducible 10-fold CV). Justify your final model choice.
- **Task 6:** Check all assumptions of the CLM. Run VIF (multicollinearity), Breusch-Pagan or White test (heteroscedasticity), Q-Q plot (normality of residuals), residuals vs fitted plot (linearity, homoscedasticity), Durbin-Watson test (autocorrelation — though this is cross-sectional so less critical). Identify outliers and high-leverage points (Cook's distance). Propose remedies for any issues.

## Relevant lectures
- L6: Model selection (AIC, BIC, Adjusted R², CV, forward/backward/stepwise)
- L7, L8: Diagnostics (residual plots, transformations, autocorrelation, Q-Q, collinearity, outliers, leverage, Cook's distance)

## Inputs
- `../../data/communities_baseline_model.csv` — baseline 7 predictors + target
- `../../data/communities_master.csv` — full dataset with IDs and the `fold` column, used for CV and for tracing outliers back to specific communities
- Mohamed's fitted model from Section B (you'll need to coordinate)

## Expected outputs in your folder
- `scripts/05_model_selection.py` and `scripts/06_diagnostics.py`
- `outputs/tables/model_comparison.csv` (AIC/BIC/Adj R²/CV RMSE for each candidate)
- `outputs/plots/` (residual plots, Q-Q plot, leverage plots)
- `report/03_section_C.md` (your writeup)

## Important context from Section A
Read `../../Section_A_Preprocessing_and_Model_Definition/report/01_section_A.md` first. The Limitations section already flags:
- Multicollinearity in the disadvantage block — expect high VIFs here
- Target may need transformation if diagnostics show heteroscedasticity or non-normal residuals
- State fixed effects and racePctWhite as possible robustness checks
