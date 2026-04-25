# Section B — Task 3: Plain-English Interpretation of the OLS Results

This section interprets the baseline OLS regression for `ViolentCrimesPerPop` using the shared dataset `data/communities_baseline_model.csv`. Because the variables in this project are normalized to the `[0, 1]` scale, each coefficient should be interpreted as the change in predicted `ViolentCrimesPerPop` associated with a one-unit increase in the normalized predictor, holding all other variables constant.

## Coefficient interpretations

- **Intercept (`const = 0.6752`)**  
  When all predictors are equal to 0, the model predicts `ViolentCrimesPerPop = 0.6752`. This serves mainly as a baseline reference point rather than the most substantively important result.

- **`PctPopUnderPov` (`coef = -0.0918`, `p = 0.0010`)**  
  Holding the other predictors fixed, a one-unit increase in the normalized poverty measure is associated with a **0.0918 decrease** in predicted violent crime. This coefficient is statistically significant at the 5% level. However, this negative sign should be interpreted cautiously because the disadvantage-related variables are highly correlated, so multicollinearity may be affecting the sign and size of this estimate.

- **`PctNotHSGrad` (`coef = 0.0041`, `p = 0.8789`)**  
  Holding the other predictors fixed, a one-unit increase in the normalized share of people without a high-school diploma is associated with only a **0.0041 increase** in predicted violent crime. This effect is very small and not statistically significant, so the model does not provide evidence of an independent effect for this variable once the other regressors are included.

- **`PctUnemployed` (`coef = 0.0383`, `p = 0.2168`)**  
  Holding the other predictors fixed, a one-unit increase in the normalized unemployment measure is associated with a **0.0383 increase** in predicted violent crime. The coefficient is positive, but it is not statistically significant at the 5% level, so it should not be treated as strong evidence of an independent relationship in this model.

- **`PctKids2Par` (`coef = -0.7674`, `p < 0.001`)**  
  Holding the other predictors fixed, a one-unit increase in the normalized share of kids living in two-parent households is associated with a **0.7674 decrease** in predicted violent crime. This is the largest coefficient in absolute value and is highly statistically significant, suggesting a strong negative association with violent crime in the fitted model.

- **`PctVacantBoarded` (`coef = 0.0898`, `p < 0.001`)**  
  Holding the other predictors fixed, a one-unit increase in the normalized share of vacant boarded housing is associated with a **0.0898 increase** in predicted violent crime. This coefficient is statistically significant, indicating a meaningful positive relationship.

- **`PopDens` (`coef = 0.0744`, `p < 0.001`)**  
  Holding the other predictors fixed, a one-unit increase in normalized population density is associated with a **0.0744 increase** in predicted violent crime. This coefficient is statistically significant, showing a positive association between density and violent crime in the model.

- **`population` (`coef = 0.2724`, `p < 0.001`)**  
  Holding the other predictors fixed, a one-unit increase in normalized population size is associated with a **0.2724 increase** in predicted violent crime. This is a large, positive, and statistically significant coefficient, suggesting that larger communities tend to have higher predicted violent crime, conditional on the other variables in the model.

## Overall interpretation

The OLS results show that `PctKids2Par`, `PctVacantBoarded`, `PopDens`, and `population` are statistically significant predictors of violent crime in the baseline model. `PctNotHSGrad` and `PctUnemployed` are not statistically significant. `PctPopUnderPov` is statistically significant, but its negative sign should be interpreted carefully because the disadvantage-related regressors are highly correlated, which may create multicollinearity and unstable coefficient signs.

Overall, the model explains about **58.7%** of the variation in `ViolentCrimesPerPop` (`R^2 = 0.5866`), which suggests a reasonably strong fit for a baseline econometric model.
