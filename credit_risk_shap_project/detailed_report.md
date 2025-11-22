
# Interpretable Machine Learning for Credit Risk - Strategic Analysis

### Model & Preprocessing
- Model: LightGBM with binary objective trained with 5-fold stratified cross-validation and early stopping.
- Preprocessing: Numeric median imputation; categorical most-frequent imputation + one-hot encoding.

### Global Explanation (SHAP)
- The SHAP analysis highlights the features that, on average, contribute most to predicted default probability.
- Actionable interpretation for risk management committee:
  - If 'interest_rate' and 'revol_util' are top contributors, consider changes to underwriting (e.g., stricter limits for high revol_util borrowers).
  - If 'annual_inc' has strong negative SHAP (reduces default probability), ensure verification of income is robust.

### Local Explanations (three loan applications)
- High-risk: Top positive SHAP contributors (e.g., high interest_rate, high revol_util, recent delinquencies) indicate why the model predicts high default probability.
- Low-risk: Top negative SHAP contributors (e.g., high annual_inc, low dti, many open accounts with clean history) explain low predicted probability.
- Borderline: Balanced contributions; suggest manual review or additional documents.

### Feature interactions
- Use SHAP dependence plots to identify interactions; e.g., how 'revol_util' effect changes by 'dti' level.
- If interaction shows compounding risk, policy may require different thresholds when both variables are high.

### Limitations & Next steps
- Synthetic dataset used here — results are demonstration-only. For production, run pipeline on real, cleaned dataset and calibrate thresholds.
- Consider adding monotonic constraints, calibration (Platt/Isotonic), and fairness checks before deployment.
