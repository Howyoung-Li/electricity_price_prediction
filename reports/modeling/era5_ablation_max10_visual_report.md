# ERA5 Ablation Max10 Visual Report

## Files

- Figure directory: `reports/figures/era5_ablation_max10/`
- Best-model prediction table: `reports/modeling/era5_ablation_max10_best_model_predictions.csv`
- SHAP importance tables:
  - `reports/modeling/era5_ablation_max10_shap_importance_with_era5_xgboost.csv`
  - `reports/modeling/era5_ablation_max10_shap_importance_without_era5_catboost.csv`
- Feature drift table: `reports/modeling/era5_ablation_max10_feature_drift_psi.csv`

## Main Figures

![Hourly trend](../figures/era5_ablation_max10/best_models_2023_hourly_trend.png)

The full-year hourly plot is dense by design, so the short-window version below focuses on the last 240 hours of 2023.

![Last 240 hours](../figures/era5_ablation_max10/best_models_2023_last_240h_trend.png)

![7-day trend](../figures/era5_ablation_max10/best_models_2023_rolling_7d_trend.png)

![Rolling residuals](../figures/era5_ablation_max10/best_models_2023_rolling_residuals.png)

![Monthly MAE](../figures/era5_ablation_max10/monthly_mae_best_models.png)

![Error distribution](../figures/era5_ablation_max10/error_distribution_best_models.png)

![Predicted vs actual](../figures/era5_ablation_max10/predicted_vs_actual_hexbin.png)

![With ERA5 month-hour MAE](../figures/era5_ablation_max10/heatmap_month_hour_mae_with_era5_xgboost.png)

![Without ERA5 month-hour MAE](../figures/era5_ablation_max10/heatmap_month_hour_mae_without_era5_catboost.png)

## SHAP

SHAP summary plots use the standard color coding: red means high feature value, blue means low feature value.

![SHAP summary with ERA5 XGBoost](../figures/era5_ablation_max10/shap_summary_with_era5_xgboost.png)

![SHAP bar with ERA5 XGBoost](../figures/era5_ablation_max10/shap_bar_with_era5_xgboost.png)

![SHAP summary without ERA5 CatBoost](../figures/era5_ablation_max10/shap_summary_without_era5_catboost.png)

![SHAP bar without ERA5 CatBoost](../figures/era5_ablation_max10/shap_bar_without_era5_catboost.png)

## Feature Drift

![Feature drift PSI](../figures/era5_ablation_max10/feature_drift_top30_psi.png)

The strongest drift is concentrated in historical price rolling statistics, especially rolling volatility and rolling mean/max features. This is consistent with 2023 having a different price regime from the 2019-2022 train/validation period.

## Initial Takeaways

- The best with-ERA5 model is XGBoost with MAE 19.1792.
- The best without-ERA5 model is CatBoost with MAE 19.1754.
- At the global yearly MAE level, these are effectively tied.
- SHAP for both models is dominated by market/fundamental variables: residual load forecast, price lags, rolling price statistics, renewable share, load forecast, and wind share.
- ERA5 may still matter in model-specific or scenario-specific ways. The next useful check is slicing performance by high wind, high solar, high residual load, negative price, and price spike periods.
