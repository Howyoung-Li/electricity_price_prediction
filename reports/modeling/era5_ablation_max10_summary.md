# ERA5 Ablation Max10 Summary

- Best with ERA5 MAE: 19.1792
- Best without ERA5 MAE: 19.1754
- Delta MAE, with minus without: 0.0038

Negative delta means ERA5 improved the best model. Here the delta is slightly positive, so the best without-ERA5 model is ahead by 0.0038 MAE, which is effectively a tie.

| feature_set | model | mae | rmse | smape | r2 | direction_accuracy_prev_hour | direction_accuracy_lag24 | negative_price_precision | negative_price_recall | top10_price_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| with_era5 | xgboost | 19.1792 | 26.4897 | 0.3171 | 0.6900 | 0.6305 | 0.7450 | 0.5958 | 0.7567 | 97.2037 |
| without_era5 | catboost | 19.1754 | 26.1358 | 0.3254 | 0.6983 | 0.6305 | 0.7553 | 0.5776 | 0.6700 | 98.8439 |
