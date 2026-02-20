# Multi-Store Retail Sales Forecasting & Inventory Optimization

Forecast daily product sales across multiple store locations to reduce stockouts, minimize overstock, and support inventory allocation decisions.

**Dataset:** [Kaggle Store Sales - Time Series Forecasting]

## Business Goal

- Reduce stockouts
- Minimize overstock
- Improve revenue planning
- Support inventory allocation and reorder point decisions

## Modeling Steps

1. **Phase 1 - EDA:** Trend decomposition, seasonality (weekly, monthly), holiday effects, ADF stationarity test, ACF/PACF
2. **Phase 2 - Baselines:** Naive, Moving Average, Linear Regression with lag features
3. **Phase 3 - Statistical:** ARIMA/SARIMA, SARIMAX (promotions as exogenous), AIC/BIC grid search, residual diagnostics
4. **Phase 4 - ML:** XGBoost, LightGBM, Random Forest with time-series split (no random split to avoid temporal leakage)
5. **Phase 5 - Business:** Reorder points, safety stock, inventory simulation, stockout reduction analysis

## Evaluation Metrics

- **MAE** : Mean Absolute Error
- **RMSE** : Root Mean Squared Error  
- **MAPE** : Mean Absolute Percentage Error (preferred for business; expresses error as % of demand)

## Results & Tradeoffs

- **SARIMA** vs **ML:** SARIMA is interpretable and works well for univariate series; tree-based ML models handle exogenous features (promotions, holidays) and scale to many series
- **MAPE** vs **MAE:** MAPE is easier to interpret for stakeholders (% error); MAE is robust to scale
- Best-performing model typically: LightGBM or XGBoost with lag and rolling features
- Simulated inventory policy using ML forecasts reduces projected stockouts vs naive-based policy

## License

MIT
