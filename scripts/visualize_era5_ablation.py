#!/usr/bin/env python3
"""Visualize ERA5 ablation results and SHAP explanations."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostRegressor
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_tabular_models import load_data, time_mask  # noqa: E402


CONFIG = {
    "project": {"target": "day_ahead_price", "timestamp_col": "timestamp_utc"},
    "data": {"model_table": "data/processed/de_lu_hourly_2019_2023.csv"},
}
FIG_DIR = ROOT / "reports/figures/era5_ablation_max10"
REPORT_DIR = ROOT / "reports/modeling"
PRED_DIR = ROOT / "data/predictions"


def read_best_params(path: Path, model: str) -> dict:
    df = pd.read_csv(path)
    value = df.loc[df["model"] == model, "best_params"].iloc[0]
    return ast.literal_eval(value)


def train_tree_model(model_name: str, params: dict, drop_era5: bool) -> tuple[object, pd.DataFrame, pd.DataFrame, pd.Series]:
    df, features, target = load_data(CONFIG, drop_era5=drop_era5)
    train_val_mask = time_mask(df, "2019-01-01T00:00:00Z", "2022-12-31T23:00:00Z")
    test_mask = time_mask(df, "2023-01-01T00:00:00Z", "2023-12-31T23:00:00Z")
    required = ["price_lag_24h", "price_lag_168h", target]
    train_val_mask = train_val_mask & df[required].notna().all(axis=1)
    test_mask = test_mask & df[required].notna().all(axis=1)

    X = df[features].replace([np.inf, -np.inf], np.nan)
    y = df[target]
    X_train = X.loc[train_val_mask]
    X_test = X.loc[test_mask]
    y_train = y.loc[train_val_mask]

    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=features, index=X_train.index)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=features, index=X_test.index)

    if model_name == "xgboost":
        model = XGBRegressor(
            random_state=42,
            n_jobs=4,
            objective="reg:squarederror",
            tree_method="hist",
            **params,
        )
    elif model_name == "catboost":
        model = CatBoostRegressor(
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
            loss_function="RMSE",
            **params,
        )
    else:
        raise ValueError(model_name)

    model.fit(X_train_imp, y_train)
    return model, X_train_imp, X_test_imp, y.loc[test_mask]


def save_line_plots() -> pd.DataFrame:
    with_pred = pd.read_csv(PRED_DIR / "era5_ablation_with_era5_max10_test_predictions_2023.csv", parse_dates=["timestamp_utc"])
    without_pred = pd.read_csv(PRED_DIR / "era5_ablation_without_era5_max10_test_predictions_2023.csv", parse_dates=["timestamp_utc"])

    pred = with_pred[["timestamp_utc", "y_true", "pred_xgboost"]].rename(columns={"pred_xgboost": "with_era5_xgboost"})
    pred["without_era5_catboost"] = without_pred["pred_catboost"].to_numpy()
    pred["err_with_era5_xgboost"] = pred["with_era5_xgboost"] - pred["y_true"]
    pred["err_without_era5_catboost"] = pred["without_era5_catboost"] - pred["y_true"]
    pred.to_csv(REPORT_DIR / "era5_ablation_max10_best_model_predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(18, 7))
    ax.plot(pred["timestamp_utc"], pred["y_true"], color="black", linewidth=0.7, alpha=0.65, label="Actual")
    ax.plot(pred["timestamp_utc"], pred["with_era5_xgboost"], color="#d62728", linewidth=0.55, alpha=0.75, label="With ERA5: XGBoost")
    ax.plot(pred["timestamp_utc"], pred["without_era5_catboost"], color="#1f77b4", linewidth=0.55, alpha=0.75, label="Without ERA5: CatBoost")
    ax.set_title("2023 hourly day-ahead price: actual vs best with/without ERA5 models")
    ax.set_ylabel("EUR/MWh")
    ax.legend(ncol=3, frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "best_models_2023_hourly_trend.png", dpi=180)
    plt.close(fig)

    short = pred.tail(240)
    fig, ax = plt.subplots(figsize=(18, 7))
    ax.plot(short["timestamp_utc"], short["y_true"], color="black", linewidth=1.6, label="Actual")
    ax.plot(short["timestamp_utc"], short["with_era5_xgboost"], color="#d62728", linewidth=1.2, label="With ERA5: XGBoost")
    ax.plot(short["timestamp_utc"], short["without_era5_catboost"], color="#1f77b4", linewidth=1.2, label="Without ERA5: CatBoost")
    ax.set_title("Last 240 hours in 2023: actual vs best with/without ERA5 models")
    ax.set_ylabel("EUR/MWh")
    ax.legend(ncol=3, frameon=False)
    ax.grid(alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "best_models_2023_last_240h_trend.png", dpi=180)
    plt.close(fig)

    smooth = pred.set_index("timestamp_utc")[["y_true", "with_era5_xgboost", "without_era5_catboost"]].rolling(168, min_periods=24).mean()
    fig, ax = plt.subplots(figsize=(18, 7))
    ax.plot(smooth.index, smooth["y_true"], color="black", linewidth=1.6, label="Actual, 7-day rolling")
    ax.plot(smooth.index, smooth["with_era5_xgboost"], color="#d62728", linewidth=1.3, label="With ERA5: XGBoost")
    ax.plot(smooth.index, smooth["without_era5_catboost"], color="#1f77b4", linewidth=1.3, label="Without ERA5: CatBoost")
    ax.set_title("2023 price trend, 7-day rolling mean")
    ax.set_ylabel("EUR/MWh")
    ax.legend(ncol=3, frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "best_models_2023_rolling_7d_trend.png", dpi=180)
    plt.close(fig)

    residual_smooth = pred.set_index("timestamp_utc")[["err_with_era5_xgboost", "err_without_era5_catboost"]].rolling(168, min_periods=24).mean()
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.plot(residual_smooth.index, residual_smooth["err_with_era5_xgboost"], color="#d62728", label="With ERA5: XGBoost")
    ax.plot(residual_smooth.index, residual_smooth["err_without_era5_catboost"], color="#1f77b4", label="Without ERA5: CatBoost")
    ax.set_title("Prediction residuals, 7-day rolling mean")
    ax.set_ylabel("Prediction - actual")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "best_models_2023_rolling_residuals.png", dpi=180)
    plt.close(fig)
    return pred


def save_error_plots(pred: pd.DataFrame) -> None:
    pred["month"] = pred["timestamp_utc"].dt.month
    pred["hour"] = pred["timestamp_utc"].dt.hour
    pred["abs_err_with"] = pred["err_with_era5_xgboost"].abs()
    pred["abs_err_without"] = pred["err_without_era5_catboost"].abs()

    monthly = pred.groupby("month")[["abs_err_with", "abs_err_without"]].mean()
    monthly.to_csv(REPORT_DIR / "era5_ablation_max10_monthly_mae.csv")
    x = np.arange(len(monthly.index))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - 0.18, monthly["abs_err_with"], width=0.36, color="#d62728", label="With ERA5: XGBoost")
    ax.bar(x + 0.18, monthly["abs_err_without"], width=0.36, color="#1f77b4", label="Without ERA5: CatBoost")
    ax.set_xticks(x, monthly.index)
    ax.set_xlabel("Month")
    ax.set_ylabel("MAE")
    ax.set_title("Monthly MAE in 2023")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "monthly_mae_best_models.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(-120, 120, 90)
    ax.hist(pred["err_with_era5_xgboost"], bins=bins, alpha=0.55, color="#d62728", label="With ERA5: XGBoost")
    ax.hist(pred["err_without_era5_catboost"], bins=bins, alpha=0.55, color="#1f77b4", label="Without ERA5: CatBoost")
    ax.set_title("Prediction error distribution")
    ax.set_xlabel("Prediction - actual")
    ax.set_ylabel("Hours")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "error_distribution_best_models.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    for ax, col, title in [
        (axes[0], "with_era5_xgboost", "With ERA5: XGBoost"),
        (axes[1], "without_era5_catboost", "Without ERA5: CatBoost"),
    ]:
        hb = ax.hexbin(pred["y_true"], pred[col], gridsize=55, cmap="viridis", mincnt=1)
        lo = min(pred["y_true"].min(), pred[col].min())
        hi = max(pred["y_true"].max(), pred[col].max())
        ax.plot([lo, hi], [lo, hi], color="white", linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel("Actual")
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("Predicted")
    fig.colorbar(hb, ax=axes, label="Hour count")
    fig.suptitle("Predicted vs actual price density")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "predicted_vs_actual_hexbin.png", dpi=180)
    plt.close(fig)

    for col, title, filename in [
        ("abs_err_with", "With ERA5: XGBoost hourly MAE by month and hour", "heatmap_month_hour_mae_with_era5_xgboost.png"),
        ("abs_err_without", "Without ERA5: CatBoost hourly MAE by month and hour", "heatmap_month_hour_mae_without_era5_catboost.png"),
    ]:
        matrix = pred.pivot_table(index="month", columns="hour", values=col, aggfunc="mean")
        fig, ax = plt.subplots(figsize=(14, 5))
        im = ax.imshow(matrix, aspect="auto", cmap="magma")
        ax.set_title(title)
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Month")
        ax.set_xticks(np.arange(24))
        ax.set_yticks(np.arange(len(matrix.index)), matrix.index)
        fig.colorbar(im, ax=ax, label="MAE")
        fig.tight_layout()
        fig.savefig(FIG_DIR / filename, dpi=180)
        plt.close(fig)


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return np.nan
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    expected_pct = np.histogram(expected, bins=edges)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=edges)[0] / len(actual)
    expected_pct = np.clip(expected_pct, 1e-6, None)
    actual_pct = np.clip(actual_pct, 1e-6, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def save_drift_plot() -> None:
    drift_rows = []
    for feature_set, drop_era5 in [("with_era5", False), ("without_era5", True)]:
        df, features, target = load_data(CONFIG, drop_era5=drop_era5)
        train_val_mask = time_mask(df, "2019-01-01T00:00:00Z", "2022-12-31T23:00:00Z")
        test_mask = time_mask(df, "2023-01-01T00:00:00Z", "2023-12-31T23:00:00Z")
        required = ["price_lag_24h", "price_lag_168h", target]
        train_val_mask = train_val_mask & df[required].notna().all(axis=1)
        test_mask = test_mask & df[required].notna().all(axis=1)
        X = df[features].replace([np.inf, -np.inf], np.nan)
        for feature in features:
            drift_rows.append({
                "feature_set": feature_set,
                "feature": feature,
                "psi_2019_2022_vs_2023": psi(X.loc[train_val_mask, feature].to_numpy(), X.loc[test_mask, feature].to_numpy()),
            })
    drift = pd.DataFrame(drift_rows).sort_values("psi_2019_2022_vs_2023", ascending=False)
    drift.to_csv(REPORT_DIR / "era5_ablation_max10_feature_drift_psi.csv", index=False)

    top = drift.head(30).iloc[::-1]
    colors = np.where(top["feature_set"].eq("with_era5"), "#d62728", "#1f77b4")
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.barh(top["feature_set"] + ": " + top["feature"], top["psi_2019_2022_vs_2023"], color=colors)
    ax.set_title("Top feature drift by PSI: train/validation 2019-2022 vs test 2023")
    ax.set_xlabel("Population Stability Index")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_drift_top30_psi.png", dpi=180)
    plt.close(fig)


def save_shap_plots(model_name: str, feature_set: str, model, X_test: pd.DataFrame, sample_size: int = 3000) -> None:
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(X_test), size=min(sample_size, len(X_test)), replace=False)
    X_sample = X_test.iloc[np.sort(sample_idx)]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = (
        pd.DataFrame({"feature": X_sample.columns, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
    )
    importance.to_csv(REPORT_DIR / f"era5_ablation_max10_shap_importance_{feature_set}_{model_name}.csv", index=False)

    plt.figure(figsize=(11, 8))
    shap.summary_plot(shap_values, X_sample, max_display=25, show=False)
    plt.title(f"SHAP summary: {feature_set}, {model_name}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"shap_summary_{feature_set}_{model_name}.png", dpi=180, bbox_inches="tight")
    plt.close()

    top = importance.head(25).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#4c78a8")
    ax.set_title(f"Mean absolute SHAP importance: {feature_set}, {model_name}")
    ax.set_xlabel("mean(|SHAP value|)")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"shap_bar_{feature_set}_{model_name}.png", dpi=180)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pred = save_line_plots()
    save_error_plots(pred)
    save_drift_plot()

    with_params = read_best_params(REPORT_DIR / "era5_ablation_with_era5_max10_results.csv", "xgboost")
    without_params = read_best_params(REPORT_DIR / "era5_ablation_without_era5_max10_results.csv", "catboost")

    with_model, _, with_X_test, _ = train_tree_model("xgboost", with_params, drop_era5=False)
    save_shap_plots("xgboost", "with_era5", with_model, with_X_test)

    without_model, _, without_X_test, _ = train_tree_model("catboost", without_params, drop_era5=True)
    save_shap_plots("catboost", "without_era5", without_model, without_X_test)

    print(f"wrote figures -> {FIG_DIR}")
    print(f"wrote comparison predictions -> {REPORT_DIR / 'era5_ablation_max10_best_model_predictions.csv'}")
    print(f"wrote drift table -> {REPORT_DIR / 'era5_ablation_max10_feature_drift_psi.csv'}")


if __name__ == "__main__":
    main()
