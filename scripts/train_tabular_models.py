#!/usr/bin/env python3
"""Train tuned tabular electricity price models and two stacking ensembles."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from scipy.optimize import nnls
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


CONFIG_PATH = Path("configs/modeling_config.json")
ERA5_FEATURE_PATH = Path("data/processed/era5/era5_de_lu_hourly_region_features_2019_2023.csv")


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.abs(y_true) + np.abs(y_pred)
    mask = denominator > 1e-9
    return float(np.mean(2 * np.abs(y_pred[mask] - y_true[mask]) / denominator[mask]))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray, reference: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(reference)
    if not mask.any():
        return float("nan")
    true_direction = np.sign(y_true[mask] - reference[mask])
    pred_direction = np.sign(y_pred[mask] - reference[mask])
    return float(np.mean(true_direction == pred_direction))


def binary_precision_recall(y_true_flag: np.ndarray, y_pred_flag: np.ndarray) -> tuple[float, float]:
    tp = float(np.sum(y_true_flag & y_pred_flag))
    fp = float(np.sum(~y_true_flag & y_pred_flag))
    fn = float(np.sum(y_true_flag & ~y_pred_flag))
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    return precision, recall


def metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prev_price: np.ndarray | None = None,
    lag24_price: np.ndarray | None = None,
    high_price_threshold: float | None = None,
) -> dict[str, float]:
    result = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "smape": smape(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }
    if prev_price is not None:
        result["direction_accuracy_prev_hour"] = directional_accuracy(y_true, y_pred, prev_price)
    if lag24_price is not None:
        result["direction_accuracy_lag24"] = directional_accuracy(y_true, y_pred, lag24_price)

    true_negative = y_true < 0
    pred_negative = y_pred < 0
    precision, recall = binary_precision_recall(true_negative, pred_negative)
    result["negative_price_precision"] = precision
    result["negative_price_recall"] = recall

    if high_price_threshold is not None:
        high_mask = y_true >= high_price_threshold
        result["top10_price_mae"] = (
            float(mean_absolute_error(y_true[high_mask], y_pred[high_mask]))
            if high_mask.any()
            else float("nan")
        )
    return result


def sample_param(spec: dict[str, Any], rng: random.Random) -> Any:
    kind = spec["type"]
    if kind == "choice":
        return rng.choice(spec["values"])
    if kind == "int":
        return rng.randint(int(spec["low"]), int(spec["high"]))
    if kind == "uniform":
        return rng.uniform(float(spec["low"]), float(spec["high"]))
    if kind == "loguniform":
        low = math.log(float(spec["low"]))
        high = math.log(float(spec["high"]))
        return math.exp(rng.uniform(low, high))
    raise ValueError(f"unsupported search space type: {kind}")


def sample_params(space: dict[str, dict[str, Any]], rng: random.Random) -> dict[str, Any]:
    return {name: sample_param(spec, rng) for name, spec in space.items()}


def load_data(config: dict[str, Any], drop_era5: bool = False) -> tuple[pd.DataFrame, list[str], str]:
    path = Path(config["data"]["model_table"])
    df = pd.read_csv(path, parse_dates=[config["project"]["timestamp_col"]])
    target = config["project"]["target"]

    leakage_exact = {
        "timestamp_utc",
        target,
        "load_actual",
        "solar_actual",
        "wind_onshore_actual",
        "wind_offshore_actual",
        "wind_total_actual",
        "renewable_actual",
        "residual_load_actual"
    }
    leakage_suffixes = ("_actual",)
    feature_cols = [
        col for col in df.columns
        if col not in leakage_exact and not col.endswith(leakage_suffixes)
    ]
    numeric_feature_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(df[col])]
    if drop_era5:
        era5_cols = pd.read_csv(ERA5_FEATURE_PATH, nrows=0).columns.tolist()
        era5_features = set(col for col in era5_cols if col != "timestamp_utc")
        numeric_feature_cols = [col for col in numeric_feature_cols if col not in era5_features]
    return df, numeric_feature_cols, target


def time_mask(df: pd.DataFrame, start: str, end: str) -> pd.Series:
    ts = df["timestamp_utc"]
    return (ts >= pd.Timestamp(start)) & (ts <= pd.Timestamp(end))


def cv_folds(df: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    periods = [
        ("2021-01-01T00:00:00Z", "2021-06-30T23:00:00Z"),
        ("2021-07-01T00:00:00Z", "2021-12-31T23:00:00Z"),
        ("2022-01-01T00:00:00Z", "2022-06-30T23:00:00Z"),
        ("2022-07-01T00:00:00Z", "2022-12-31T23:00:00Z"),
    ]
    folds = []
    ts = df["timestamp_utc"]
    for start, end in periods:
        val_mask = time_mask(df, start, end)
        train_mask = ts < pd.Timestamp(start) - pd.Timedelta(hours=24)
        folds.append((np.flatnonzero(train_mask.to_numpy()), np.flatnonzero(val_mask.to_numpy())))
    return folds


def make_model(name: str, params: dict[str, Any], scale: bool) -> Any:
    if name == "elastic_net":
        model = ElasticNet(random_state=42, **params)
    elif name == "xgboost":
        model = XGBRegressor(
            random_state=42,
            n_jobs=4,
            objective="reg:squarederror",
            tree_method="hist",
            **params,
        )
    elif name == "lightgbm":
        model = LGBMRegressor(random_state=42, n_jobs=4, verbose=-1, objective="regression", **params)
    elif name == "catboost":
        model = CatBoostRegressor(
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
            loss_function="RMSE",
            **params,
        )
    else:
        raise ValueError(f"unsupported model: {name}")

    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def tune_model(
    name: str,
    spec: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    max_trials: int,
    rng: random.Random,
) -> tuple[dict[str, Any], float]:
    best_params: dict[str, Any] | None = None
    best_score = float("inf")
    trials = max(1, max_trials)

    for trial in range(trials):
        params = sample_params(spec.get("search_space", {}), rng)
        fold_scores = []
        for train_idx, val_idx in folds:
            model = make_model(name, params, bool(spec["requires_scaling"]))
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = model.predict(X.iloc[val_idx])
            fold_scores.append(mean_absolute_error(y.iloc[val_idx], pred))
        score = float(np.mean(fold_scores))
        if score < best_score:
            best_score = score
            best_params = params
        print(f"{name} trial {trial + 1}/{trials}: cv_mae={score:.4f}", flush=True)

    assert best_params is not None
    print(f"{name} best_cv_mae={best_score:.4f} params={best_params}", flush=True)
    return best_params, best_score


def fit_predict_model(name: str, spec: dict[str, Any], params: dict[str, Any], X_train, y_train, X_test):
    model = make_model(name, params, bool(spec["requires_scaling"]))
    model.fit(X_train, y_train)
    return model, model.predict(X_test)


def fit_naive(df: pd.DataFrame, test_mask: pd.Series, col: str) -> np.ndarray:
    return df.loc[test_mask, col].to_numpy()


class SequenceRegressor(nn.Module):
    def __init__(self, kind: str, input_size: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        dropout_value = dropout if num_layers > 1 else 0.0
        if kind == "lstm":
            self.seq = nn.LSTM(input_size, hidden_size, num_layers=num_layers, dropout=dropout_value, batch_first=True)
        elif kind == "rnn":
            self.seq = nn.RNN(input_size, hidden_size, num_layers=num_layers, dropout=dropout_value, batch_first=True)
        else:
            raise ValueError(kind)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.seq(x)
        return self.head(out[:, -1, :])


def fit_sequence_predict(
    kind: str,
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    pred_idx: np.ndarray,
    lookback: int,
    epochs: int,
    hidden_size: int,
    batch_size: int,
) -> np.ndarray:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train_tab = imputer.fit_transform(X.iloc[train_idx])
    scaler.fit(x_train_tab)
    x_all = scaler.transform(imputer.transform(X))
    y_all = y.to_numpy(dtype=np.float32)

    train_set = set(train_idx.tolist())
    seq_x, seq_y = [], []
    for idx in train_idx:
        if idx < lookback:
            continue
        window = np.arange(idx - lookback, idx)
        if all(i in train_set for i in window):
            seq_x.append(x_all[window])
            seq_y.append(y_all[idx])

    fallback = float(np.nanmean(y_all[train_idx]))
    if not seq_x:
        return np.full(len(pred_idx), fallback)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = SequenceRegressor(kind, input_size=x_all.shape[1], hidden_size=hidden_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.002)
    loss_fn = nn.MSELoss()
    tx = torch.tensor(np.asarray(seq_x), dtype=torch.float32, device=device)
    ty = torch.tensor(np.asarray(seq_y)[:, None], dtype=torch.float32, device=device)
    generator = torch.Generator(device="cpu").manual_seed(42)

    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(tx), generator=generator).tolist()
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            opt.zero_grad()
            pred = model(tx[batch])
            loss = loss_fn(pred, ty[batch])
            loss.backward()
            opt.step()

    pred_windows = []
    valid_positions = []
    for pos, idx in enumerate(pred_idx):
        if idx >= lookback:
            pred_windows.append(x_all[idx - lookback:idx])
            valid_positions.append(pos)

    out = np.full(len(pred_idx), fallback, dtype=np.float32)
    if pred_windows:
        model.eval()
        with torch.no_grad():
            px = torch.tensor(np.asarray(pred_windows), dtype=torch.float32, device=device)
            pp = model(px).detach().cpu().numpy().ravel()
        out[np.asarray(valid_positions)] = pp
    return out


def select_sequence_features(
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    method: str,
    top_k: int,
) -> tuple[list[str], dict[str, float]]:
    if method == "none":
        return list(X.columns), {}

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(X.iloc[train_idx]))
    y_train = y.iloc[train_idx].to_numpy()

    if method == "lasso":
        selector = Lasso(alpha=0.001, max_iter=20000, random_state=42)
    elif method == "elastic_net":
        selector = ElasticNet(alpha=0.001, l1_ratio=0.9, max_iter=20000, random_state=42)
    else:
        raise ValueError(f"unsupported sequence feature selection method: {method}")

    selector.fit(x_train, y_train)
    coef = np.abs(selector.coef_)
    finite_coef = np.where(np.isfinite(coef), coef, 0.0)
    order = np.argsort(finite_coef)[::-1]
    positive = [idx for idx in order if finite_coef[idx] > 0]
    selected_idx = positive[:top_k] if positive else order[:top_k]
    selected_cols = [X.columns[idx] for idx in selected_idx]
    scores = {X.columns[idx]: float(finite_coef[idx]) for idx in selected_idx}
    return selected_cols, scores


def nnls_stack(train_pred: np.ndarray, y: np.ndarray, test_pred: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    # Fit intercept separately by centering, then solve non-negative least squares.
    y_mean = y.mean()
    x_mean = train_pred.mean(axis=0)
    weights, _ = nnls(train_pred - x_mean, y - y_mean)
    intercept = float(y_mean - x_mean @ weights)
    pred = test_pred @ weights + intercept
    return pred, {"weights": weights.tolist(), "intercept": intercept, "sum_weights": float(weights.sum())}


def gbm_stack(train_pred: np.ndarray, y: np.ndarray, test_pred: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    model = LGBMRegressor(
        random_state=42,
        n_jobs=4,
        verbose=-1,
        objective="regression",
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=7,
        min_child_samples=30,
        reg_lambda=1.0,
    )
    model.fit(train_pred, y)
    return model.predict(test_pred), {"meta_model": "lightgbm", "params": model.get_params()}


def generate_oof_and_test_predictions(
    name: str,
    spec: dict[str, Any],
    params: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    train_val_idx: np.ndarray,
    test_idx: np.ndarray,
    sequence_args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    oof = np.full(len(X), np.nan, dtype=np.float64)
    for fold_no, (train_idx, val_idx) in enumerate(folds, start=1):
        if name in {"lstm", "rnn"}:
            pred = fit_sequence_predict(
                name,
                X,
                y,
                train_idx,
                val_idx,
                sequence_args.lookback,
                sequence_args.sequence_epochs,
                sequence_args.hidden_size,
                sequence_args.batch_size,
            )
        else:
            _, pred = fit_predict_model(name, spec, params, X.iloc[train_idx], y.iloc[train_idx], X.iloc[val_idx])
        oof[val_idx] = pred
        print(f"{name} OOF fold {fold_no}: mae={mean_absolute_error(y.iloc[val_idx], pred):.4f}", flush=True)

    if name in {"lstm", "rnn"}:
        test_pred = fit_sequence_predict(
            name,
            X,
            y,
            train_val_idx,
            test_idx,
            sequence_args.lookback,
            sequence_args.sequence_epochs,
            sequence_args.hidden_size,
            sequence_args.batch_size,
        )
    else:
        _, test_pred = fit_predict_model(name, spec, params, X.iloc[train_val_idx], y.iloc[train_val_idx], X.iloc[test_idx])
    return oof, test_pred


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-trials", type=int, default=10, help="Random-search trials per model for this run.")
    parser.add_argument("--include-sequence", action="store_true", help="Include LSTM and RNN in the formal run.")
    parser.add_argument("--sequence-epochs", type=int, default=5)
    parser.add_argument("--lookback", type=int, default=24)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sequence-feature-selection", choices=["none", "lasso", "elastic_net"], default="none")
    parser.add_argument("--sequence-top-k", type=int, default=50)
    parser.add_argument("--stacking-max-base-mae-ratio", type=float, default=1.25)
    parser.add_argument("--drop-era5", action="store_true", help="Remove ERA5 features while keeping SMARD, PVGIS, lags, and calendar.")
    parser.add_argument("--experiment-name", default="tabular_model", help="Prefix for result and prediction files.")
    args = parser.parse_args()

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    df, features, target = load_data(config, drop_era5=args.drop_era5)
    folds = cv_folds(df)
    rng = random.Random(config["hyperparameter_tuning"]["random_seed"])

    train_val_mask = time_mask(df, "2019-01-01T00:00:00Z", "2022-12-31T23:00:00Z")
    test_mask = time_mask(df, "2023-01-01T00:00:00Z", "2023-12-31T23:00:00Z")

    # Drop rows without lag features from the train/validation period.
    required = ["price_lag_24h", "price_lag_168h", target]
    train_val_mask = train_val_mask & df[required].notna().all(axis=1)
    test_mask = test_mask & df[required].notna().all(axis=1)

    X = df[features]
    X = X.replace([np.inf, -np.inf], np.nan)
    y = df[target]
    train_val_idx = np.flatnonzero(train_val_mask.to_numpy())
    test_idx = np.flatnonzero(test_mask.to_numpy())
    X_train_val = X.loc[train_val_mask]
    y_train_val = y.loc[train_val_mask]
    X_test = X.loc[test_mask]
    y_test = y.loc[test_mask].to_numpy()
    test_prev_price = y.shift(1).loc[test_mask].to_numpy()
    test_lag24_price = df.loc[test_mask, "price_lag_24h"].to_numpy()
    high_price_threshold = float(y.loc[train_val_mask].quantile(0.9))

    def test_metrics(pred: np.ndarray) -> dict[str, float]:
        return metrics(
            y_test,
            pred,
            prev_price=test_prev_price,
            lag24_price=test_lag24_price,
            high_price_threshold=high_price_threshold,
        )

    output_dir = Path(config["outputs"]["reports_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = Path(config["outputs"]["prediction_dir"])
    pred_dir.mkdir(parents=True, exist_ok=True)

    results = []
    test_predictions: dict[str, np.ndarray] = {}
    run_metadata: dict[str, Any] = {
        "experiment_name": args.experiment_name,
        "drop_era5": args.drop_era5,
        "feature_count": len(features),
        "sequence_feature_selection": {
            "method": args.sequence_feature_selection,
            "top_k": args.sequence_top_k,
            "selected_features": [],
            "scores": {},
        }
    }

    for name, lag_col in [("naive_lag_24h", "price_lag_24h"), ("seasonal_naive_lag_168h", "price_lag_168h")]:
        pred = fit_naive(df, test_mask, lag_col)
        test_predictions[name] = pred
        result = {"model": name, "cv_mae": None, **test_metrics(pred), "best_params": {}}
        results.append(result)
        print(f"{name} test_mae={result['mae']:.4f}", flush=True)

    tuned_specs = {
        name: spec for name, spec in config["models"].items()
        if name in {"elastic_net", "xgboost", "lightgbm", "catboost"}
    }
    best_params_by_model = {}
    for name, spec in tuned_specs.items():
        params, cv_score = tune_model(name, spec, X, y, folds, args.max_trials, rng)
        best_params_by_model[name] = params
        _, pred = fit_predict_model(name, spec, params, X_train_val, y_train_val, X_test)
        test_predictions[name] = pred
        result = {"model": name, "cv_mae": cv_score, **test_metrics(pred), "best_params": params}
        results.append(result)
        print(f"{name} test_mae={result['mae']:.4f}", flush=True)

    if args.include_sequence:
        sequence_features, sequence_scores = select_sequence_features(
            X,
            y,
            train_val_idx,
            args.sequence_feature_selection,
            args.sequence_top_k,
        )
        run_metadata["sequence_feature_selection"]["selected_features"] = sequence_features
        run_metadata["sequence_feature_selection"]["scores"] = sequence_scores
        X_sequence = X[sequence_features]
        print(
            f"sequence features: method={args.sequence_feature_selection}, "
            f"count={len(sequence_features)}",
            flush=True,
        )
        for name in ["lstm", "rnn"]:
            spec = config["models"][name]
            params = {
                "lookback_hours": args.lookback,
                "hidden_size": args.hidden_size,
                "batch_size": args.batch_size,
                "epochs": args.sequence_epochs,
                "feature_selection": args.sequence_feature_selection,
                "feature_count": len(sequence_features),
            }
            _, pred = generate_oof_and_test_predictions(
                name,
                spec,
                params,
                X_sequence,
                y,
                [],
                train_val_idx,
                test_idx,
                args,
            )
            test_predictions[name] = pred
            result = {"model": name, "cv_mae": None, **test_metrics(pred), "best_params": params}
            results.append(result)
            best_params_by_model[name] = params
            tuned_specs[name] = spec
            print(f"{name} test_mae={result['mae']:.4f}", flush=True)
    else:
        X_sequence = X

    # Formal time-series OOF stacking.
    oof_by_model = {}
    test_pred_by_base = {}
    base_names = list(tuned_specs.keys())
    for name in base_names:
        spec = tuned_specs[name]
        params = best_params_by_model[name]
        oof, test_pred = generate_oof_and_test_predictions(
            name,
            spec,
            params,
            X_sequence if name in {"lstm", "rnn"} else X,
            y,
            folds,
            train_val_idx,
            test_idx,
            args,
        )
        oof_by_model[name] = oof
        test_pred_by_base[name] = test_pred

    stack_mask = train_val_mask.to_numpy().copy()
    for name in base_names:
        stack_mask &= np.isfinite(oof_by_model[name])

    stack_base_mae = {
        name: float(mean_absolute_error(y.to_numpy()[stack_mask], oof_by_model[name][stack_mask]))
        for name in base_names
    }
    best_stack_base = min(stack_base_mae.values())
    selected_base_names = [
        name for name in base_names
        if stack_base_mae[name] <= best_stack_base * args.stacking_max_base_mae_ratio
    ]
    if len(selected_base_names) < 2:
        selected_base_names = sorted(stack_base_mae, key=stack_base_mae.get)[:2]
    print(f"stacking base OOF MAE: {stack_base_mae}", flush=True)
    print(f"stacking selected bases: {selected_base_names}", flush=True)

    meta_train_array = np.column_stack([oof_by_model[name][stack_mask] for name in selected_base_names])
    meta_test_array = np.column_stack([test_pred_by_base[name] for name in selected_base_names])
    y_meta = y.to_numpy()[stack_mask]

    for stack_name, stack_fn in [
        ("stacker_non_negative_weights_oof", nnls_stack),
        ("stacker_gbm_oof", gbm_stack),
    ]:
        pred, stack_info = stack_fn(meta_train_array, y_meta, meta_test_array)
        stack_info["base_models"] = selected_base_names
        test_predictions[stack_name] = pred
        result = {"model": stack_name, "cv_mae": None, **test_metrics(pred), "best_params": stack_info}
        results.append(result)
        print(f"{stack_name} test_mae={result['mae']:.4f}", flush=True)

    results_df = pd.DataFrame(results).sort_values("mae")
    results_path = output_dir / f"{args.experiment_name}_results.csv"
    results_df.to_csv(results_path, index=False)
    metadata_path = output_dir / f"{args.experiment_name}_run_metadata.json"
    metadata_path.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")

    pred_df = pd.DataFrame({"timestamp_utc": df.loc[test_mask, "timestamp_utc"].astype(str), "y_true": y_test})
    for name, pred in test_predictions.items():
        pred_df[f"pred_{name}"] = pred
    pred_path = pred_dir / f"{args.experiment_name}_test_predictions_2023.csv"
    pred_df.to_csv(pred_path, index=False)

    print(f"wrote results -> {results_path}")
    print(f"wrote metadata -> {metadata_path}")
    print(f"wrote predictions -> {pred_path}")
    display_cols = [
        "model",
        "mae",
        "rmse",
        "smape",
        "r2",
        "direction_accuracy_prev_hour",
        "direction_accuracy_lag24",
        "negative_price_precision",
        "negative_price_recall",
        "top10_price_mae",
    ]
    print(results_df[[col for col in display_cols if col in results_df.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
