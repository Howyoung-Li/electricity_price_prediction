#!/usr/bin/env python3
"""Smoke test LSTM/RNN base models and time-series OOF stacking.

This script is intentionally small and fast. It checks that the project can:

1. Train tabular GBM/linear base models.
2. Train PyTorch LSTM and RNN base models.
3. Generate chronological out-of-fold predictions.
4. Fit two stackers:
   - LightGBM meta learner.
   - Non-negative linear weights without sum-to-one constraint.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from scipy.optimize import nnls
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


CONFIG_PATH = Path("configs/modeling_config.json")
OUT_DIR = Path("reports/modeling")
PRED_DIR = Path("data/predictions")


def metric_summary(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(math.sqrt(mean_squared_error(y, pred))),
    }


def load_features(sample_rows: int) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    df = pd.read_csv(cfg["data"]["model_table"], parse_dates=["timestamp_utc"])
    target = cfg["project"]["target"]
    leakage = {
        "timestamp_utc",
        target,
        "load_actual",
        "solar_actual",
        "wind_onshore_actual",
        "wind_offshore_actual",
        "wind_total_actual",
        "renewable_actual",
        "residual_load_actual",
    }
    features = [
        col for col in df.columns
        if col not in leakage and not col.endswith("_actual") and pd.api.types.is_numeric_dtype(df[col])
    ]
    required = ["price_lag_24h", "price_lag_168h", target]
    df = df[df[required].notna().all(axis=1)].reset_index(drop=True)
    df = df.iloc[:sample_rows].copy()
    return df[features], df[target], df["timestamp_utc"], features


def make_tabular_model(name: str):
    if name == "elastic_net":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=0.001, l1_ratio=0.2, max_iter=10000, random_state=42)),
        ])
    if name == "xgboost":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBRegressor(
                n_estimators=120,
                learning_rate=0.05,
                max_depth=3,
                min_child_weight=5.0,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                tree_method="hist",
                n_jobs=4,
                random_state=42,
            )),
        ])
    if name == "lightgbm":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LGBMRegressor(
                n_estimators=160,
                learning_rate=0.04,
                num_leaves=31,
                min_child_samples=30,
                objective="regression",
                n_jobs=4,
                random_state=42,
                verbose=-1,
            )),
        ])
    if name == "catboost":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", CatBoostRegressor(
                iterations=160,
                learning_rate=0.04,
                depth=4,
                loss_function="RMSE",
                verbose=False,
                allow_writing_files=False,
                random_seed=42,
            )),
        ])
    raise ValueError(name)


class SequenceRegressor(nn.Module):
    def __init__(self, kind: str, input_size: int, hidden_size: int = 24):
        super().__init__()
        if kind == "lstm":
            self.seq = nn.LSTM(input_size, hidden_size, batch_first=True)
        elif kind == "rnn":
            self.seq = nn.RNN(input_size, hidden_size, batch_first=True)
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

    if not seq_x:
        return np.full(len(pred_idx), float(np.nanmean(y_all[train_idx])))

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = SequenceRegressor(kind, input_size=x_all.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.002)
    loss_fn = nn.MSELoss()

    tx = torch.tensor(np.asarray(seq_x), dtype=torch.float32, device=device)
    ty = torch.tensor(np.asarray(seq_y)[:, None], dtype=torch.float32, device=device)
    batch_size = min(128, len(tx))
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
    fallback = float(np.nanmean(y_all[train_idx]))
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


def chronological_folds(n: int, n_folds: int, test_size: int, min_train: int) -> list[tuple[np.ndarray, np.ndarray]]:
    folds = []
    for fold in range(n_folds):
        val_start = min_train + fold * test_size
        val_end = min(val_start + test_size, n)
        if val_end <= val_start:
            break
        train_end = max(0, val_start - 24)
        folds.append((np.arange(train_end), np.arange(val_start, val_end)))
    return folds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-rows", type=int, default=2500)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--fold-size", type=int, default=360)
    parser.add_argument("--lookback", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    X, y, timestamps, features = load_features(args.sample_rows)
    X = X.replace([np.inf, -np.inf], np.nan)
    n = len(X)
    final_test_size = min(360, n // 5)
    train_stack_end = n - final_test_size
    test_idx = np.arange(train_stack_end, n)
    folds = chronological_folds(train_stack_end, args.folds, args.fold_size, min_train=max(720, args.lookback + 200))

    base_models = ["elastic_net", "xgboost", "lightgbm", "catboost", "lstm", "rnn"]
    oof = {name: np.full(train_stack_end, np.nan, dtype=np.float64) for name in base_models}
    test_preds = {}
    results = []

    for name in base_models:
        print(f"base model: {name}", flush=True)
        for fold_no, (train_idx, val_idx) in enumerate(folds, start=1):
            if name in {"lstm", "rnn"}:
                pred = fit_sequence_predict(name, X, y, train_idx, val_idx, args.lookback, args.epochs)
            else:
                model = make_tabular_model(name)
                model.fit(X.iloc[train_idx], y.iloc[train_idx])
                pred = model.predict(X.iloc[val_idx])
            oof[name][val_idx] = pred
            print(f"  fold {fold_no}: mae={mean_absolute_error(y.iloc[val_idx], pred):.4f}", flush=True)

        full_train_idx = np.arange(train_stack_end)
        if name in {"lstm", "rnn"}:
            test_pred = fit_sequence_predict(name, X, y, full_train_idx, test_idx, args.lookback, args.epochs)
        else:
            model = make_tabular_model(name)
            model.fit(X.iloc[full_train_idx], y.iloc[full_train_idx])
            test_pred = model.predict(X.iloc[test_idx])
        test_preds[name] = test_pred
        result = {"model": name, **metric_summary(y.iloc[test_idx].to_numpy(), test_pred)}
        results.append(result)
        print(f"  test: mae={result['mae']:.4f}", flush=True)

    stack_train_mask = np.ones(train_stack_end, dtype=bool)
    for name in base_models:
        stack_train_mask &= np.isfinite(oof[name])
    stack_train = np.column_stack([oof[name][stack_train_mask] for name in base_models])
    y_stack = y.iloc[:train_stack_end].to_numpy()[stack_train_mask]
    stack_test = np.column_stack([test_preds[name] for name in base_models])
    y_test = y.iloc[test_idx].to_numpy()

    weights, _ = nnls(stack_train - stack_train.mean(axis=0), y_stack - y_stack.mean())
    intercept = float(y_stack.mean() - stack_train.mean(axis=0) @ weights)
    nnls_pred = stack_test @ weights + intercept
    results.append({
        "model": "stacker_non_negative_weights_oof",
        **metric_summary(y_test, nnls_pred),
        "weights": json.dumps({name: float(weight) for name, weight in zip(base_models, weights)}),
        "intercept": intercept,
        "sum_weights": float(weights.sum()),
    })

    gbm = LGBMRegressor(
        n_estimators=120,
        learning_rate=0.03,
        num_leaves=7,
        min_child_samples=20,
        reg_lambda=1.0,
        objective="regression",
        random_state=42,
        verbose=-1,
    )
    gbm.fit(stack_train, y_stack)
    gbm_pred = gbm.predict(stack_test)
    results.append({"model": "stacker_gbm_oof", **metric_summary(y_test, gbm_pred)})

    result_df = pd.DataFrame(results).sort_values("mae")
    result_path = OUT_DIR / "oof_stacking_smoke_results.csv"
    result_df.to_csv(result_path, index=False)

    pred_df = pd.DataFrame({"timestamp_utc": timestamps.iloc[test_idx].astype(str), "y_true": y_test})
    for name, pred in test_preds.items():
        pred_df[f"pred_{name}"] = pred
    pred_df["pred_stacker_non_negative_weights_oof"] = nnls_pred
    pred_df["pred_stacker_gbm_oof"] = gbm_pred
    pred_path = PRED_DIR / "oof_stacking_smoke_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    print(f"wrote results -> {result_path}")
    print(f"wrote predictions -> {pred_path}")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
