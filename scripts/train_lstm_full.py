#!/usr/bin/env python3
"""Train a standalone LSTM on the full electricity price dataset."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


CONFIG_PATH = Path("configs/modeling_config.json")
OUT_DIR = Path("reports/modeling")
PRED_DIR = Path("data/predictions")


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 1e-9
    return float(np.mean(2 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]))


def metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, pred))),
        "smape": smape(y_true, pred),
        "r2": float(r2_score(y_true, pred)),
    }


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
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
    X = df[features].replace([np.inf, -np.inf], np.nan)
    y = df[target]
    return X, y, df["timestamp_utc"]


def select_features(
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
        selector = Lasso(alpha=0.001, max_iter=30000, random_state=42)
    elif method == "elastic_net":
        selector = ElasticNet(alpha=0.001, l1_ratio=0.9, max_iter=30000, random_state=42)
    else:
        raise ValueError(method)
    selector.fit(x_train, y_train)
    coef = np.abs(selector.coef_)
    coef = np.where(np.isfinite(coef), coef, 0.0)
    order = np.argsort(coef)[::-1]
    positive = [idx for idx in order if coef[idx] > 0]
    selected = positive[:top_k] if positive else order[:top_k]
    return [X.columns[i] for i in selected], {X.columns[i]: float(coef[i]) for i in selected}


def make_sequences(x: np.ndarray, y: np.ndarray, indices: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index_set = set(indices.tolist())
    xs, ys, target_indices = [], [], []
    for idx in indices:
        if idx < lookback:
            continue
        window = np.arange(idx - lookback, idx)
        if all(i in index_set for i in window):
            xs.append(x[window])
            ys.append(y[idx])
            target_indices.append(idx)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), np.asarray(target_indices)


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def predict(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.tensor(x[start:start + batch_size], dtype=torch.float32, device=device)
            preds.append(model(batch).detach().cpu().numpy().ravel())
    return np.concatenate(preds)


def train_lstm(args: argparse.Namespace) -> None:
    X, y, timestamps = load_data()
    train_idx = np.flatnonzero((timestamps >= pd.Timestamp("2019-01-01T00:00:00Z")) & (timestamps <= pd.Timestamp("2021-12-31T23:00:00Z")))
    val_idx = np.flatnonzero((timestamps >= pd.Timestamp("2022-01-01T00:00:00Z")) & (timestamps <= pd.Timestamp("2022-12-31T23:00:00Z")))
    test_idx = np.flatnonzero((timestamps >= pd.Timestamp("2023-01-01T00:00:00Z")) & (timestamps <= pd.Timestamp("2023-12-31T23:00:00Z")))

    selected, scores = select_features(X, y, train_idx, args.feature_selection, args.top_k)
    X = X[selected]

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train_tab = imputer.fit_transform(X.iloc[train_idx])
    scaler.fit(x_train_tab)
    x_all = scaler.transform(imputer.transform(X)).astype(np.float32)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y.iloc[train_idx].to_numpy()[:, None]).ravel().astype(np.float32)
    y_all_scaled = y_scaler.transform(y.to_numpy()[:, None]).ravel().astype(np.float32)
    y_all_raw = y.to_numpy(dtype=np.float32)

    x_train, y_train, _ = make_sequences(x_all, y_all_scaled, train_idx, args.lookback)
    x_val, _, val_targets = make_sequences(x_all, y_all_scaled, val_idx, args.lookback)
    x_test, _, test_targets = make_sequences(x_all, y_all_scaled, test_idx, args.lookback)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = LSTMRegressor(
        input_size=x_train.shape[2],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    generator = torch.Generator(device="cpu").manual_seed(42)

    tx = torch.tensor(x_train, dtype=torch.float32, device=device)
    ty = torch.tensor(y_train[:, None], dtype=torch.float32, device=device)

    best_val = float("inf")
    best_state = None
    patience_left = args.patience
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(tx), generator=generator).tolist()
        losses = []
        for start in range(0, len(order), args.batch_size):
            batch = order[start:start + args.batch_size]
            opt.zero_grad()
            pred = model(tx[batch])
            loss = loss_fn(pred, ty[batch])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))

        val_scaled = predict(model, x_val, args.batch_size, device)
        val_pred = y_scaler.inverse_transform(val_scaled[:, None]).ravel()
        val_true = y_all_raw[val_targets]
        val_mae = mean_absolute_error(val_true, val_pred)
        print(f"epoch {epoch:03d}: train_loss={np.mean(losses):.5f} val_mae={val_mae:.4f}", flush=True)

        if val_mae < best_val:
            best_val = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"early stopping at epoch {epoch}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_scaled = predict(model, x_test, args.batch_size, device)
    test_pred = y_scaler.inverse_transform(test_scaled[:, None]).ravel()
    test_true = y_all_raw[test_targets]
    result = {
        "model": "lstm_full",
        "lookback": args.lookback,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "epochs_requested": args.epochs,
        "best_validation_mae": best_val,
        "feature_selection": args.feature_selection,
        "feature_count": len(selected),
        **metrics(test_true, test_pred),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    result_path = OUT_DIR / "lstm_full_results.json"
    metadata_path = OUT_DIR / "lstm_full_selected_features.json"
    pred_path = PRED_DIR / "lstm_full_test_predictions_2023.csv"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    metadata_path.write_text(json.dumps({"selected_features": selected, "scores": scores}, indent=2), encoding="utf-8")
    pd.DataFrame({
        "timestamp_utc": timestamps.iloc[test_targets].astype(str).to_numpy(),
        "y_true": test_true,
        "pred_lstm": test_pred,
    }).to_csv(pred_path, index=False)

    print(json.dumps(result, indent=2), flush=True)
    print(f"wrote result -> {result_path}")
    print(f"wrote selected features -> {metadata_path}")
    print(f"wrote predictions -> {pred_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-selection", choices=["none", "lasso", "elastic_net"], default="lasso")
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--lookback", type=int, default=168)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    args = parser.parse_args()
    train_lstm(args)


if __name__ == "__main__":
    main()
