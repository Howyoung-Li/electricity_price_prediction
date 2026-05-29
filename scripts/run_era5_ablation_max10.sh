#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="reports/modeling"
mkdir -p "$LOG_DIR" data/predictions

echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "cwd=$ROOT_DIR"

run_experiment() {
  local name="$1"
  shift
  echo "==== ${name} started $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
  python3 scripts/train_tabular_models.py \
    --experiment-name "$name" \
    --max-trials 10 \
    --include-sequence \
    --sequence-epochs 10 \
    --lookback 168 \
    --hidden-size 64 \
    --batch-size 128 \
    --sequence-feature-selection lasso \
    --sequence-top-k 40 \
    --stacking-max-base-mae-ratio 1.5 \
    "$@"
  echo "==== ${name} finished $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
}

run_experiment "era5_ablation_with_era5_max10"
run_experiment "era5_ablation_without_era5_max10" --drop-era5

python3 - <<'PY'
from pathlib import Path
import pandas as pd

report_dir = Path("reports/modeling")
experiments = {
    "with_era5": report_dir / "era5_ablation_with_era5_max10_results.csv",
    "without_era5": report_dir / "era5_ablation_without_era5_max10_results.csv",
}

frames = []
for feature_set, path in experiments.items():
    df = pd.read_csv(path)
    df.insert(0, "feature_set", feature_set)
    frames.append(df)

all_results = pd.concat(frames, ignore_index=True)
summary_path = report_dir / "era5_ablation_max10_all_results.csv"
all_results.to_csv(summary_path, index=False)

best = all_results.sort_values("mae").groupby("feature_set", as_index=False).first()
best_path = report_dir / "era5_ablation_max10_summary.csv"
best.to_csv(best_path, index=False)

with_era5_mae = float(best.loc[best["feature_set"] == "with_era5", "mae"].iloc[0])
without_era5_mae = float(best.loc[best["feature_set"] == "without_era5", "mae"].iloc[0])
delta = with_era5_mae - without_era5_mae

display_cols = [
    "feature_set",
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
table = best[display_cols].copy()
header = "| " + " | ".join(display_cols) + " |"
separator = "| " + " | ".join(["---"] * len(display_cols)) + " |"
rows = []
for _, row in table.iterrows():
    values = []
    for col in display_cols:
        value = row[col]
        if isinstance(value, float):
            values.append(f"{value:.4f}")
        else:
            values.append(str(value))
    rows.append("| " + " | ".join(values) + " |")

md = [
    "# ERA5 Ablation Max10 Summary",
    "",
    f"- Best with ERA5 MAE: {with_era5_mae:.4f}",
    f"- Best without ERA5 MAE: {without_era5_mae:.4f}",
    f"- Delta MAE, with minus without: {delta:.4f}",
    "",
    "Negative delta means ERA5 improved the best model.",
    "",
    header,
    separator,
    *rows,
]
(report_dir / "era5_ablation_max10_summary.md").write_text("\n".join(md), encoding="utf-8")

print(f"wrote {summary_path}")
print(f"wrote {best_path}")
print("wrote reports/modeling/era5_ablation_max10_summary.md")
PY

echo "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
