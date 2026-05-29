# Modeling Environment Check

检查日期：2026-05-28

## 1. 系统环境

- 系统：macOS arm64
- 芯片：Apple M5
- Python：3.9.6
- pip：21.2.4
- Homebrew：`/opt/homebrew/bin/brew`

当前 shell 的 `PATH` 未包含 `/opt/homebrew/bin`，因此直接运行 `brew` 可能找不到。需要时使用：

```bash
/opt/homebrew/bin/brew ...
```

或在 shell 配置中加入：

```bash
export PATH="/opt/homebrew/bin:$PATH"
```

## 2. 已安装并验证的建模库

| 类别 | 包 | 状态 | 说明 |
|---|---|---|---|
| 数据处理 | `numpy` 2.0.2 | 可用 | 已安装 |
| 数据处理 | `pandas` 2.3.3 | 可用 | 已安装 |
| 机器学习 | `scikit-learn` 1.6.1 | 可用 | RandomForest、HistGradientBoosting、ElasticNet 已训练通过 |
| 机器学习 | `xgboost` 2.1.4 | 可用 | 安装 `libomp` 后训练通过 |
| 机器学习 | `lightgbm` 4.6.0 | 可用 | 安装 `libomp` 后训练通过 |
| 机器学习 | `catboost` 1.2.10 | 可用 | 小样本训练通过 |
| 传统时序 | `statsmodels` 0.14.6 | 可用 | ARIMA 小样本预测通过 |
| 深度学习 | `torch` 2.8.0 | 可用 | LSTM/RNN 小样本训练通过 |

PyTorch MPS 状态：

```text
torch.backends.mps.is_available() = True
```

这意味着后续 LSTM、RNN、TCN、Transformer 可以尝试使用 Apple GPU 加速。

## 3. 已修复的问题

`XGBoost` 和 `LightGBM` 初始报错：

```text
Library not loaded: @rpath/libomp.dylib
```

原因：macOS 上缺少 OpenMP runtime。

已通过 Homebrew 安装：

```bash
/opt/homebrew/bin/brew install libomp
```

安装位置：

```text
/opt/homebrew/opt/libomp/lib/libomp.dylib
```

安装后 `XGBoost` 和 `LightGBM` 均已完成小样本训练测试。

## 4. 当前建议

第一阶段主力模型按 `configs/modeling_config.json` 收紧为：

- `LightGBM`
- `XGBoost`
- `CatBoost`
- `ElasticNet`
- PyTorch `LSTM`
- PyTorch `RNN`

基线模型：

- 历史同小时均值
- `ARIMA`

深度学习模型：

- PyTorch `LSTM`
- PyTorch `RNN`
- 后续可扩展 `TCN` 或 Transformer

不建议第一阶段使用 TensorFlow。macOS Apple Silicon 上 TensorFlow 配置更容易受版本影响；当前 PyTorch 已经足够覆盖 LSTM/RNN/Transformer 需求。

## 5. 复现安装命令

```bash
python3 -m pip install --user scikit-learn xgboost lightgbm statsmodels torch catboost
/opt/homebrew/bin/brew install libomp
```

## 6. 后续注意

- 如果新开终端后 `xgboost` 或 `lightgbm` 再次找不到 `libomp`，优先确认 `/opt/homebrew/opt/libomp/lib/libomp.dylib` 是否存在。
- 如果命令行找不到 `brew`，使用 `/opt/homebrew/bin/brew`。
- 大规模训练时，表格模型优先 CPU；PyTorch 深度学习可以尝试 MPS。
