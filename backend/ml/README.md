# AI Core — XGBoost + LSTM

The repository now contains two independent BTC next-hour return models evaluated under the same causal protocol:

1. `backend.ml.xgboost_core` — tabular XGBoost baseline.
2. `backend.ml.lstm_core` — sequence LSTM baseline.

Neither model can place live orders. Both produce out-of-sample forecasts and pass them through the same cost-aware long-only backtest and deterministic baselines.

## Shared methodology

- Causal hourly features only; sentiment remains opt-in.
- Target: next-hour log return derived from `target_return_1h`.
- Fixed rolling walk-forward evaluation; no random train/test shuffle.
- Default six-month-friendly windows: 90d train / 30d validation / 30d test / 30d step.
- Validation is used for early stopping; test windows are untouched until final fold evaluation.
- Fee, slippage, and spread assumptions are included in the strategy backtest.
- Benchmarks: buy-and-hold and EMA20 > EMA50 with price > EMA200.
- Artifacts use the repository's existing `RunLogger`.

## Install

The ML dependencies are:

```text
numpy>=2.0,<3.0
xgboost>=3.0,<4.0
tensorflow>=2.21,<2.22
```

Update the environment:

```powershell
pip install -r requirements.txt
```

TensorFlow 2.21 supports the project's Python 3.11 environment. Native Windows TensorFlow runs on CPU; use WSL2 or Linux for TensorFlow GPU acceleration.

## Build historical dataset

```powershell
python -m backend.data_collection.historical_dataset --months 6 --skip-news
```

The default model input is:

```text
data/processed/training/btc_hourly.jsonl
```

## Step 1 — XGBoost

```powershell
python -m backend.ml.xgboost_core `
  --train-days 90 `
  --validation-days 30 `
  --test-days 30 `
  --step-days 30 `
  --fee-bps 5 `
  --slippage-bps 3 `
  --spread-bps 2 `
  --execution-lambda 2
```

XGBoost artifacts include per-fold JSON models and gain feature importance.

## Step 2 — LSTM

The LSTM consumes rolling sequences of the same causal feature matrix. The default history is 48 hourly bars. For every walk-forward fold:

- feature normalization is fitted on the training region only;
- target normalization is fitted on the training region only;
- training sequences cannot reach before that fold's training start;
- validation/test sequences may use already-observed historical context but never future feature rows;
- timestamp continuity is checked, so a missing hourly bar invalidates a sequence rather than silently changing the time scale;
- Keras early stopping restores the best validation weights.

Run:

```powershell
python -m backend.ml.lstm_core `
  --train-days 90 `
  --validation-days 30 `
  --test-days 30 `
  --step-days 30 `
  --sequence-length 48 `
  --lstm-units 64 `
  --dense-units 32 `
  --dropout 0.20 `
  --epochs 50 `
  --batch-size 64 `
  --early-stopping-patience 7 `
  --fee-bps 5 `
  --slippage-bps 3 `
  --spread-bps 2 `
  --execution-lambda 2
```

For the first comparison, keep sentiment disabled for both algorithms and keep the same walk-forward/cost arguments.

## Artifacts

All runs are written under:

```text
artifacts/ml/runs/<run-id>/
```

Common files:

- `run.json`
- `events.jsonl`
- `summary.json`
- `predictions.jsonl`

XGBoost adds:

- `feature_importance_gain.json`
- `models/fold_XX.json`

LSTM adds:

- `models/fold_XX.keras`
- `scalers/fold_XX.json`

The saved scaler file is required with the corresponding LSTM model because preprocessing parameters are fold-specific and fitted only from training data.
