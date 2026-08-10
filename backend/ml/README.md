# AI Core Step 1 — XGBoost

This patch adds the first model only: next-hour BTC log-return forecasting with XGBoost.
It deliberately does not add LSTM, Mamba, Transformers, or live execution.

## Methodology

- Causal hourly technical features only; sentiment is opt-in.
- Target: next-hour log return, derived from the existing `target_return_1h` field.
- Fixed rolling walk-forward evaluation; no random shuffle.
- Default six-month-friendly windows: 90d train / 30d validation / 30d test / 30d step.
- Early stopping on validation only.
- Out-of-sample predictions are evaluated once.
- Cost-aware long-only execution with separate fee, slippage, and spread assumptions plus a hurdle multiplier.
- Benchmarks: buy-and-hold and EMA20 > EMA50 with price > EMA200.
- Run artifacts use the repository's existing `RunLogger`.

## Install

Add these to both `requirements.txt` and `environment.yml` pip dependencies:

```text
numpy>=2.0,<3.0
xgboost>=3.0,<4.0
```

Then update the environment:

```powershell
pip install -r requirements.txt
```

## Build historical dataset

```powershell
python -m backend.data_collection.historical_dataset --months 6 --skip-news
```

## Train/evaluate XGBoost

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

For the first run, keep sentiment disabled. Once the price/technical-only benchmark is stable,
rerun with `--include-sentiment` and compare the OOS result as an ablation.

Artifacts are written under `artifacts/ml/runs/<run-id>/`:

- `run.json`
- `events.jsonl`
- `summary.json`
- `predictions.jsonl`
- `feature_importance_gain.json`
- `models/fold_XX.json`
