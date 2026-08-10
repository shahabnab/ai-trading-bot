# GPU/server training and durable model storage

This workflow is designed for the `step-2-coinex-readonly` branch.

## Validation protocol

BTC is time-series data, so ordinary random K-fold is intentionally not used. The training script uses chronological walk-forward evaluation:

`TRAIN -> EARLY-STOP VALIDATION -> THRESHOLD VALIDATION -> UNTOUCHED TEST`

With the default 365/60/60/60-day configuration, the 60-day validation window is split chronologically. The first half controls model early stopping and the second half selects the existing `execution_lambda` trading hurdle. The outer 60-day test window is not used for either choice.

## Current trained approaches

- XGBoost: CPU is sufficient for the current hourly dataset; CUDA is optional.
- LSTM: TensorFlow uses an NVIDIA GPU when detected and this is the model that benefits most from the rented GPU server.

Moving-average and buy-and-hold are also reported as non-trained baselines.

## Threshold being optimized

The script tunes the existing execution hurdle used by the project:

`abs(predicted return) > execution_lambda * estimated transaction cost`

It does **not** tune `PAPER_MIN_CONFIDENCE`. The current XGBoost/LSTM models are regressors and do not yet produce calibrated confidence probabilities, so optimizing the confidence gate would not be statistically defensible yet.

The default cost assumptions for this server-training script are 20 bps fee + 5 bps slippage. They are explicit command-line parameters and are recorded in every report.

## Files produced

Each run is written under:

`artifacts/ml/server_training/<RUN_ID>/`

It contains:

- every XGBoost fold model
- every LSTM fold model
- every LSTM fold scaler
- fold predictions
- threshold sweep results
- fold reports
- final XGBoost deployment candidate
- final LSTM deployment candidate + scaler
- manifests and metadata
- `REPORT.md`
- `model_comparison.csv`

Two archives are created:

- `<RUN_ID>_training_report.zip`: complete run including fold models and diagnostics.
- `<RUN_ID>_deployment_models.zip`: final deployment candidates plus the key reports and metadata.

## Why GitHub Releases are used for trained models

The repository intentionally ignores `artifacts/`. Large/binary ML outputs should not be committed repeatedly into normal Git history because that makes the source repository grow quickly.

`publish_training_release.sh` instead uploads the two ZIP files as assets of a GitHub **prerelease** tagged `training-<RUN_ID>`. This keeps the trained models and reports permanently on GitHub without polluting the normal source-code history.

## GPU-server setup

Clone the branch and upload the dataset to:

`data/processed/training/btc_hourly.jsonl`

Authenticate GitHub CLI on the server with either:

```bash
export GH_TOKEN='YOUR_GITHUB_TOKEN'
```

or:

```bash
gh auth login
```

Do not commit the token.

Then run:

```bash
chmod +x run_training_gpu.sh publish_training_release.sh
./run_training_gpu.sh
```

Publishing to GitHub is ON by default. If GitHub authentication is unavailable, the publish step fails loudly after training, so do not destroy the GPU server until the release is successfully created.

For the dataset validated on 2026-08-10, you can enforce its exact hash before training:

```bash
export EXPECTED_DATASET_SHA256=9527c28914f269346f256541cd5d1661b91d697f0d2c369687593fb5200f28b4
./run_training_gpu.sh
```

To deliberately skip publishing:

```bash
PUBLISH_TO_GITHUB=0 ./run_training_gpu.sh
```

## After training

Verify the GitHub prerelease exists and contains both the full training ZIP and the deployment ZIP. Only then terminate the ephemeral GPU machine.

The smaller deployment ZIP can later be downloaded to a cheap CPU VPS for hourly inference.

The final deployment candidates are trained on all currently available data **after** the walk-forward evaluation is complete. They are candidates for forward paper testing, not proof of future profitability.
