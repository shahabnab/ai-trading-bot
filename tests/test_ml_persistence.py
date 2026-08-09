import json

import pytest

from backend.ml import ModelRegistry, ModelRegistryError, RunLogger


def test_model_registry_register_verify_and_restore(tmp_path):
    source = tmp_path / "model.bin"
    source.write_bytes(b"trained-model-bytes")

    registry_root = tmp_path / "registry"
    registry = ModelRegistry(registry_root)
    manifest = registry.register_model(
        model_id="btc4h-market-v001",
        model_type="xgboost",
        symbol="btcusdt",
        sampling_period="15m",
        prediction_horizon="4h",
        feature_version="market-v1",
        dataset_version="btc-v1",
        validation_method="purged-walk-forward",
        artifact_paths=[source],
        metrics={"balanced_accuracy": 0.56},
        metadata={"feature_count": 42},
        git_commit="abc123",
    )

    assert manifest["model_id"] == "btc4h-market-v001"
    assert manifest["symbol"] == "BTCUSDT"
    assert registry.get_pointer("latest")["model_id"] == "btc4h-market-v001"
    assert registry.verify_model("btc4h-market-v001")["valid"] is True

    registry.set_pointer("best", "btc4h-market-v001")
    registry.set_pointer("active", "btc4h-market-v001")

    # Re-open from disk to simulate restarting on another machine.
    restored = ModelRegistry(registry_root)
    assert restored.resolve_pointer("active")["model_id"] == "btc4h-market-v001"
    assert restored.resolve_pointer("best")["metrics"]["balanced_accuracy"] == 0.56


def test_registry_detects_tampered_artifact(tmp_path):
    source = tmp_path / "model.bin"
    source.write_bytes(b"original")
    registry = ModelRegistry(tmp_path / "registry")
    registry.register_model(
        model_id="btc-v1",
        model_type="lightgbm",
        symbol="BTCUSDT",
        sampling_period="15m",
        prediction_horizon="4h",
        feature_version="f1",
        dataset_version="d1",
        validation_method="walk-forward",
        artifact_paths=[source],
    )

    artifact = tmp_path / "registry" / "models" / "btc-v1" / "artifacts" / "model.bin"
    artifact.write_bytes(b"changed")
    result = registry.verify_model("btc-v1")
    assert result["valid"] is False
    assert result["artifacts"][0]["valid"] is False


def test_model_registry_rejects_duplicate_ids(tmp_path):
    source = tmp_path / "model.bin"
    source.write_bytes(b"model")
    registry = ModelRegistry(tmp_path / "registry")
    kwargs = dict(
        model_id="btc-v1",
        model_type="xgboost",
        symbol="BTCUSDT",
        sampling_period="15m",
        prediction_horizon="4h",
        feature_version="f1",
        dataset_version="d1",
        validation_method="walk-forward",
        artifact_paths=[source],
    )
    registry.register_model(**kwargs)
    with pytest.raises(ModelRegistryError):
        registry.register_model(**kwargs)


def test_run_logger_survives_restart(tmp_path):
    root = tmp_path / "runs"
    logger = RunLogger(root)
    run = logger.start_run(
        run_id="run-001",
        purpose="btc-4h-training",
        model_family="xgboost",
        dataset_version="btc-v1",
        feature_version="market-v1",
        config={"max_depth": 5},
        machine={"provider": "cloud-gpu"},
    )
    logger.log_event(run["run_id"], "epoch", step=1, validation_loss=0.42)
    logger.finish_run(run["run_id"], summary={"best_score": 0.58})

    restored = RunLogger(root)
    loaded = restored.get_run("run-001")
    assert loaded["status"] == "completed"
    assert loaded["summary"]["best_score"] == 0.58

    events_path = root / "run-001" / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert events[0]["event"] == "epoch"
    assert events[0]["step"] == 1
