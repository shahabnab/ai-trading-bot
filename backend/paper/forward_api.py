from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query

from backend.config import settings
from backend.paper.model_catalog import PAPER_MODELS
from backend.paper.model_engine import ModelPaperStore


router = APIRouter(prefix="/api/paper/forward-v3", tags=["forward-v3"])
DEFAULT_DEPLOYMENT_ROOT = Path("artifacts/ml/forward_deployment/v3-paper")
DEFAULT_STATE_ROOT = Path("state/forward_v3")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    rows.sort(key=lambda row: int(row.get("feature_timestamp", 0) or 0))
    return rows


def _artifact_status(model_id: str) -> dict[str, object]:
    root = DEFAULT_DEPLOYMENT_ROOT / model_id
    required = {
        "model": root / "model.keras",
        "manifest": root / "manifest.json",
        "standardizer": root / "standardizer.json",
    }
    present = {name: path.is_file() for name, path in required.items()}
    return {
        "ready": all(present.values()),
        "files": present,
    }


@router.get("")
def forward_v3_status(limit: int = Query(default=720, ge=1, le=5000)) -> dict[str, object]:
    """Read-only telemetry for the frozen V3 prospective paper experiment.

    This endpoint never performs inference and never places a trade. It exposes
    the immutable hourly prediction log, model-specific paper trades, and
    artifact readiness so the dashboard can show exactly why a paper deal was
    or was not made.
    """
    records = _read_jsonl(DEFAULT_STATE_ROOT / "predictions.jsonl")
    store = ModelPaperStore(settings.paper_db_path, settings.paper_model_initial_balance_eur_equiv)

    models: list[dict[str, object]] = []
    for spec in PAPER_MODELS:
        store.ensure_account(spec.model_id, spec.display_name)
        history = [row for row in records if row.get("model_id") == spec.model_id][-limit:]
        models.append(
            {
                **spec.to_dict(),
                "artifact": _artifact_status(spec.model_id),
                "latest": history[-1] if history else None,
                "history": history,
                "trades": store.list_trades(spec.model_id, min(limit, 1000)),
                "decisions": store.list_decisions(spec.model_id, min(limit, 1000)),
                "performance": store.performance_summary(spec.model_id),
                "account": store.get_account(spec.model_id),
                "positions": store.get_positions(spec.model_id),
            }
        )

    return {
        "mode": "prospective_forward_paper",
        "paper_only": True,
        "real_orders_enabled": False,
        "prediction_log_exists": (DEFAULT_STATE_ROOT / "predictions.jsonl").is_file(),
        "models": models,
    }
