from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4


class ModelRegistryError(RuntimeError):
    """Raised when a model artifact or registry entry is invalid."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(path)


@dataclass(frozen=True)
class ArtifactRecord:
    name: str
    relative_path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class ModelManifest:
    model_id: str
    created_at: str
    model_type: str
    symbol: str
    sampling_period: str
    prediction_horizon: str
    feature_version: str
    dataset_version: str
    git_commit: str | None
    validation_method: str
    training_start: str | None
    training_end: str | None
    metrics: dict[str, Any]
    metadata: dict[str, Any]
    artifacts: list[ArtifactRecord]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["artifacts"] = [asdict(item) for item in self.artifacts]
        return payload


class ModelRegistry:
    """Filesystem-backed model registry designed for machine-to-machine portability.

    Each model version lives in its own directory with a manifest and copied model
    artifacts. Pointers for latest/best/active are stored as small JSON files. The
    entire registry directory can therefore be copied or synced to another machine.
    """

    POINTERS = {"latest", "best", "active"}

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.models_dir = self.root / "models"
        self.pointers_dir = self.root / "pointers"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.pointers_dir.mkdir(parents=True, exist_ok=True)

    def register_model(
        self,
        *,
        model_id: str,
        model_type: str,
        symbol: str,
        sampling_period: str,
        prediction_horizon: str,
        feature_version: str,
        dataset_version: str,
        validation_method: str,
        artifact_paths: Iterable[str | Path],
        metrics: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        git_commit: str | None = None,
        training_start: str | None = None,
        training_end: str | None = None,
        set_latest: bool = True,
    ) -> dict[str, Any]:
        normalized_id = model_id.strip()
        if not normalized_id:
            raise ModelRegistryError("model_id cannot be empty")

        model_dir = self.models_dir / normalized_id
        if model_dir.exists():
            raise ModelRegistryError(f"Model '{normalized_id}' already exists")

        artifact_dir = model_dir / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=False)

        records: list[ArtifactRecord] = []
        try:
            for raw_path in artifact_paths:
                source = Path(raw_path)
                if not source.is_file():
                    raise ModelRegistryError(f"Artifact does not exist: {source}")
                destination = artifact_dir / source.name
                if destination.exists():
                    raise ModelRegistryError(
                        f"Duplicate artifact filename for model '{normalized_id}': {source.name}"
                    )
                shutil.copy2(source, destination)
                records.append(
                    ArtifactRecord(
                        name=source.name,
                        relative_path=str(destination.relative_to(model_dir)),
                        sha256=_sha256(destination),
                        size_bytes=destination.stat().st_size,
                    )
                )

            if not records:
                raise ModelRegistryError("At least one model artifact is required")

            manifest = ModelManifest(
                model_id=normalized_id,
                created_at=_utc_now(),
                model_type=model_type,
                symbol=symbol.upper(),
                sampling_period=sampling_period,
                prediction_horizon=prediction_horizon,
                feature_version=feature_version,
                dataset_version=dataset_version,
                git_commit=git_commit,
                validation_method=validation_method,
                training_start=training_start,
                training_end=training_end,
                metrics=metrics or {},
                metadata=metadata or {},
                artifacts=records,
            )
            _write_json_atomic(model_dir / "manifest.json", manifest.as_dict())
        except Exception:
            shutil.rmtree(model_dir, ignore_errors=True)
            raise

        if set_latest:
            self.set_pointer("latest", normalized_id)
        return manifest.as_dict()

    def list_models(self) -> list[dict[str, Any]]:
        manifests: list[dict[str, Any]] = []
        for manifest_path in self.models_dir.glob("*/manifest.json"):
            try:
                manifests.append(json.loads(manifest_path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        manifests.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return manifests

    def get_manifest(self, model_id: str) -> dict[str, Any]:
        path = self.models_dir / model_id / "manifest.json"
        if not path.is_file():
            raise ModelRegistryError(f"Model '{model_id}' was not found")
        return json.loads(path.read_text(encoding="utf-8"))

    def set_pointer(self, pointer: str, model_id: str) -> dict[str, str]:
        if pointer not in self.POINTERS:
            raise ModelRegistryError(f"Unknown model pointer: {pointer}")
        self.get_manifest(model_id)
        payload = {"model_id": model_id, "updated_at": _utc_now()}
        _write_json_atomic(self.pointers_dir / f"{pointer}.json", payload)
        return payload

    def get_pointer(self, pointer: str) -> dict[str, str] | None:
        if pointer not in self.POINTERS:
            raise ModelRegistryError(f"Unknown model pointer: {pointer}")
        path = self.pointers_dir / f"{pointer}.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def resolve_pointer(self, pointer: str) -> dict[str, Any] | None:
        value = self.get_pointer(pointer)
        if value is None:
            return None
        return self.get_manifest(value["model_id"])

    def verify_model(self, model_id: str) -> dict[str, Any]:
        manifest = self.get_manifest(model_id)
        model_dir = self.models_dir / model_id
        results: list[dict[str, Any]] = []
        all_valid = True

        for artifact in manifest.get("artifacts", []):
            path = model_dir / artifact["relative_path"]
            exists = path.is_file()
            actual_hash = _sha256(path) if exists else None
            expected_hash = artifact["sha256"]
            valid = exists and actual_hash == expected_hash
            all_valid = all_valid and valid
            results.append(
                {
                    "name": artifact["name"],
                    "exists": exists,
                    "valid": valid,
                    "expected_sha256": expected_hash,
                    "actual_sha256": actual_hash,
                }
            )

        return {"model_id": model_id, "valid": all_valid, "artifacts": results}

    def summary(self) -> dict[str, Any]:
        models = self.list_models()
        return {
            "model_count": len(models),
            "latest": self.get_pointer("latest"),
            "best": self.get_pointer("best"),
            "active": self.get_pointer("active"),
            "models": models,
        }


class RunLogger:
    """Append-only JSONL experiment logger with persistent run metadata."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def start_run(
        self,
        *,
        purpose: str,
        model_family: str,
        dataset_version: str,
        feature_version: str,
        config: dict[str, Any] | None = None,
        machine: dict[str, Any] | None = None,
        git_commit: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        identifier = run_id or f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
        run_dir = self.root / identifier
        if run_dir.exists():
            raise ModelRegistryError(f"Run '{identifier}' already exists")
        run_dir.mkdir(parents=True)

        payload = {
            "run_id": identifier,
            "status": "running",
            "started_at": _utc_now(),
            "finished_at": None,
            "purpose": purpose,
            "model_family": model_family,
            "dataset_version": dataset_version,
            "feature_version": feature_version,
            "git_commit": git_commit,
            "config": config or {},
            "machine": machine or {},
            "summary": {},
        }
        _write_json_atomic(run_dir / "run.json", payload)
        (run_dir / "events.jsonl").touch()
        return payload

    def log_event(self, run_id: str, event: str, **payload: Any) -> dict[str, Any]:
        run_dir = self._run_dir(run_id)
        record = {"timestamp": _utc_now(), "event": event, **payload}
        with (run_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    def finish_run(
        self,
        run_id: str,
        *,
        status: str = "completed",
        summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        run_dir = self._run_dir(run_id)
        path = run_dir / "run.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["status"] = status
        payload["finished_at"] = _utc_now()
        payload["summary"] = summary or {}
        _write_json_atomic(path, payload)
        return payload

    def get_run(self, run_id: str) -> dict[str, Any]:
        run_dir = self._run_dir(run_id)
        return json.loads((run_dir / "run.json").read_text(encoding="utf-8"))

    def list_runs(self) -> list[dict[str, Any]]:
        runs: list[dict[str, Any]] = []
        for run_path in self.root.glob("*/run.json"):
            try:
                runs.append(json.loads(run_path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        runs.sort(key=lambda item: item.get("started_at", ""), reverse=True)
        return runs

    def _run_dir(self, run_id: str) -> Path:
        path = self.root / run_id
        if not (path / "run.json").is_file():
            raise ModelRegistryError(f"Run '{run_id}' was not found")
        return path
