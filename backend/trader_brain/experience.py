from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from .contracts import ExpertForecast, GateResult, RegimePosterior


class TraderExperienceStore:
    """Persistent state/action/outcome log for stacking, reliability and bandit learning."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        try:
            yield conn; conn.commit()
        except Exception:
            conn.rollback(); raise
        finally:
            conn.close()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _initialize(self) -> None:
        with self.connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS trader_brain_experiences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    feature_timestamp INTEGER NOT NULL,
                    target_timestamp INTEGER NOT NULL,
                    reference_price REAL NOT NULL,
                    position_before TEXT NOT NULL,
                    action TEXT NOT NULL,
                    gate_vector_json TEXT NOT NULL,
                    bandit_vector_json TEXT NOT NULL,
                    gate_json TEXT NOT NULL,
                    experts_json TEXT NOT NULL,
                    regime_json TEXT NOT NULL,
                    estimated_one_way_cost REAL NOT NULL,
                    resolved_at TEXT,
                    resolved_price REAL,
                    realized_return REAL,
                    actual_class INTEGER,
                    reward REAL,
                    shadow_rewards_json TEXT,
                    UNIQUE(model_id, feature_timestamp)
                );
                CREATE INDEX IF NOT EXISTS idx_trader_brain_due
                ON trader_brain_experiences(model_id, resolved_at, target_timestamp);
                """
            )

    def has_experience(self, model_id: str, feature_timestamp: int) -> bool:
        with self.connection() as conn:
            return conn.execute(
                "SELECT 1 FROM trader_brain_experiences WHERE model_id=? AND feature_timestamp=?",
                (model_id, int(feature_timestamp)),
            ).fetchone() is not None

    def record(
        self,
        *,
        model_id: str,
        feature_timestamp: int,
        target_timestamp: int,
        reference_price: float,
        position_before: str,
        action: str,
        gate_vector: Sequence[float],
        bandit_vector: Sequence[float],
        gate: GateResult,
        experts: Sequence[ExpertForecast],
        regime: RegimePosterior,
        estimated_one_way_cost: float,
    ) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO trader_brain_experiences (
                    model_id, created_at, feature_timestamp, target_timestamp, reference_price,
                    position_before, action, gate_vector_json, bandit_vector_json, gate_json,
                    experts_json, regime_json, estimated_one_way_cost
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model_id, self._now(), int(feature_timestamp), int(target_timestamp), float(reference_price),
                    position_before, action, json.dumps([float(v) for v in gate_vector]),
                    json.dumps([float(v) for v in bandit_vector]), json.dumps(gate.to_dict(), sort_keys=True),
                    json.dumps([expert.to_dict() for expert in experts], sort_keys=True),
                    json.dumps(regime.to_dict(), sort_keys=True), float(estimated_one_way_cost),
                ),
            )

    @staticmethod
    def _shadow_rewards(position_before: str, market_return: float, one_way_cost: float) -> dict[str, float]:
        if position_before == "LONG":
            return {"NO_TRADE": float(market_return), "EXIT": float(-one_way_cost)}
        return {"NO_TRADE": 0.0, "LONG": float(market_return - 2.0 * one_way_cost)}

    def resolve_due(
        self,
        *,
        model_id: str,
        current_timestamp: int,
        current_price: float,
        price_by_timestamp: Mapping[int, float] | None = None,
    ) -> int:
        price_by_timestamp = price_by_timestamp or {}
        with self.connection() as conn:
            rows = conn.execute(
                """SELECT * FROM trader_brain_experiences
                   WHERE model_id=? AND resolved_at IS NULL AND target_timestamp<=?
                   ORDER BY target_timestamp, id""",
                (model_id, int(current_timestamp)),
            ).fetchall()
            resolved = 0
            for row in rows:
                target_ts = int(row["target_timestamp"])
                target_price = price_by_timestamp.get(target_ts)
                if target_price is None:
                    if target_ts == int(current_timestamp):
                        target_price = float(current_price)
                    else:
                        # Never silently stretch a 4h learning horizon across downtime.
                        continue
                reference = float(row["reference_price"])
                if reference <= 0 or target_price <= 0:
                    continue
                market_return = float(target_price / reference - 1.0)
                cost = float(row["estimated_one_way_cost"])
                threshold = 2.0 * cost
                actual_class = 2 if market_return > threshold else 0 if market_return < -threshold else 1
                shadow = self._shadow_rewards(str(row["position_before"]), market_return, cost)
                reward = float(shadow.get(str(row["action"]), 0.0))
                conn.execute(
                    """UPDATE trader_brain_experiences
                       SET resolved_at=?, resolved_price=?, realized_return=?, actual_class=?, reward=?, shadow_rewards_json=?
                       WHERE id=?""",
                    (self._now(), float(target_price), market_return, actual_class, reward, json.dumps(shadow, sort_keys=True), int(row["id"])),
                )
                resolved += 1
            return resolved

    def _resolved(self, model_id: str, limit: int = 3000) -> list[sqlite3.Row]:
        with self.connection() as conn:
            return conn.execute(
                """SELECT * FROM trader_brain_experiences
                   WHERE model_id=? AND resolved_at IS NOT NULL ORDER BY id DESC LIMIT ?""",
                (model_id, max(1, int(limit))),
            ).fetchall()

    def gate_training_data(self, model_id: str, limit: int = 3000) -> tuple[np.ndarray, np.ndarray]:
        vectors: list[list[float]] = []; targets: list[int] = []; width: int | None = None
        for row in reversed(self._resolved(model_id, limit)):
            try:
                vector = [float(v) for v in json.loads(row["gate_vector_json"])]
                target = int(row["actual_class"])
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if target not in {0, 1, 2} or not np.all(np.isfinite(vector)):
                continue
            width = len(vector) if width is None else width
            if len(vector) != width:
                continue
            vectors.append(vector); targets.append(target)
        if not vectors:
            return np.empty((0, 0), dtype=np.float64), np.empty((0,), dtype=np.int32)
        return np.asarray(vectors, dtype=np.float64), np.asarray(targets, dtype=np.int32)

    def bandit_shadow_samples(self, model_id: str, limit: int = 3000) -> list[tuple[list[float], dict[str, float]]]:
        samples: list[tuple[list[float], dict[str, float]]] = []
        for row in reversed(self._resolved(model_id, limit)):
            if not row["shadow_rewards_json"]:
                continue
            try:
                vector = [float(v) for v in json.loads(row["bandit_vector_json"])]
                rewards = {str(k): float(v) for k, v in json.loads(row["shadow_rewards_json"]).items()}
            except (TypeError, ValueError, json.JSONDecodeError, AttributeError):
                continue
            if vector and np.all(np.isfinite(vector)) and all(np.isfinite(v) for v in rewards.values()):
                samples.append((vector, rewards))
        return samples

    def expert_reliability(
        self,
        model_id: str,
        limit: int = 300,
        *,
        prior_samples: float = 50.0,
    ) -> dict[str, float]:
        """Estimate trailing expert reliability with sparse-sample shrinkage.

        Raw reliability compares each expert's mean log loss with a uniform
        three-class forecast. With only a handful of resolved forward samples,
        that estimate is extremely noisy. We therefore shrink it toward the
        neutral value 1.0 using ``n / (n + prior_samples)``. Setting
        ``prior_samples=0`` reproduces the historical unshrunk estimate for
        diagnostics, while the default prevents a few early outcomes from
        materially reweighting the live mixture-of-experts gate.
        """
        if prior_samples < 0.0:
            raise ValueError("prior_samples must be non-negative")
        losses: dict[str, list[float]] = {}
        for row in self._resolved(model_id, limit):
            try:
                actual = int(row["actual_class"]); experts = json.loads(row["experts_json"])
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if actual not in {0, 1, 2} or not isinstance(experts, list):
                continue
            for expert in experts:
                if not isinstance(expert, dict) or not expert.get("available"):
                    continue
                try:
                    p = [float(expert["p_down"]), float(expert["p_flat"]), float(expert["p_up"])][actual]
                    loss = -float(np.log(np.clip(p, 1e-6, 1.0)))
                except (KeyError, TypeError, ValueError, IndexError):
                    continue
                losses.setdefault(str(expert.get("name", "unknown")), []).append(loss)
        uniform = float(np.log(3.0))
        reliability: dict[str, float] = {}
        for name, values in losses.items():
            if not values:
                continue
            raw = float(np.clip(np.exp(-(float(np.mean(values)) - uniform)), 0.20, 1.50))
            n = float(len(values))
            weight = 1.0 if prior_samples == 0.0 else n / (n + float(prior_samples))
            shrunk = 1.0 + weight * (raw - 1.0)
            reliability[name] = float(np.clip(shrunk, 0.20, 1.50))
        return reliability

    def report(self, model_id: str) -> dict[str, Any]:
        with self.connection() as conn:
            row = conn.execute(
                """SELECT COUNT(*) total,
                          SUM(CASE WHEN resolved_at IS NOT NULL THEN 1 ELSE 0 END) resolved,
                          AVG(CASE WHEN resolved_at IS NOT NULL THEN reward END) avg_reward,
                          AVG(CASE WHEN resolved_at IS NOT NULL THEN realized_return END) avg_market_return
                   FROM trader_brain_experiences WHERE model_id=?""",
                (model_id,),
            ).fetchone()
        return {
            "model_id": model_id,
            "experience_count": int(row["total"] or 0),
            "resolved_count": int(row["resolved"] or 0),
            "average_reward": float(row["avg_reward"] or 0.0),
            "average_market_return": float(row["avg_market_return"] or 0.0),
            "expert_reliability": self.expert_reliability(model_id),
        }
