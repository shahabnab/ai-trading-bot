from __future__ import annotations

import asyncio
import json
import math
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from backend.config import settings
from backend.data_collection.binance_research_context import (
    _aggregate_aggtrade_archive,
    _finalize_agg_hour,
)
from backend.market import CoinExMarketClient
from backend.ml.calibration import PayoffEstimate, PlattCalibrator
from backend.ml.context_features import build_context_feature_matrix
from backend.ml.features import build_feature_dataset
from backend.paper.model_catalog import PAPER_MODELS, get_paper_model
from backend.paper.model_engine import ModelPaperStore


HOUR_MS = 60 * 60 * 1000
DAY_MS = 24 * HOUR_MS
SPOT_BASE_URL = "https://data-api.binance.vision"
FUTURES_BASE_URL = "https://fapi.binance.com"
BINANCE_ARCHIVE_BASE = "https://data.binance.vision/data"
DEFAULT_DEPLOYMENT_ROOT = Path("artifacts/ml/forward_deployment/v3-paper")
DEFAULT_STATE_ROOT = Path("state/forward_v3")


class ForwardV3Error(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ForwardV3Error(f"cannot read JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ForwardV3Error(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def _read_jsonl_map(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                timestamp = int(row["timestamp"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
            if isinstance(row, dict):
                rows[timestamp] = row
    return rows


def _write_jsonl_map(path: Path, rows: dict[int, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for timestamp in sorted(rows):
            handle.write(json.dumps(rows[timestamp], sort_keys=True, separators=(",", ":")) + "\n")


def _latest_completed_hour_ms(now: datetime | None = None) -> int:
    observed = now or datetime.now(UTC)
    now_ms = int(observed.timestamp() * 1000)
    return now_ms // HOUR_MS * HOUR_MS


def _iso(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000.0, UTC).isoformat()


async def _get_json(client: httpx.AsyncClient, url: str, params: dict[str, object]) -> Any:
    try:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise ForwardV3Error(f"market-data request failed: {url}: {exc}") from exc


def _spot_training_row(kline: list[Any]) -> dict[str, Any]:
    open_time = int(kline[0])
    quote_volume = float(kline[7])
    taker_quote = float(kline[10])
    timestamp = open_time + HOUR_MS
    return {
        "timestamp": timestamp,
        "feature_window_start": open_time,
        "feature_window_end": timestamp,
        "symbol": "BTCUSDT",
        "open": float(kline[1]),
        "high": float(kline[2]),
        "low": float(kline[3]),
        "close": float(kline[4]),
        "volume": float(kline[5]),
        "quote_volume": quote_volume,
        "number_of_trades": int(kline[8]),
        "taker_buy_quote_ratio": taker_quote / quote_volume if quote_volume > 0.0 else 0.5,
        # The feature builder does not use the target as an input. A finite
        # placeholder lets it retain the newest fully closed candle for inference.
        "target_return_1h": 0.0,
        "news_count": 0,
        "news_overall_sentiment_mean": 0.0,
        "news_btc_sentiment_mean": 0.0,
        "news_btc_relevance_mean": 0.0,
        "fear_greed_value": None,
    }


def _context_kline_row(kline: list[Any], source: str) -> dict[str, Any]:
    open_time = int(kline[0])
    quote_volume = float(kline[7])
    taker_quote = float(kline[10])
    return {
        "timestamp": open_time + HOUR_MS,
        "source": source,
        "open": float(kline[1]),
        "high": float(kline[2]),
        "low": float(kline[3]),
        "close": float(kline[4]),
        "base_volume": float(kline[5]),
        "quote_volume": quote_volume,
        "trade_count": int(kline[8]),
        "taker_buy_quote_ratio": taker_quote / quote_volume if quote_volume > 0.0 else 0.5,
        "minute_count": 60,
    }


async def _spot_klines(client: httpx.AsyncClient, symbol: str, limit: int, latest_hour: int) -> list[list[Any]]:
    payload = await _get_json(
        client,
        f"{SPOT_BASE_URL}/api/v3/klines",
        {"symbol": symbol, "interval": "1h", "limit": min(max(limit, 1), 1000)},
    )
    if not isinstance(payload, list):
        raise ForwardV3Error(f"unexpected Binance spot kline payload for {symbol}")
    rows = [row for row in payload if isinstance(row, list) and len(row) >= 11 and int(row[0]) + HOUR_MS <= latest_hour]
    if not rows:
        raise ForwardV3Error(f"no completed Binance spot klines for {symbol}")
    return rows


async def _futures_klines(client: httpx.AsyncClient, symbol: str, limit: int, latest_hour: int) -> list[list[Any]]:
    payload = await _get_json(
        client,
        f"{FUTURES_BASE_URL}/fapi/v1/klines",
        {"symbol": symbol, "interval": "1h", "limit": min(max(limit, 1), 1000)},
    )
    if not isinstance(payload, list):
        raise ForwardV3Error(f"unexpected Binance futures kline payload for {symbol}")
    rows = [row for row in payload if isinstance(row, list) and len(row) >= 11 and int(row[0]) + HOUR_MS <= latest_hour]
    if not rows:
        raise ForwardV3Error(f"no completed Binance futures klines for {symbol}")
    return rows


async def _download_daily_aggtrades(client: httpx.AsyncClient, day: date) -> list[dict[str, Any]] | None:
    stamp = day.isoformat()
    name = f"BTCUSDT-aggTrades-{stamp}.zip"
    url = f"{BINANCE_ARCHIVE_BASE}/spot/daily/aggTrades/BTCUSDT/{name}"
    try:
        response = await client.get(url)
    except httpx.HTTPError as exc:
        raise ForwardV3Error(f"Binance aggTrade archive request failed: {exc}") from exc
    if response.status_code == 404:
        return None
    try:
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise ForwardV3Error(f"Binance aggTrade archive request failed: {url}: {exc}") from exc
    return _aggregate_aggtrade_archive(response.content, name)


async def _fetch_aggtrade_hour(client: httpx.AsyncClient, hour_start: int) -> dict[str, Any]:
    hour_end = hour_start + HOUR_MS
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    params: dict[str, object] = {
        "symbol": "BTCUSDT",
        "startTime": hour_start,
        "endTime": hour_end - 1,
        "limit": 1000,
    }

    while True:
        payload = await _get_json(client, f"{SPOT_BASE_URL}/api/v3/aggTrades", params)
        if not isinstance(payload, list):
            raise ForwardV3Error("unexpected Binance aggTrades payload")
        if not payload:
            break

        reached_end = False
        for item in payload:
            if not isinstance(item, dict):
                continue
            try:
                trade_id = int(item["a"])
                timestamp = int(item["T"])
            except (KeyError, TypeError, ValueError):
                continue
            if timestamp >= hour_end:
                reached_end = True
                continue
            if timestamp < hour_start or trade_id in seen:
                continue
            seen.add(trade_id)
            rows.append(item)

        if reached_end or len(payload) < 1000:
            break
        try:
            last_id = int(payload[-1]["a"])
        except (KeyError, TypeError, ValueError):
            break
        params = {"symbol": "BTCUSDT", "fromId": last_id + 1, "limit": 1000}

    if not rows:
        raise ForwardV3Error(f"no BTCUSDT aggTrades for completed hour {_iso(hour_end)}")

    rows.sort(key=lambda item: int(item["T"]))
    prices: list[float] = []
    quote_sizes: list[float] = []
    aggressive_buy: list[bool] = []
    for item in rows:
        price = float(item["p"])
        quantity = float(item["q"])
        prices.append(price)
        quote_sizes.append(max(price * quantity, 0.0))
        aggressive_buy.append(not bool(item["m"]))
    return _finalize_agg_hour(hour_start, prices, quote_sizes, aggressive_buy)


async def _ensure_micro_cache(
    client: httpx.AsyncClient,
    *,
    cache_path: Path,
    latest_hour: int,
    required_hours: int,
) -> None:
    cache = _read_jsonl_map(cache_path)
    first_required = latest_hour - (required_hours - 1) * HOUR_MS
    required = list(range(first_required, latest_hour + 1, HOUR_MS))

    start_day = datetime.fromtimestamp((first_required - HOUR_MS) / 1000.0, UTC).date()
    today = datetime.now(UTC).date()
    day = start_day
    while day < today:
        day_start = int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp() * 1000)
        day_required = [ts for ts in required if day_start < ts <= day_start + DAY_MS]
        if day_required and any(ts not in cache for ts in day_required):
            archive_rows = await _download_daily_aggtrades(client, day)
            if archive_rows:
                for row in archive_rows:
                    cache[int(row["timestamp"])] = row
        day += timedelta(days=1)

    # Fill any remaining gaps (normally only completed hours from the current UTC day)
    # with the official public aggregate-trades REST endpoint.
    for timestamp in required:
        if timestamp in cache:
            continue
        cache[timestamp] = await _fetch_aggtrade_hour(client, timestamp - HOUR_MS)

    missing = [timestamp for timestamp in required if timestamp not in cache]
    if missing:
        raise ForwardV3Error(
            f"microstructure cache is missing {len(missing)} required hourly rows; first={_iso(missing[0])}"
        )

    # Keep a modest rolling cache instead of allowing this file to grow forever.
    keep_after = first_required - 48 * HOUR_MS
    cache = {timestamp: row for timestamp, row in cache.items() if timestamp >= keep_after}
    _write_jsonl_map(cache_path, cache)


def _write_context_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    mapping = {int(row["timestamp"]): row for row in rows}
    _write_jsonl_map(path, mapping)


def _load_artifact(deployment_root: Path, model_id: str) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    root = deployment_root / model_id
    model_path = root / "model.keras"
    manifest_path = root / "manifest.json"
    standardizer_path = root / "standardizer.json"
    for path in (model_path, manifest_path, standardizer_path):
        if not path.is_file():
            raise ForwardV3Error(f"frozen forward artifact missing for {model_id}: {path}")
    manifest = _read_json(manifest_path)
    standardizer = _read_json(standardizer_path)
    if str(manifest.get("model_id", model_id)) != model_id:
        raise ForwardV3Error(f"artifact model_id mismatch for {model_id}")
    return model_path, manifest, standardizer


def _build_live_features(
    *,
    spot_rows: list[dict[str, Any]],
    feature_set: str,
    micro_path: Path,
    futures_path: Path,
    eth_path: Path,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    base = build_feature_dataset(spot_rows, include_sentiment=False)
    context = build_context_feature_matrix(
        base.timestamps,
        base.closes,
        feature_set=feature_set,
        micro_path=micro_path,
        futures_path=futures_path,
        eth_path=eth_path,
    )
    if context.X.shape[1] == 0:
        return base.timestamps, base.X, list(base.feature_names)
    matrix = np.column_stack([base.X, context.X]).astype(np.float32, copy=False)
    return base.timestamps, matrix, list(base.feature_names) + list(context.feature_names)


def _input_window(
    *,
    timestamps: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
    manifest: dict[str, Any],
    standardizer: dict[str, Any],
    latest_hour: int,
) -> np.ndarray:
    candidate = manifest.get("candidate")
    if not isinstance(candidate, dict):
        raise ForwardV3Error("manifest candidate metadata is missing")
    sequence_length = int(candidate["sequence_length"])
    expected_names = [str(name) for name in manifest.get("feature_names", [])]
    if feature_names != expected_names:
        raise ForwardV3Error(
            f"feature contract mismatch: live={len(feature_names)} frozen={len(expected_names)}"
        )
    if len(timestamps) < sequence_length:
        raise ForwardV3Error("not enough live feature rows for the frozen sequence length")

    window_timestamps = np.asarray(timestamps[-sequence_length:], dtype=np.int64)
    if int(window_timestamps[-1]) != int(latest_hour):
        raise ForwardV3Error(
            f"latest feature timestamp {_iso(int(window_timestamps[-1]))} does not match latest completed hour {_iso(latest_hour)}"
        )
    if sequence_length > 1 and np.any(np.diff(window_timestamps) != HOUR_MS):
        raise ForwardV3Error("live inference sequence contains an hourly gap")

    for availability_name in ("micro_available", "futures_available", "eth_available"):
        if availability_name in feature_names:
            idx = feature_names.index(availability_name)
            if np.any(np.asarray(X[-sequence_length:, idx]) < 0.5):
                raise ForwardV3Error(f"{availability_name} is missing inside the live inference sequence")

    mean = np.asarray(standardizer.get("x_mean"), dtype=np.float64)
    scale = np.asarray(standardizer.get("x_scale"), dtype=np.float64)
    if mean.shape != (len(feature_names),) or scale.shape != (len(feature_names),):
        raise ForwardV3Error("standardizer shape does not match the frozen feature contract")
    if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ForwardV3Error("standardizer contains invalid values")

    window = np.asarray(X[-sequence_length:], dtype=np.float64)
    standardized = ((window - mean) / scale).astype(np.float32, copy=False)
    if np.any(~np.isfinite(standardized)):
        raise ForwardV3Error("standardized live model input contains non-finite values")
    return standardized[np.newaxis, :, :]


def _calibrated_probability(manifest: dict[str, Any], raw_probability: float) -> float:
    payload = manifest.get("calibrator")
    if not isinstance(payload, dict):
        raise ForwardV3Error("frozen calibrator is missing")
    calibrator = PlattCalibrator(
        slope=float(payload["slope"]),
        intercept=float(payload["intercept"]),
        clip_eps=float(payload.get("clip_eps", 1e-6)),
    )
    return float(calibrator.transform(np.asarray([raw_probability], dtype=np.float64))[0])


def _payoff(manifest: dict[str, Any]) -> PayoffEstimate:
    payload = manifest.get("payoff")
    if not isinstance(payload, dict):
        raise ForwardV3Error("frozen payoff estimate is missing")
    return PayoffEstimate(
        event_hurdle_bps=float(payload["event_hurdle_bps"]),
        trim_fraction=float(payload["trim_fraction"]),
        event_count=int(payload["event_count"]),
        non_event_count=int(payload["non_event_count"]),
        event_rate=float(payload["event_rate"]),
        mean_event_return=float(payload["mean_event_return"]),
        mean_non_event_return=float(payload["mean_non_event_return"]),
        median_event_return=float(payload["median_event_return"]),
        median_non_event_return=float(payload["median_non_event_return"]),
    )


def _load_runtime_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"models": {}}
    payload = _read_json(path)
    if not isinstance(payload.get("models"), dict):
        payload["models"] = {}
    return payload


def _paper_signal_for_policy(
    *,
    position_is_open: bool,
    calibrated_probability: float,
    payoff: PayoffEstimate,
    one_way_cost_rate: float,
    entry_margin: float,
) -> tuple[str, float, float]:
    gross_ev = float(payoff.expected_gross_return(np.asarray([calibrated_probability]))[0])
    if not position_is_open:
        decision_ev = gross_ev - 2.0 * one_way_cost_rate
        return ("BUY" if decision_ev > entry_margin else "HOLD", gross_ev, decision_ev)
    decision_ev = gross_ev
    return ("SELL" if decision_ev <= 0.0 else "HOLD", gross_ev, decision_ev)


def _strategy_version(model_id: str, manifest: dict[str, Any]) -> str:
    sha = str(manifest.get("model_sha256", "unknown"))[:12]
    return f"v3-frozen-forward:{model_id}:{sha}"


def _record_or_execute(
    *,
    store: ModelPaperStore,
    model_id: str,
    signal: str,
    confidence: float,
    reason: str,
    strategy_version: str,
    market_price: Decimal,
) -> dict[str, Any] | None:
    # This is deliberately separate from the generic percentage-cap risk manager.
    # The historical V3 strategy is binary 0/1 exposure. To preserve the frozen
    # research policy in PAPER mode, a BUY invests the available virtual cash
    # without leverage and a SELL closes the model's entire virtual BTC position.
    if settings.trading_mode.value != "paper":
        raise ForwardV3Error("frozen V3 forward execution is PAPER-only")

    if signal == "HOLD":
        store.record_decision(
            model_id,
            symbol="BTCUSDT",
            signal="HOLD",
            confidence=confidence,
            approved=True,
            reason=reason,
            strategy_version=strategy_version,
            market_price=market_price,
        )
        return None

    if signal == "BUY":
        if store.get_position(model_id, "BTCUSDT") is not None:
            raise ForwardV3Error(f"{model_id} requested BUY while already long")
        account = store.get_account(model_id)
        cash = Decimal(account["cash_usdt"])
        if cash <= 0:
            raise ForwardV3Error(f"{model_id} has no paper cash available")
        divisor = Decimal("1") + settings.paper_fee_rate
        notional = (cash / divisor).quantize(Decimal("0.00000001"), rounding=ROUND_DOWN)
        if notional <= 0:
            raise ForwardV3Error(f"{model_id} investable paper notional is zero")
        trade = store.buy(
            model_id,
            symbol="BTCUSDT",
            market_price=market_price,
            notional_usdt=notional,
            fee_rate=settings.paper_fee_rate,
            slippage_bps=settings.paper_slippage_bps,
            confidence=confidence,
            strategy_version=strategy_version,
        )
    elif signal == "SELL":
        position = store.get_position(model_id, "BTCUSDT")
        if position is None:
            raise ForwardV3Error(f"{model_id} requested SELL without a paper BTC position")
        trade = store.sell(
            model_id,
            symbol="BTCUSDT",
            market_price=market_price,
            quantity=None,
            fee_rate=settings.paper_fee_rate,
            slippage_bps=settings.paper_slippage_bps,
            confidence=confidence,
            strategy_version=strategy_version,
        )
    else:
        raise ForwardV3Error(f"unsupported frozen policy signal: {signal}")

    store.record_decision(
        model_id,
        symbol="BTCUSDT",
        signal=signal,
        confidence=confidence,
        approved=True,
        reason=reason,
        strategy_version=strategy_version,
        market_price=market_price,
    )
    return trade


async def run_forward_once(
    *,
    deployment_root: Path = DEFAULT_DEPLOYMENT_ROOT,
    state_root: Path = DEFAULT_STATE_ROOT,
    dry_run: bool = False,
    only_model_id: str | None = None,
) -> list[dict[str, Any]]:
    settings.assert_safe_mode()
    specs = list(PAPER_MODELS)
    if only_model_id is not None:
        specs = [get_paper_model(only_model_id)]

    artifacts: dict[str, tuple[Path, dict[str, Any], dict[str, Any]]] = {}
    max_sequence = 1
    for spec in specs:
        artifact = _load_artifact(deployment_root, spec.model_id)
        artifacts[spec.model_id] = artifact
        candidate = artifact[1].get("candidate")
        if not isinstance(candidate, dict):
            raise ForwardV3Error(f"candidate metadata missing for {spec.model_id}")
        max_sequence = max(max_sequence, int(candidate["sequence_length"]))

    latest_hour = _latest_completed_hour_ms()
    state_root.mkdir(parents=True, exist_ok=True)
    context_root = state_root / "context"
    micro_path = context_root / "btc_spot_aggtrades_hourly.jsonl"
    futures_path = context_root / "btc_um_futures_hourly.jsonl"
    eth_path = context_root / "eth_spot_hourly.jsonl"

    async with httpx.AsyncClient(timeout=180.0, follow_redirects=True) as client:
        btc_klines, eth_klines, futures_klines = await asyncio.gather(
            _spot_klines(client, "BTCUSDT", 1000, latest_hour),
            _spot_klines(client, "ETHUSDT", 1000, latest_hour),
            _futures_klines(client, "BTCUSDT", 1000, latest_hour),
        )
        await _ensure_micro_cache(
            client,
            cache_path=micro_path,
            latest_hour=latest_hour,
            required_hours=max_sequence + 48,
        )

    spot_rows = [_spot_training_row(row) for row in btc_klines]
    futures_rows = [_context_kline_row(row, "btc_um_futures_1h") for row in futures_klines]
    eth_rows = [_context_kline_row(row, "eth_spot_1h") for row in eth_klines]
    _write_context_rows(futures_path, futures_rows)
    _write_context_rows(eth_path, eth_rows)

    store = ModelPaperStore(settings.paper_db_path, settings.paper_model_initial_balance_eur_equiv)
    for spec in PAPER_MODELS:
        store.ensure_account(spec.model_id, spec.display_name)

    runtime_path = state_root / "runtime_state.json"
    runtime = _load_runtime_state(runtime_path)
    model_state = runtime.setdefault("models", {})
    predictions_path = state_root / "predictions.jsonl"
    market_client = CoinExMarketClient()
    quote = await market_client.get_ticker("BTCUSDT")
    one_way_cost_rate = float(settings.paper_fee_rate) + float(settings.paper_slippage_bps) / 10_000.0

    # TensorFlow is imported only after all input/artifact checks have passed so a
    # missing file or market-data outage does not pay the heavy import/startup cost.
    import tensorflow as tf

    results: list[dict[str, Any]] = []
    for spec in specs:
        model_path, manifest, standardizer = artifacts[spec.model_id]
        state = model_state.setdefault(spec.model_id, {})
        last_prediction = int(state.get("last_prediction_feature_timestamp", 0) or 0)
        if last_prediction >= latest_hour:
            results.append({"model_id": spec.model_id, "status": "already_processed", "feature_timestamp": latest_hour})
            continue

        timestamps, matrix, feature_names = _build_live_features(
            spot_rows=spot_rows,
            feature_set=spec.feature_set,
            micro_path=micro_path,
            futures_path=futures_path,
            eth_path=eth_path,
        )
        window = _input_window(
            timestamps=timestamps,
            X=matrix,
            feature_names=feature_names,
            manifest=manifest,
            standardizer=standardizer,
            latest_hour=latest_hour,
        )

        tf.keras.backend.clear_session()
        model = tf.keras.models.load_model(model_path, compile=False)
        outputs = model(window, training=False)
        if not isinstance(outputs, (list, tuple)) or len(outputs) < 2:
            raise ForwardV3Error(f"unexpected Keras output contract for {spec.model_id}")
        raw_probability = float(np.asarray(outputs[1]).reshape(-1)[0])
        if not math.isfinite(raw_probability):
            raise ForwardV3Error(f"non-finite raw probability for {spec.model_id}")
        raw_probability = float(np.clip(raw_probability, 1e-6, 1.0 - 1e-6))
        calibrated_probability = _calibrated_probability(manifest, raw_probability)
        payoff = _payoff(manifest)

        position_is_open = store.get_position(spec.model_id, "BTCUSDT") is not None
        horizon_hours = int(manifest.get("horizon_commitment_hours", spec.horizon_hours))
        last_policy = int(state.get("last_policy_feature_timestamp", 0) or 0)
        policy_due = last_policy <= 0 or latest_hour - last_policy >= horizon_hours * HOUR_MS
        entry_margin = float(manifest.get("entry_margin_bps", 0.0)) / 10_000.0

        gross_ev = float(payoff.expected_gross_return(np.asarray([calibrated_probability]))[0])
        signal = "COMMITMENT"
        decision_ev: float | None = None
        trade: dict[str, Any] | None = None
        if policy_due:
            signal, gross_ev, decision_ev = _paper_signal_for_policy(
                position_is_open=position_is_open,
                calibrated_probability=calibrated_probability,
                payoff=payoff,
                one_way_cost_rate=one_way_cost_rate,
                entry_margin=entry_margin,
            )
            reason = (
                f"Frozen V3 policy @ {_iso(latest_hour)}; p={calibrated_probability:.6f}; "
                f"gross_ev={gross_ev:+.6%}; decision_ev={decision_ev:+.6%}; "
                f"entry_margin={entry_margin:+.6%}; horizon={horizon_hours}h"
            )
            if not dry_run:
                trade = _record_or_execute(
                    store=store,
                    model_id=spec.model_id,
                    signal=signal,
                    confidence=calibrated_probability,
                    reason=reason,
                    strategy_version=_strategy_version(spec.model_id, manifest),
                    market_price=quote.last,
                )
                state["last_policy_feature_timestamp"] = latest_hour
        else:
            next_policy = last_policy + horizon_hours * HOUR_MS
            reason = f"Frozen horizon commitment active until {_iso(next_policy)}"

        record = {
            "recorded_at": datetime.now(UTC).isoformat(),
            "model_id": spec.model_id,
            "display_name": spec.display_name,
            "feature_timestamp": latest_hour,
            "feature_time_utc": _iso(latest_hour),
            "model_sha256": manifest.get("model_sha256"),
            "feature_version": manifest.get("feature_version"),
            "raw_probability": raw_probability,
            "calibrated_probability": calibrated_probability,
            "expected_gross_ev": gross_ev,
            "decision_ev": decision_ev,
            "entry_margin_bps": float(manifest.get("entry_margin_bps", 0.0)),
            "one_way_cost_rate": one_way_cost_rate,
            "horizon_commitment_hours": horizon_hours,
            "position_before": "LONG" if position_is_open else "CASH",
            "policy_due": policy_due,
            "signal": signal,
            "reason": reason,
            "paper_market_price": str(quote.last),
            "dry_run": dry_run,
            "trade": trade,
        }
        _append_jsonl(predictions_path, record)
        state["last_prediction_feature_timestamp"] = latest_hour
        state["last_probability"] = calibrated_probability
        state["last_signal"] = signal
        state["updated_at"] = record["recorded_at"]
        results.append(record)

    runtime["updated_at"] = datetime.now(UTC).isoformat()
    runtime["latest_completed_hour"] = latest_hour
    if not dry_run:
        _write_json(runtime_path, runtime)
    return results
