from __future__ import annotations

import asyncio
import json
from collections import deque
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from time import time
from typing import Any

from backend.market import CoinExMarketClient, MarketDataError

BUCKET_MS = 15 * 60 * 1000
DEFAULT_STATE_ROOT = Path("state/short_term")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")


def _decimal(value: object) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")


@dataclass
class BucketAccumulator:
    bucket_start: int
    first_observed_ms: int
    last_observed_ms: int
    trade_count: int = 0
    buy_notional: Decimal = Decimal("0")
    sell_notional: Decimal = Decimal("0")
    buy_amount: Decimal = Decimal("0")
    sell_amount: Decimal = Decimal("0")
    spread_sum_bps: float = 0.0
    book_imbalance_sum: float = 0.0
    depth_snapshots: int = 0

    def add_deal(self, deal: dict[str, object]) -> None:
        notional = _decimal(deal.get("notional"))
        amount = _decimal(deal.get("amount"))
        if str(deal.get("side", "")).lower() == "buy":
            self.buy_notional += notional
            self.buy_amount += amount
        else:
            self.sell_notional += notional
            self.sell_amount += amount
        self.trade_count += 1

    def add_depth(self, depth: dict[str, object], observed_ms: int) -> None:
        bids = depth.get("bids") if isinstance(depth.get("bids"), list) else []
        asks = depth.get("asks") if isinstance(depth.get("asks"), list) else []
        if not bids or not asks:
            return
        best_bid = _decimal(bids[0].get("price") if isinstance(bids[0], dict) else 0)
        best_ask = _decimal(asks[0].get("price") if isinstance(asks[0], dict) else 0)
        if best_bid <= 0 or best_ask <= 0 or best_ask < best_bid:
            return
        mid = (best_bid + best_ask) / Decimal("2")
        spread_bps = float((best_ask - best_bid) / mid * Decimal("10000")) if mid > 0 else 0.0
        bid_notional = sum(
            (_decimal(level.get("price")) * _decimal(level.get("amount")))
            for level in bids if isinstance(level, dict)
        )
        ask_notional = sum(
            (_decimal(level.get("price")) * _decimal(level.get("amount")))
            for level in asks if isinstance(level, dict)
        )
        total = bid_notional + ask_notional
        imbalance = float((bid_notional - ask_notional) / total) if total > 0 else 0.0
        self.spread_sum_bps += spread_bps
        self.book_imbalance_sum += max(-1.0, min(1.0, imbalance))
        self.depth_snapshots += 1
        self.last_observed_ms = max(self.last_observed_ms, observed_ms)

    def to_row(self) -> dict[str, object]:
        total = self.buy_notional + self.sell_notional
        buy_ratio = float(self.buy_notional / total) if total > 0 else 0.5
        imbalance = float((self.buy_notional - self.sell_notional) / total) if total > 0 else 0.0
        observed_span = max(0, min(BUCKET_MS, self.last_observed_ms - self.first_observed_ms))
        coverage = min(1.0, observed_span / BUCKET_MS)
        return {
            "bucket_start": self.bucket_start,
            "bucket_end": self.bucket_start + BUCKET_MS,
            "trade_count": self.trade_count,
            "buy_notional": str(self.buy_notional),
            "sell_notional": str(self.sell_notional),
            "buy_amount": str(self.buy_amount),
            "sell_amount": str(self.sell_amount),
            "buy_ratio": buy_ratio,
            "trade_imbalance": imbalance,
            "spread_bps": self.spread_sum_bps / self.depth_snapshots if self.depth_snapshots else None,
            "book_imbalance": self.book_imbalance_sum / self.depth_snapshots if self.depth_snapshots else None,
            "depth_snapshots": self.depth_snapshots,
            "first_observed_ms": self.first_observed_ms,
            "last_observed_ms": self.last_observed_ms,
            "coverage_fraction": coverage,
        }


def accumulator_for_timestamp(
    timestamp_ms: int,
    current: BucketAccumulator,
    previous: BucketAccumulator | None,
) -> BucketAccumulator | None:
    """Return the live accumulator that owns a deal timestamp."""
    if current.bucket_start <= timestamp_ms < current.bucket_start + BUCKET_MS:
        return current
    if previous is not None and previous.bucket_start <= timestamp_ms < previous.bucket_start + BUCKET_MS:
        return previous
    return None


async def run_short_term_collector(
    *,
    symbol: str = "BTCUSDT",
    state_root: Path = DEFAULT_STATE_ROOT,
    poll_seconds: float = 5.0,
) -> None:
    """Continuously aggregate public CoinEx trades/depth into 15-minute rows.

    Recent trade requests overlap heavily, so deal IDs are deduplicated in
    memory. At a bucket rollover the previous accumulator remains alive for one
    additional poll. This captures deals from the final few seconds that CoinEx
    may return only after the wall clock has crossed into the next bucket.
    """
    market = CoinExMarketClient()
    output = state_root / "microstructure.jsonl"
    seen_order: deque[int] = deque()
    seen: set[int] = set()
    max_seen = 50_000
    current: BucketAccumulator | None = None
    previous: BucketAccumulator | None = None
    previous_grace_complete = False

    while True:
        # Flush a rolled bucket only after it has survived one extra fetch cycle.
        if previous is not None and previous_grace_complete:
            _append_jsonl(output, previous.to_row())
            previous = None
            previous_grace_complete = False

        now_ms = int(time() * 1000)
        bucket_start = now_ms // BUCKET_MS * BUCKET_MS
        if current is None:
            current = BucketAccumulator(bucket_start, now_ms, now_ms)
        elif bucket_start != current.bucket_start:
            # Normally previous has already been flushed at the top of the loop.
            # The defensive flush avoids losing it if a very long poll spans more
            # than one boundary.
            if previous is not None:
                _append_jsonl(output, previous.to_row())
            previous = current
            current = BucketAccumulator(bucket_start, now_ms, now_ms)
            previous_grace_complete = False

        try:
            deals, depth = await asyncio.gather(
                market.get_deals(symbol, limit=1000),
                market.get_depth(symbol, limit=20, interval="0.01"),
            )
            for deal in deals:
                deal_id = int(deal.get("deal_id", 0) or 0)
                ts = int(deal.get("created_at", 0) or 0)
                if deal_id <= 0 or deal_id in seen:
                    continue

                # Assign before permanently marking the ID as seen. On rollover,
                # a late previous-bucket deal can therefore still reach the old
                # accumulator rather than being silently discarded.
                target = accumulator_for_timestamp(ts, current, previous)
                if target is not None:
                    target.add_deal(deal)
                    target.last_observed_ms = max(target.last_observed_ms, ts)

                if len(seen_order) >= max_seen:
                    old = seen_order.popleft()
                    seen.discard(old)
                seen_order.append(deal_id)
                seen.add(deal_id)

            current.add_depth(depth, now_ms)
        except (MarketDataError, OSError, ValueError):
            # The service is supervised by systemd; a temporary public-data
            # outage should reduce coverage rather than kill the collector.
            pass

        if previous is not None:
            previous_grace_complete = True

        await asyncio.sleep(max(1.0, poll_seconds))
