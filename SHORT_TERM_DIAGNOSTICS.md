# Short-Term Decision Diagnostics

This layer is observational. It does **not** change the official short-term entry threshold, does not create paper fills, and cannot place real orders.

## Why it exists

The 15-minute strategies can remain on `HOLD` for two very different reasons:

1. the setup logic is not finding a credible opportunity, or
2. the setup is credible but its edge proxy does not clear the economic hurdle.

The diagnostics separate those cases and then score what happened after the decision.

## Decision telemetry

Each newly processed 15-minute bucket appends one row per short-term model to:

- `state/short_term/decision_diagnostics.jsonl`

The row includes:

- setup/checklist score (`confirmation_score`),
- existing risk-facing score (`confidence`),
- whether the setup was ready before the edge hurdle,
- edge proxy in basis points,
- official threshold in basis points,
- edge gap (`edge_proxy - official_threshold`),
- configured round-trip cost,
- official action and execution result,
- non-executing shadow-policy decisions.

The existing `confidence` value is a transformed checklist score. It must not be interpreted as a calibrated probability that the trade will be profitable.

## Shadow policies

The official policy stays untouched. The same setup is also evaluated against:

- official dynamic hurdle (`round_trip_cost + 15 bps`),
- 55 bps,
- 45 bps,
- 35 bps,
- raw setup (no edge hurdle).

These shadow policies never call the broker and never alter the paper ledger.

## Outcome analyzer

Once the full two-hour future window exists, the runtime appends an outcome to:

- `state/short_term/decision_outcomes.jsonl`

It records close-to-close returns after 15m, 30m, 1h, and 2h, plus two-hour maximum favorable/adverse excursion.

For rejected long setups the main labels are:

- `MISSED_LONG`: setup was ready, official entry was rejected, and the 2h close later cleared the official hurdle upward,
- `AVOIDED_LOSS`: setup was ready, official entry was rejected, and the 2h close later moved downward by at least the official hurdle,
- `GOOD_HOLD`: setup was ready but neither of the above occurred,
- `NO_SETUP`: the strategy checklist/microstructure gate itself was not ready,
- `POSITION_MANAGEMENT`: a paper position was already open, so the row is not an entry-opportunity evaluation,
- `ENTRY_SIGNAL`: the official strategy produced an entry signal.

The short-term baselines are currently long-only, so this diagnostic intentionally does not claim `MISSED_SHORT` events.

## Shadow return convention

For diagnosis only, a shadow entry is scored as:

`2h close-to-close return - configured round-trip cost`

This is **not** the official paper P/L and does not reproduce the strategy's real exit rules. It is a controlled way to compare entry hurdles while keeping the raw setup identical.

## Dashboard

The dashboard reads the diagnostic JSONL files through `/api/short-term-diagnostics` and shows:

- resolved vs pending decisions,
- setup candidates,
- missed-long and avoided-loss counts,
- average edge gap,
- 65/55/45/35/raw shadow-policy comparisons,
- setup-score buckets versus subsequent 2h direction.

The global `CURRENT LEADER` card now requires at least 10 closed forward trades before naming a leader.

## CLI report

```bash
python scripts/report_short_term_diagnostics.py
```

Optional:

```bash
python scripts/report_short_term_diagnostics.py --limit 5000
```

## Historical replay limitation

A faithful six-month replay of the current short-term strategies requires point-in-time historical taker-flow/order-book microstructure. The repository currently has live microstructure collection, but it does not contain a six-month historical microstructure archive. We therefore do not silently substitute missing microstructure or relax the safety gate for backtesting. Doing so would make the replay incomparable to the live strategy.
