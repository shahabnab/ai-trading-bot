# Claude Code Instructions

Read `PROJECT.md` before making material changes.

## Non-negotiable rules

- The current mode is **paper trading only**.
- Never add a default that enables live trading.
- Never commit credentials, tokens, private keys, account identifiers, or `.env`.
- A strategy/model/LLM must not call a broker execution method directly. Route orders through the risk layer.
- Avoid look-ahead leakage in all time-series code.
- Do not randomly shuffle chronological trading data for train/test evaluation.
- Backtests must make transaction costs/slippage configurable.
- Do not present in-sample metrics as evidence of trading performance.

## Engineering conventions

- Python 3.11+.
- Prefer typed, small modules with clear boundaries.
- Add tests with behavioral changes.
- Keep notebooks for exploration only; move reusable logic into package modules.
- Use environment variables for secrets/configuration.
- Fail closed: if risk/config validation fails, do not submit an order.
- Make external/broker calls idempotent where possible.
- Use UTC internally for timestamps; convert only for presentation.
- Preserve raw market data; derive features separately.

## Collaboration

`PROJECT.md` is the shared specification used by the owner, ChatGPT, and Claude. If an implementation changes architecture, assumptions, or project phase, update `PROJECT.md` explicitly rather than allowing documentation to drift.

Prefer small commits whose message says what behavior changed. Do not rewrite unrelated files.
