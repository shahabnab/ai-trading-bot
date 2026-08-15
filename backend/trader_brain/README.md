# Trader-Brain V1

This package implements the first production-safe stage of the research design **Building a Trader-Style, Regime-Aware Mixture-of-Experts Trading System**. It augments the existing frozen models rather than replacing them.

## Decision hierarchy

```text
point-in-time market data
  -> T1-T10 trader-style technical concepts
  -> causal probabilistic regime posterior
  -> Technical / Macro / Derivatives / News / Existing-AI experts
  -> trailing OOS reliability
  -> XGBoost stacking after enough resolved OOS samples
  -> cost + uncertainty + regime-entropy decision
  -> LONG / SHORT / NO_TRADE forecast intent
  -> spot PAPER translation (LONG / EXIT / NO_TRADE)
  -> existing hard RiskManager
  -> simulated execution
  -> state/action/outcome + shadow-reward learning log
  -> optional contextual-bandit PAPER policy
```

The current paper broker is spot/long-only. A bearish Trader-Brain forecast therefore exits an existing BTC paper position or stays flat. It **never fabricates a short fill**. A future perpetual-paper broker can add real SHORT execution without changing the expert/gate contracts.

## Regimes

`PersistentGaussianRegimeDetector` uses GMM emissions plus an empirical Markov transition matrix and exposes forward-filtered probabilities only. There is deliberately no full-history smoother in the live interface. State IDs are canonicalized by the fitted BTC-24h-return centroid to reduce label permutation across refits; they remain machine states, not hard-coded economic labels.

## Experts

All experts share `ExpertForecast` and return `p_down/p_flat/p_up`, expected return, uncertainty, confidence, quality, timestamp, signals and explanations.

- Technical: T1-T10 from hourly BTC OHLCV.
- Macro: M1-M9 when point-in-time data exists. Gold direction is conditional on M8 BTC/gold correlation; no permanent `gold up => BTC up` rule exists.
- Derivatives: D1-D10 when available. Funding is interpreted conditionally with leverage/order-flow context, not as an automatic contrarian signal.
- News: N1-N7 when a timestamp-valid event snapshot exists. Missing news is masked, not zero-filled as if neutral/current.
- Existing AI: recent frozen V3 predictions can be wrapped as specialists.

## Optional external context

A collector may write `state/trader_brain/external_context.json`:

```json
{
  "timestamp": 1786802400000,
  "macro": {"M1": 0.42, "M2": -0.18, "M3": 0.31, "M8": 0.15},
  "derivatives": {"D1": 0.22, "D4": 0.08, "D6": 0.31, "D8": 0.12},
  "news": {"N1": 0.25, "N3": 0.70, "N4": 0.22, "N5": 0.18}
}
```

The whole snapshot is rejected when its timestamp is in the future or more than six hours old. Missing domains lower expert quality/availability. Real point-in-time collectors for gold, NQ/ES, DXY, VIX, yields, funding, OI, liquidations, options and structured news should be added incrementally; the runtime must never invent those inputs.

## Learning without leakage

Every hourly decision creates a `trader_brain_experiences` row containing the exact regime posterior, expert forecasts, gate vector, portfolio state, action and cost assumption available at decision time. The row resolves only when the exact target timestamp price is available.

The first learned meta-gate is XGBoost and trains only after the configured number of **resolved forward/OOS** samples. It never trains on an expert's in-sample predictions.

The RL candidate is deliberately a LinUCB contextual bandit rather than unrestricted deep RL. During warm-up it follows the supervised MoE. Once enough resolved observations exist it learns from both executed-deal and shadow/counterfactual rewards. The hard `RiskManager` stays outside the learner and cannot be overridden.

## Research sequence

1. Collect forward Trader-Brain experiences.
2. Compare reliability-weighted MoE against frozen V3 controls.
3. Activate/assess XGBoost stacking after sufficient OOS history.
4. Compare the contextual-bandit account independently.
5. Add real point-in-time macro/derivatives/news collectors.
6. Run chronological historical walk-forward ablations with fees, slippage, delay and stress tests before any consideration of real capital.

No component in this package is evidence of profitability by itself.
