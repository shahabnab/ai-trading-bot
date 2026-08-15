"""Regime-aware trader-style decision system for paper trading research."""

from .contracts import ExpertForecast, GateResult, RegimePosterior, TraderDecision
from .decision import DecisionConfig, decide
from .runtime import run_trader_brain_once

__all__ = [
    "DecisionConfig",
    "ExpertForecast",
    "GateResult",
    "RegimePosterior",
    "TraderDecision",
    "decide",
    "run_trader_brain_once",
]
