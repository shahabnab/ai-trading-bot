"""Data collection utilities for training and research datasets.

Keep exchange/account access in the existing backend.coinex/backend.market
packages. This package coordinates collection and persistence of historical
market and text data used by ML experiments.
"""

from .storage import append_unique_jsonl

__all__ = ["append_unique_jsonl"]
