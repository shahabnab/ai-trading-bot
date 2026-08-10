from decimal import Decimal

import pytest

from backend.config import Settings


def test_exposure_limit_order_is_validated() -> None:
    settings = Settings(
        PAPER_MAX_ORDER_FRACTION=Decimal("0.30"),
        PAPER_MAX_SYMBOL_EXPOSURE_FRACTION=Decimal("0.20"),
        PAPER_MAX_TOTAL_EXPOSURE_FRACTION=Decimal("0.50"),
    )

    with pytest.raises(RuntimeError, match="order <= symbol <= total"):
        settings.assert_safe_mode()
