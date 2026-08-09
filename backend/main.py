from fastapi import FastAPI, HTTPException

from backend.coinex import CoinExAPIError, CoinExClient
from backend.config import settings


app = FastAPI(title=settings.app_name, version="0.2.0")


def _coinex_client() -> CoinExClient:
    if not settings.coinex_configured:
        raise HTTPException(
            status_code=503,
            detail="CoinEx is not configured. Add credentials to the local .env file.",
        )

    return CoinExClient(
        access_id=settings.coinex_access_id or "",
        secret_key=settings.coinex_secret_key or "",
    )


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "environment": settings.app_env,
        "trading_mode": settings.trading_mode.value,
        "coinex_configured": settings.coinex_configured,
    }


@app.get("/api/coinex/debug")
async def coinex_debug() -> dict[str, object]:
    """Return safe diagnostics for the CoinEx Spot balance response."""
    client = _coinex_client()

    try:
        debug = await client.get_spot_balance_debug()
    except CoinExAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {
        "exchange": "coinex",
        "account": "spot",
        "read_only": True,
        **debug,
    }


@app.get("/api/coinex/balances")
async def coinex_balances() -> dict[str, object]:
    """Return non-zero CoinEx spot balances. This endpoint cannot place orders."""
    client = _coinex_client()

    try:
        balances = await client.get_spot_balances()
    except CoinExAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {
        "exchange": "coinex",
        "account": "spot",
        "read_only": True,
        "balances": [balance.as_dict() for balance in balances],
    }
