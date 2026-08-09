from fastapi import FastAPI

from backend.config import settings


app = FastAPI(title=settings.app_name, version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "environment": settings.app_env,
        "trading_mode": settings.trading_mode.value,
    }
