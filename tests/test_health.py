from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_health_is_paper_mode() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["trading_mode"] == "paper"
