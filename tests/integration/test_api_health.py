import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from hm_recsys.api.main import app


@pytest.mark.integration
def test_health_endpoint() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
