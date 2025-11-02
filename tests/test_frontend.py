from pathlib import Path
import sys

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from service.dspy_sidecar import app


def test_root_serves_index_html():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "Idea Validator Console" in response.text
    assert "Run Full Analysis" in response.text


def test_static_assets_available():
    client = TestClient(app)
    response = client.get("/static/app.js")
    assert response.status_code == 200
    assert "runAnalysis" in response.text
