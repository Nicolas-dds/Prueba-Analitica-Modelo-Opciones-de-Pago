"""Pruebas de contrato de la API (RNF-09): rutas registradas y codigos de respuesta esperados.

Usa un registry aislado en `tmp_path` (via monkeypatch de get_config) en vez
del registry real del repo, para que estas pruebas de contrato no dependan
de si ya se entreno o promovio un modelo campeon (evita acoplar tests
estructurales al estado mutable de `models/registry/`, RNF-14).
"""
import pytest
from fastapi.testclient import TestClient

from api import dependencies
from api.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _registry_vacio(tmp_path, monkeypatch):
    """Fuerza a la API a ver un registry vacio (sin campeon) en cada test, sin importar el estado real del repo."""
    config = {
        "data": {"scores_path": str(tmp_path / "scores")},
        "registry": {"path": str(tmp_path / "registry"), "promotion_metric": "f1", "min_improvement": 0.0},
    }
    monkeypatch.setattr(dependencies, "get_config", lambda: config)
    monkeypatch.setattr("api.main.get_config", lambda: config)
    yield


def test_health_ok() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info_sin_modelo_registrado() -> None:
    response = client.get("/model/info")
    assert response.status_code == 200
    assert response.json()["version"] == "sin-modelo-registrado"


def test_score_sin_batch_calificado_devuelve_404() -> None:
    response = client.get("/score/OB-0001")
    assert response.status_code == 404


def test_score_batch_sin_modelo_campeon_devuelve_501() -> None:
    response = client.post("/score/batch", params={"month": "2026-09"})
    assert response.status_code == 501


def test_promote_version_inexistente_devuelve_404() -> None:
    response = client.post("/model/promote", params={"challenger_version": "v-inexistente"})
    assert response.status_code == 404
