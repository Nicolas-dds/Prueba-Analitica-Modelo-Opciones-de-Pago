"""Pruebas de `HttpPropensionClient` contra un stub HTTP real (tarea 5.1).

Ver Component 3 de `design.md` y Requirements 3.1, 14.3 de `requirements.md`.

Usa un servidor HTTP mínimo (`http.server`) en un hilo de fondo para ejercitar el
cliente sin depender de que el servicio real `modelo-propension` esté corriendo,
cubriendo los casos: 200 (score real), 404 (`ScoreNotFoundError`), 500
(`PropensionServiceUnavailableError`) y timeout (`PropensionServiceUnavailableError`).
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from sistema_agentico.context.propension_client import (
    HttpPropensionClient,
    PropensionServiceUnavailableError,
    ScoreNotFoundError,
)
from sistema_agentico.types import ScoreInfo

VALID_SCORE_PAYLOAD = {
    "id_obligacion": "OBL-1",
    "score": 0.73,
    "decil": 8,
    "version_modelo": "v20260913073248",
    "fecha_calificacion": "2026-09-30",
}


class _StubHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 (nombre impuesto por BaseHTTPRequestHandler)
        if self.path == "/score/OBL-1" or self.path == "/score/257335%23444821%23635511":
            body = json.dumps(VALID_SCORE_PAYLOAD).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/score/OBL-404":
            body = json.dumps({"detail": "sin score"}).encode("utf-8")
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/score/OBL-500":
            body = json.dumps({"detail": "error interno"}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/score/OBL-LENTO":
            time.sleep(1.0)
            self.send_response(200)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass  # silenciar logs del servidor de pruebas


@pytest.fixture(scope="module")
def stub_server() -> str:
    server = HTTPServer(("127.0.0.1", 0), _StubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    yield base_url
    server.shutdown()
    thread.join(timeout=5)


def test_get_score_returns_score_info_no_degradado(stub_server: str) -> None:
    client = HttpPropensionClient(base_url=stub_server, default_timeout_seconds=2.0)

    result = client.get_score("OBL-1", timeout_seconds=2.0)

    assert result == ScoreInfo(
        score=0.73,
        decil=8,
        version_modelo="v20260913073248",
        fecha_calificacion="2026-09-30",
        degradado=False,
    )


def test_get_score_404_raises_score_not_found_error(stub_server: str) -> None:
    client = HttpPropensionClient(base_url=stub_server, default_timeout_seconds=2.0)

    with pytest.raises(ScoreNotFoundError):
        client.get_score("OBL-404", timeout_seconds=2.0)


def test_get_score_5xx_raises_service_unavailable_error(stub_server: str) -> None:
    client = HttpPropensionClient(base_url=stub_server, default_timeout_seconds=2.0)

    with pytest.raises(PropensionServiceUnavailableError):
        client.get_score("OBL-500", timeout_seconds=2.0)


def test_get_score_timeout_raises_service_unavailable_error(stub_server: str) -> None:
    client = HttpPropensionClient(base_url=stub_server, default_timeout_seconds=2.0)

    with pytest.raises(PropensionServiceUnavailableError):
        client.get_score("OBL-LENTO", timeout_seconds=0.1)


def test_get_score_connection_error_raises_service_unavailable_error() -> None:
    # Puerto sin servidor escuchando -> ConnectionError / URLError.
    client = HttpPropensionClient(base_url="http://127.0.0.1:1", default_timeout_seconds=0.5)

    with pytest.raises(PropensionServiceUnavailableError):
        client.get_score("OBL-1", timeout_seconds=0.5)


def test_default_timeout_is_used_when_not_specified(stub_server: str) -> None:
    client = HttpPropensionClient(base_url=stub_server, default_timeout_seconds=2.0)

    result = client.get_score("OBL-1")

    assert result.degradado is False


def test_get_score_url_encodes_reserved_characters_in_id_obligacion(stub_server: str) -> None:
    # `id_obligacion` real (ver data_output/scores/*.parquet de modelo-propension) puede
    # contener '#', que sin URL-encoding se interpreta como fragmento y trunca el path.
    client = HttpPropensionClient(base_url=stub_server, default_timeout_seconds=2.0)

    result = client.get_score("257335#444821#635511")

    assert result.degradado is False
