"""Cobertura enfocada de la API HTTP (tarea 17.3).

Usa `fastapi.testclient.TestClient`, que ejecuta el pipeline completamente en
proceso (sin red real): por defecto (sin `USE_REAL_LLM`/`USE_REAL_PROPENSION`) el
pipeline sólo usa componentes fake/en memoria, igual que `run_demo.py`.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from sistema_agentico.api.main import app

client = TestClient(app)

# Ids sintéticos por defecto de los fakes (`with_default_synthetic_*`, seed=0).
_ID_OBLIGACION = "SYN-OBL-00000"

# Claves que jamás deben aparecer en una respuesta HTTP (PII o detalle interno).
_FORBIDDEN_KEYS = {
    "id_cliente",
    "contexto",
    "escalation_case",
    "trace_completo",
    "reglas_evaluadas",
    "eligibility_result",
    "whitelist",
    "score_info",
    "version_modelo_propension",
}


def _assert_no_forbidden_keys(payload: object) -> None:
    """Recorre recursivamente el JSON de respuesta buscando claves prohibidas."""
    if isinstance(payload, dict):
        present = _FORBIDDEN_KEYS & payload.keys()
        assert not present, f"claves prohibidas encontradas en la respuesta: {present}"
        for value in payload.values():
            _assert_no_forbidden_keys(value)
    elif isinstance(payload, list):
        for item in payload:
            _assert_no_forbidden_keys(item)


def test_health_check() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "timestamp" in body


def test_reactiva_happy_path_returns_final_or_escalated_outcome() -> None:
    response = client.post(
        "/interacciones/reactiva",
        json={
            "id_obligacion": _ID_OBLIGACION,
            "mensaje_cliente": "Hola, quisiera saber qué opciones tengo para pagar mi deuda",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] in ("final", "escalated")
    assert "trace_id" in body and body["trace_id"]
    assert isinstance(body["trace_logged"], bool)
    if body["status"] == "final":
        assert body["respuesta"] is not None
        assert body["escalation_reason"] is None
        assert set(body["respuesta"].keys()) == {"texto", "ofertas_mencionadas", "prompt_version"}
    else:
        assert body["respuesta"] is None
        assert body["escalation_reason"] is not None
    _assert_no_forbidden_keys(body)


def test_reactiva_rejects_empty_id_obligacion_and_mensaje_cliente_with_422() -> None:
    response = client.post(
        "/interacciones/reactiva",
        json={"id_obligacion": "", "mensaje_cliente": "hola"},
    )
    assert response.status_code == 422

    response = client.post(
        "/interacciones/reactiva",
        json={"id_obligacion": _ID_OBLIGACION, "mensaje_cliente": ""},
    )
    assert response.status_code == 422


def test_proactiva_happy_path_reorders_by_score_and_preserves_shape() -> None:
    # Scores sintéticos reales (`FakePropensionClient.with_default_synthetic_scores`,
    # n=20, seed=0) confirmados empíricamente: SYN-OBL-00001=0.7994, SYN-OBL-00002=0.3302,
    # SYN-OBL-00003=0.805. Orden esperado de mayor a menor score: 00003, 00001, 00002.
    ids = ["SYN-OBL-00001", "SYN-OBL-00002", "SYN-OBL-00003"]
    expected_order = ["SYN-OBL-00003", "SYN-OBL-00001", "SYN-OBL-00002"]
    response = client.post("/interacciones/proactiva", json={"ids_obligacion": ids})

    assert response.status_code == 200
    body = response.json()
    assert [item["id_obligacion"] for item in body] == expected_order
    assert {item["id_obligacion"] for item in body} == set(ids)
    for item in body:
        assert isinstance(item["completado"], bool)
        if item["completado"]:
            assert item["resultado"] is not None
            assert item["error"] is None
        else:
            assert item["resultado"] is None
            assert item["error"] is not None
    _assert_no_forbidden_keys(body)


def test_responses_never_leak_id_cliente_or_internal_context() -> None:
    reactiva_response = client.post(
        "/interacciones/reactiva",
        json={"id_obligacion": _ID_OBLIGACION, "mensaje_cliente": "Necesito ayuda con mi deuda"},
    )
    assert reactiva_response.status_code == 200
    body_text = reactiva_response.text
    assert "id_cliente" not in body_text
    assert "escalation_case" not in body_text

    proactiva_response = client.post(
        "/interacciones/proactiva", json={"ids_obligacion": ["SYN-OBL-00004"]}
    )
    assert proactiva_response.status_code == 200
    assert "id_cliente" not in proactiva_response.text
    assert "escalation_case" not in proactiva_response.text
