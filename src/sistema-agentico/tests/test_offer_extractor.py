"""Pruebas de regresión del extractor fail-closed de ofertas (tarea 9.4)."""
from __future__ import annotations

import pytest

from sistema_agentico.conversational import (
    ConversationalAgent,
    DeterministicFakeChatModel,
    OfferMentionExtractionError,
    extract_offers_mentioned,
)
from sistema_agentico.types import (
    ClienteObligacionContext,
    OfertaElegible,
    ScoreInfo,
    TipoOferta,
    WhitelistItem,
)


def _whitelist(*offer_ids: str) -> list[WhitelistItem]:
    return [
        WhitelistItem(
            oferta=OfertaElegible(
                tipo=TipoOferta.OPCION_PAGO,
                id_opcion=offer_id,
                detalle={},
                razon_elegibilidad="Oferta aprobada para la prueba.",
            ),
            prioridad=index,
            justificacion="Es una alternativa autorizada.",
        )
        for index, offer_id in enumerate(offer_ids, start=1)
    ]


def _contexto() -> ClienteObligacionContext:
    return ClienteObligacionContext(
        id_obligacion="obligacion-sintetica",
        id_cliente="cliente-sintetico",
        dias_mora=10,
        exposicion=1000.0,
        segmento=None,
        canal_gestion=None,
        score=ScoreInfo(0.5, 5, "test", "2026-01-01", False),
        ofertas_aplicadas_mes=[],
        ultima_opcion_aplicada=None,
        acuerdo_pago_vigente=False,
        restriccion_vigente=False,
    )


def test_extracts_unique_exact_ids_in_order_of_first_text_appearance() -> None:
    whitelist = _whitelist("ampliacion_plazo", "reduccion_cuota")

    mentioned = extract_offers_mentioned(
        "Primero reduccion_cuota; luego ampliacion_plazo y otra vez reduccion_cuota.",
        whitelist,
    )

    assert mentioned == ["reduccion_cuota", "ampliacion_plazo"]


def test_allows_a_generic_reply_without_any_offer() -> None:
    assert extract_offers_mentioned(
        "Gracias por su mensaje. Un gestor puede orientarle.", _whitelist("ampliacion_plazo")
    ) == []


@pytest.mark.parametrize(
    ("texto", "reason"),
    [
        ("La opción es alivio_especial.", "ID de oferta desconocido"),
        ("La opción es acuerdo_pago_premium.", "familia canónica no autorizado"),
        ("La opción es ampliacion_plazo_temporal.", "referencia parcial"),
        ("Podemos revisar una ampliación de plazo.", "descrita sin su ID exacto"),
    ],
)
def test_rejects_unreliable_offer_references(texto: str, reason: str) -> None:
    with pytest.raises(OfferMentionExtractionError, match=reason):
        extract_offers_mentioned(texto, _whitelist("ampliacion_plazo"))


def test_agent_marks_unreliable_extraction_for_retry_instead_of_raising() -> None:
    """Regression for task 18.1: an unreliable extraction must not propagate.

    ``OfferMentionExtractionError`` used to escape ``ConversationalAgent.generate``
    uncaught, which made ``Orchestrator`` treat it as a technical failure
    (``FALLO_TECNICO``) instead of giving it the existing single-retry-via-
    correction path designed for unauthorized/unreliable output.
    """
    chat_model = DeterministicFakeChatModel(
        response="La alternativa disponible es alivio_especial."
    )
    agent = ConversationalAgent(chat_model=chat_model, prompt_version="test-v1")

    response = agent.generate(None, _whitelist("ampliacion_plazo"), _contexto(), [])

    assert response.requiere_reintento is True
    assert response.ofertas_mencionadas == []
    assert response.prompt_version == "test-v1"
    assert chat_model.call_count == 1
