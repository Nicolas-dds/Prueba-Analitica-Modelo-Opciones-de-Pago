"""Focused unit tests for OutputGuardrail whitelist validation (Requirement 8.1)."""
from sistema_agentico.guardrails import OutputGuardrail
from sistema_agentico.types import AgentResponse, OfertaElegible, TipoOferta, WhitelistItem


def _whitelist(*ids: str) -> list[WhitelistItem]:
    return [WhitelistItem(OfertaElegible(TipoOferta.OPCION_PAGO, id_, {}, "Elegible."), i, "Autorizada.") for i, id_ in enumerate(ids, 1)]


def _response(*ids: str) -> AgentResponse:
    return AgentResponse("Respuesta sintética", list(ids), False, "test-v1")


def test_accepts_exact_subset_and_empty_mentions() -> None:
    guardrail = OutputGuardrail()
    whitelist = _whitelist("ampliacion_plazo", "reduccion_cuota")
    for response in (_response("reduccion_cuota"), _response()):
        result = guardrail.validate(response, whitelist, contexto=None)  # type: ignore[arg-type]
        assert result.valido is True
        assert result.motivo_rechazo is None
        assert result.escalar is False
        assert result.razon_escalamiento is None


def test_rejects_all_distinct_unauthorized_metadata_ids() -> None:
    result = OutputGuardrail().validate(
        _response("ampliacion_plazo", "alivio_especial", "oferta_no_existente", "alivio_especial"),
        _whitelist("ampliacion_plazo"), contexto=None,  # type: ignore[arg-type]
    )
    assert result.valido is False
    assert result.motivo_rechazo == "Ofertas fuera de la whitelist autorizada: alivio_especial, oferta_no_existente"
    assert result.escalar is False
    assert result.razon_escalamiento is None


def test_rejects_any_mention_against_an_empty_whitelist() -> None:
    result = OutputGuardrail().validate(_response("ampliacion_plazo"), _whitelist(), contexto=None)  # type: ignore[arg-type]
    assert result.valido is False
    assert result.motivo_rechazo
