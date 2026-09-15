"""Unit coverage for multi-turn alternatives from an existing whitelist."""
from __future__ import annotations

import importlib

from sistema_agentico.orchestration import AlternativeSessionController
from sistema_agentico.types import OfertaElegible, TipoOferta, WhitelistItem


def _whitelist() -> list[WhitelistItem]:
    return [
        WhitelistItem(
            OfertaElegible(TipoOferta.OPCION_PAGO, "ampliacion_plazo", {}, "Elegible."),
            1,
            "Es la opción priorizada.",
        ),
        WhitelistItem(
            OfertaElegible(TipoOferta.OPCION_PAGO, "reduccion_cuota", {}, "Elegible."),
            2,
            "Es la siguiente alternativa.",
        ),
    ]


def test_rejecting_current_presents_next_source_whitelist_offer_without_pipeline_dependencies() -> None:
    whitelist = _whitelist()
    session = AlternativeSessionController(whitelist, "prompt-v1")

    initial = session.present_current()
    next_alternative = session.reject_current()

    assert initial.presented_offer == whitelist[0]
    assert next_alternative.presented_offer == whitelist[1]
    assert next_alternative.response.ofertas_mencionadas == ["reduccion_cuota"]
    assert next_alternative.response.prompt_version == "prompt-v1"
    assert next_alternative.rejected_offer_ids == frozenset({"ampliacion_plazo"})
    assert next_alternative.source_whitelist == tuple(whitelist)
    module = importlib.import_module("sistema_agentico.orchestration.alternative_session")
    assert "EligibilityEngine" not in vars(module)
    assert "NBARanker" not in vars(module)


def test_rejecting_every_offer_returns_a_safe_exhausted_response() -> None:
    session = AlternativeSessionController(_whitelist(), "prompt-v1")

    session.reject_current()
    exhausted = session.request_next_alternative()
    repeated = session.reject_current()

    assert exhausted.exhausted is True
    assert exhausted.presented_offer is None
    assert exhausted.response.ofertas_mencionadas == []
    assert "no existen más alternativas elegibles autorizadas" in exhausted.response.texto
    assert repeated == exhausted


def test_outside_whitelist_request_never_offers_the_requested_condition() -> None:
    session = AlternativeSessionController(_whitelist(), "prompt-v1")

    outcome = session.request_outside_whitelist()

    assert outcome.outside_whitelist_requested is True
    assert outcome.presented_offer is None
    assert outcome.response.ofertas_mencionadas == []
    assert "reduccion_cuota" not in outcome.response.texto
    assert session.present_current().presented_offer == _whitelist()[0]
