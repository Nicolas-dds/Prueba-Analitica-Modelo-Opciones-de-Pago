"""Extracción determinista y fail-closed de IDs de oferta en texto LLM.

La extracción sólo reconoce tokens ``id_opcion`` exactos. Los IDs se devuelven una
vez y en orden de primera aparición en el texto; por tanto, no dependen del orden
de la whitelist. El módulo no intenta resolver equivalencias semánticas: ante
identificadores parecidos, identificadores con formato de oferta desconocidos, o
nombres canónicos de ofertas sin su ID exacto, rechaza la respuesta.
"""
from __future__ import annotations

import re
from collections.abc import Sequence

from sistema_agentico.types import WhitelistItem

__all__ = ["OfferMentionExtractionError", "extract_offers_mentioned"]


class OfferMentionExtractionError(ValueError):
    """No es seguro construir ``ofertas_mencionadas`` a partir del texto."""

    def __init__(self, reason: str, token: str) -> None:
        self.reason = reason
        self.token = token
        super().__init__(f"extracción de ofertas no confiable: {reason}: {token!r}")


# Un token de identificador no puede estar pegado a letras, números, guiones o
# guiones bajos. Se admiten signos de puntuación como delimitadores naturales.
_TOKEN_RE = re.compile(r"(?<![\w-])[\w-]+(?![\w-])")
_GENERIC_OFFER_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+$")
_CANONICAL_FAMILY_RE = re.compile(
    r"^(?:(?:ampliaci[oó]n|reducci[oó]n|renegociaci[oó]n|"
    r"reestructuraci[oó]n)(?:[_-][\w]+)*|"
    r"acuerdo(?:[_-]pago)(?:[_-][\w]+)*)$",
    re.IGNORECASE,
)

# Esto no es inferencia de lenguaje natural: son expresiones explícitas para las
# cinco familias canónicas del catálogo actual. Si aparecen sin su id_opcion
# exacto, no se puede reportar la mención con certeza y se rechaza fail-closed.
_CANONICAL_DESCRIPTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "ampliacion_plazo": re.compile(r"\bampliaci[oó]n\s+de\s+plazo\b", re.IGNORECASE),
    "reduccion_cuota": re.compile(r"\breducci[oó]n\s+de\s+cuota\b", re.IGNORECASE),
    "renegociacion_tasa": re.compile(
        r"\brenegociaci[oó]n\s+de\s+tasa\b", re.IGNORECASE
    ),
    "reestructuracion": re.compile(r"\breestructuraci[oó]n\b", re.IGNORECASE),
    "acuerdo_pago_estandar": re.compile(r"\bacuerdo\s+de\s+pago\b", re.IGNORECASE),
}


def extract_offers_mentioned(
    texto: str, whitelist: Sequence[WhitelistItem]
) -> list[str]:
    """Return exact whitelist IDs mentioned in ``texto`` in text-occurrence order.

    Duplicate mentions collapse to one entry. A repeated ID in the whitelist is
    harmless because the externally observable metadata is a unique ID list. Any
    token that could denote an offer but is not a complete allowed ID raises
    :class:`OfferMentionExtractionError`; callers must not emit an incomplete
    :class:`~sistema_agentico.types.AgentResponse` in that case.
    """
    if not isinstance(texto, str):
        raise TypeError("texto debe ser str")

    allowed_ids = {item.oferta.id_opcion for item in whitelist}
    exact_mentions = _exact_mentions(texto, allowed_ids)
    _reject_described_canonical_offers(texto, {offer_id for _, offer_id in exact_mentions})
    _reject_ambiguous_tokens(texto, allowed_ids)

    mentioned: list[str] = []
    seen: set[str] = set()
    for _, offer_id in exact_mentions:
        if offer_id not in seen:
            mentioned.append(offer_id)
            seen.add(offer_id)
    return mentioned


def _exact_mentions(texto: str, allowed_ids: set[str]) -> list[tuple[int, str]]:
    mentions: list[tuple[int, str]] = []
    for offer_id in sorted(allowed_ids):
        pattern = re.compile(rf"(?<![\w-]){re.escape(offer_id)}(?![\w-])")
        mentions.extend((match.start(), offer_id) for match in pattern.finditer(texto))
    return sorted(mentions)


def _reject_ambiguous_tokens(texto: str, allowed_ids: set[str]) -> None:
    for match in _TOKEN_RE.finditer(texto):
        token = match.group(0)
        if token in allowed_ids:
            continue
        if any(offer_id in token for offer_id in allowed_ids):
            raise OfferMentionExtractionError("referencia parcial de un ID autorizado", token)
        if _CANONICAL_FAMILY_RE.fullmatch(token):
            raise OfferMentionExtractionError("ID de familia canónica no autorizado", token)
        if _GENERIC_OFFER_ID_RE.fullmatch(token):
            raise OfferMentionExtractionError("ID de oferta desconocido", token)


def _reject_described_canonical_offers(texto: str, exact_ids: set[str]) -> None:
    for canonical_id, pattern in _CANONICAL_DESCRIPTION_PATTERNS.items():
        if pattern.search(texto) and canonical_id not in exact_ids:
            raise OfferMentionExtractionError(
                "oferta canónica descrita sin su ID exacto", canonical_id
            )
