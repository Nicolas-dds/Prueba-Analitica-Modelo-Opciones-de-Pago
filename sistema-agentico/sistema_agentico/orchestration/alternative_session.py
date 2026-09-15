"""Deterministic multi-turn handling of alternatives from one authorized whitelist.

This controller starts *after* the initial pipeline has built and prioritized a
whitelist.  It deliberately has no dependency on eligibility, ranking, or an LLM:
every customer-visible offer comes from the immutable session snapshot.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from sistema_agentico.types import AgentResponse, WhitelistItem

__all__ = ["AlternativeSessionController", "AlternativeSessionOutcome"]


@dataclass(frozen=True)
class AlternativeSessionOutcome:
    """Safe, explicit result of one alternative-selection turn.

    ``source_whitelist`` is an immutable snapshot of the whitelist that originated
    the conversation.  ``exhausted`` reflects whether every source offer has been
    rejected; an outside-whitelist request is separately identified because it must
    produce no offer even when an authorized offer remains in the session.
    """

    response: AgentResponse
    presented_offer: WhitelistItem | None
    exhausted: bool
    source_whitelist: tuple[WhitelistItem, ...]
    rejected_offer_ids: frozenset[str]
    outside_whitelist_requested: bool = False


class AlternativeSessionController:
    """Present only remaining offers from the initial ranked whitelist.

    The controller is intentionally stateful only with respect to the current
    session: it stores rejected IDs and a cursor over an immutable tuple snapshot.
    It never re-evaluates eligibility, re-ranks offers, or accepts an offer outside
    that snapshot.  Call :meth:`present_current` for the initial offer, then call
    :meth:`reject_current` or :meth:`request_next_alternative` for subsequent turns.
    """

    _NO_FURTHER_ALTERNATIVES_TEXT = (
        "En este momento no existen más alternativas elegibles autorizadas. "
        "Para recibir orientación adicional, comuníquese con un gestor."
    )

    def __init__(self, whitelist: Sequence[WhitelistItem], prompt_version: str) -> None:
        if not isinstance(prompt_version, str) or not prompt_version.strip():
            raise ValueError("prompt_version debe ser un texto no vacío")

        snapshot = tuple(whitelist)
        offer_ids = [item.oferta.id_opcion for item in snapshot]
        if len(offer_ids) != len(set(offer_ids)):
            raise ValueError("la whitelist de sesión no puede contener id_opcion duplicados")

        self._source_whitelist = snapshot
        self._prompt_version = prompt_version.strip()
        self._current_index = 0
        self._rejected_offer_ids: set[str] = set()

    @property
    def source_whitelist(self) -> tuple[WhitelistItem, ...]:
        """Return the immutable whitelist snapshot authorized for this session."""
        return self._source_whitelist

    @property
    def rejected_offer_ids(self) -> frozenset[str]:
        """Return the offer IDs explicitly declined during this session."""
        return frozenset(self._rejected_offer_ids)

    @property
    def exhausted(self) -> bool:
        """Whether every offer in the source whitelist has been rejected."""
        return self._current_offer() is None

    def present_current(self) -> AlternativeSessionOutcome:
        """Present the current highest-priority offer not already rejected."""
        return self._outcome_for_current()

    def reject_current(self) -> AlternativeSessionOutcome:
        """Record rejection of the current offer and present the next authorized one."""
        current = self._current_offer()
        if current is not None:
            self._rejected_offer_ids.add(current.oferta.id_opcion)
            self._current_index += 1
        return self._outcome_for_current()

    def request_next_alternative(self) -> AlternativeSessionOutcome:
        """Treat a request for another alternative as declining the current offer."""
        return self.reject_current()

    def request_outside_whitelist(self) -> AlternativeSessionOutcome:
        """Refuse an out-of-whitelist request without exposing or creating an offer.

        The current authorized offer remains unchanged.  A caller that maps a
        customer message to this explicit action must never include the requested
        condition in the response text.
        """
        return AlternativeSessionOutcome(
            response=self._no_further_alternatives_response(),
            presented_offer=None,
            exhausted=self.exhausted,
            source_whitelist=self._source_whitelist,
            rejected_offer_ids=self.rejected_offer_ids,
            outside_whitelist_requested=True,
        )

    def _current_offer(self) -> WhitelistItem | None:
        while self._current_index < len(self._source_whitelist):
            item = self._source_whitelist[self._current_index]
            if item.oferta.id_opcion not in self._rejected_offer_ids:
                return item
            self._current_index += 1
        return None

    def _outcome_for_current(self) -> AlternativeSessionOutcome:
        current = self._current_offer()
        if current is None:
            response = self._no_further_alternatives_response()
        else:
            response = AgentResponse(
                texto=(
                    f"La alternativa autorizada disponible es {current.oferta.id_opcion}. "
                    f"{current.justificacion}"
                ),
                ofertas_mencionadas=[current.oferta.id_opcion],
                requiere_reintento=False,
                prompt_version=self._prompt_version,
            )
        return AlternativeSessionOutcome(
            response=response,
            presented_offer=current,
            exhausted=current is None,
            source_whitelist=self._source_whitelist,
            rejected_offer_ids=self.rejected_offer_ids,
        )

    def _no_further_alternatives_response(self) -> AgentResponse:
        return AgentResponse(
            texto=self._NO_FURTHER_ALTERNATIVES_TEXT,
            ofertas_mencionadas=[],
            requiere_reintento=False,
            prompt_version=self._prompt_version,
        )
