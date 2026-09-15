"""Deterministic whitelist validator for conversational-agent output (Requirement 8.1)."""
from __future__ import annotations

from sistema_agentico.types import (
    AgentResponse,
    ClienteObligacionContext,
    ValidationResult,
    WhitelistItem,
)


class OutputGuardrail:
    """Validates only agent-reported offer IDs; it never parses response text or calls an LLM."""

    def validate(
        self,
        respuesta: AgentResponse,
        whitelist: list[WhitelistItem],
        contexto: ClienteObligacionContext,
    ) -> ValidationResult:
        """Accept iff every ID in ``respuesta.ofertas_mencionadas`` is whitelisted.

        ``contexto`` belongs to the component interface but is intentionally not
        inspected by first-level offer validation. Escalation and retry policy are
        owned by ``EscalationManager`` (task 10.2).
        """
        del contexto
        authorized_ids = {item.oferta.id_opcion for item in whitelist}
        unauthorized_ids = tuple(
            dict.fromkeys(
                offer_id
                for offer_id in respuesta.ofertas_mencionadas
                if offer_id not in authorized_ids
            )
        )
        if unauthorized_ids:
            return ValidationResult(
                valido=False,
                motivo_rechazo=(
                    "Ofertas fuera de la whitelist autorizada: "
                    f"{', '.join(unauthorized_ids)}"
                ),
                escalar=False,
                razon_escalamiento=None,
            )
        return ValidationResult(
            valido=True,
            motivo_rechazo=None,
            escalar=False,
            razon_escalamiento=None,
        )
