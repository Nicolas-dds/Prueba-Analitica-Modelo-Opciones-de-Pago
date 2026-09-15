"""Gestor de Escalamiento (`EscalationManager`).

Ver Component 7 de `design.md` y Requirement 9 de `requirements.md`.
"""
from sistema_agentico.escalation.manager import (
    EscalationCaseSink,
    EscalationManager,
    EscalationOutcome,
    EscalationTrigger,
    InMemoryEscalationCaseSink,
    UnsupportedEscalationTriggerError,
    map_trigger_to_reason,
)

__all__ = [
    "EscalationCaseSink",
    "EscalationManager",
    "EscalationOutcome",
    "EscalationTrigger",
    "InMemoryEscalationCaseSink",
    "UnsupportedEscalationTriggerError",
    "map_trigger_to_reason",
]
