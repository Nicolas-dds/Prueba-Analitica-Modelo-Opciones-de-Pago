"""Sequential runner for independent proactive batch interactions.

Each prioritized obligation crosses :class:`InputGuardrail` before the already-
sanitzed input is handed to :class:`Orchestrator`.  The runner deliberately creates
no conversation history and catches failures at item scope, so one obligation cannot
prevent later obligations in the same prioritized batch from being processed.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from sistema_agentico.types import InputMode, RawInput, SanitizedInput
from sistema_agentico.orchestration.orchestrator import InteractionOutcome

__all__ = ["ProactiveBatchItemOutcome", "ProactiveBatchRunner"]


class _InputProcessor(Protocol):
    def process(self, raw_input: RawInput) -> SanitizedInput: ...


class _InteractionHandler(Protocol):
    def handle_interaction(self, sanitized_input: SanitizedInput) -> InteractionOutcome: ...


@dataclass(frozen=True)
class ProactiveBatchItemOutcome:
    """Result recorded locally for one obligation in a proactive batch.

    Exactly one of ``outcome`` and ``error`` is present.  Errors are represented as
    stable, non-throwing records so the remaining obligations can still complete.
    """

    id_obligacion: str
    outcome: InteractionOutcome | None
    error: str | None

    def __post_init__(self) -> None:
        if not self.id_obligacion:
            raise ValueError("id_obligacion no puede ser vacío")
        if (self.outcome is None) == (self.error is None):
            raise ValueError("un resultado de lote requiere exactamente outcome o error")

    @property
    def completed(self) -> bool:
        """Whether the orchestrator completed a controlled interaction outcome."""
        return self.outcome is not None


class ProactiveBatchRunner:
    """Process prioritized obligations independently and in their received order."""

    def __init__(
        self,
        input_guardrail: _InputProcessor,
        orchestrator: _InteractionHandler,
    ) -> None:
        self._input_guardrail = input_guardrail
        self._orchestrator = orchestrator

    def run(self, prioritized_obligation_ids: Iterable[str]) -> list[ProactiveBatchItemOutcome]:
        """Run one new proactive interaction per prioritized obligation.

        Each iteration constructs a separate ``RawInput`` with no customer message,
        always processes it through the input guardrail, and invokes the orchestrator
        without conversation history.  Guardrail or orchestrator failures are captured
        for only that item; no exception stops the rest of the batch.
        """
        results: list[ProactiveBatchItemOutcome] = []
        for id_obligacion in prioritized_obligation_ids:
            try:
                raw_input = RawInput(
                    mode=InputMode.PROACTIVO,
                    id_obligacion=id_obligacion,
                    mensaje_crudo=None,
                )
                sanitized_input = self._input_guardrail.process(raw_input)
            except Exception as exc:
                results.append(self._failure(id_obligacion, "InputGuardrail", exc))
                continue

            try:
                outcome = self._orchestrator.handle_interaction(sanitized_input)
            except Exception as exc:
                results.append(self._failure(id_obligacion, "Orchestrator", exc))
                continue

            results.append(
                ProactiveBatchItemOutcome(
                    id_obligacion=id_obligacion,
                    outcome=outcome,
                    error=None,
                )
            )
        return results

    @staticmethod
    def _failure(
        id_obligacion: str,
        component: str,
        exc: Exception,
    ) -> ProactiveBatchItemOutcome:
        return ProactiveBatchItemOutcome(
            id_obligacion=id_obligacion,
            outcome=None,
            error=f"{component} failed: {type(exc).__name__}: {exc}",
        )
