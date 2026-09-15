"""Reactive entrypoint that enforces input sanitization before orchestration.

``ReactiveRunner`` accepts a raw customer message, but never gives that message to
an orchestrator or conversational agent directly.  It always constructs a
:class:`RawInput` in reactive mode and sends it through ``InputGuardrail`` first.

Callers are responsible for supplying only *previously sanitized* conversation
history.  This runner forwards that history unchanged to the orchestrator; it does
not implement multi-turn rejection or alternative-offer behavior.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from sistema_agentico.conversational import ConversationTurn
from sistema_agentico.orchestration.orchestrator import InteractionOutcome
from sistema_agentico.types import InputMode, RawInput, SanitizedInput

__all__ = ["ReactiveRunner", "ReactiveRunnerError"]


class _InputProcessor(Protocol):
    def process(self, raw_input: RawInput) -> SanitizedInput: ...


class _InteractionHandler(Protocol):
    def handle_interaction(
        self,
        sanitized_input: SanitizedInput,
        *,
        historial_conversacion: Sequence[ConversationTurn] = (),
    ) -> InteractionOutcome: ...


class ReactiveRunnerError(RuntimeError):
    """Controlled failure at the reactive boundary.

    The original error is retained as ``cause`` and as the exception cause, allowing
    an API caller to distinguish an unavailable guardrail from an orchestration
    failure without mistaking it for a final customer interaction outcome.
    """

    def __init__(self, component: str, cause: Exception) -> None:
        self.component = component
        self.cause = cause
        super().__init__(f"{component} failed: {type(cause).__name__}: {cause}")


class ReactiveRunner:
    """Run one raw reactive customer interaction through the mandatory guardrail."""

    def __init__(
        self,
        input_guardrail: _InputProcessor,
        orchestrator: _InteractionHandler,
    ) -> None:
        self._input_guardrail = input_guardrail
        self._orchestrator = orchestrator

    def run(
        self,
        id_obligacion: str,
        mensaje_cliente: str,
        *,
        historial_conversacion: Sequence[ConversationTurn] | None = None,
    ) -> InteractionOutcome:
        """Return the orchestrator's terminal outcome for one customer message.

        ``mensaje_cliente`` is intentionally raw and is sent only to
        ``InputGuardrail.process``.  ``historial_conversacion`` must already be
        sanitized by the caller because prior turns do not cross the input guardrail
        again.  A blocked sanitized input is forwarded to the orchestrator, which
        creates the controlled escalation without invoking the conversational agent.

        Guardrail and orchestrator exceptions become :class:`ReactiveRunnerError`
        rather than fabricated final or escalated outcomes.
        """
        self._validate_request(id_obligacion, mensaje_cliente, historial_conversacion)
        history = tuple(historial_conversacion or ())
        raw_input = RawInput(
            mode=InputMode.REACTIVO,
            id_obligacion=id_obligacion,
            mensaje_crudo=mensaje_cliente,
        )

        try:
            sanitized_input = self._input_guardrail.process(raw_input)
        except Exception as exc:
            raise ReactiveRunnerError("InputGuardrail", exc) from exc
        if not isinstance(sanitized_input, SanitizedInput):
            error = TypeError("InputGuardrail.process debe retornar SanitizedInput")
            raise ReactiveRunnerError("InputGuardrail", error) from error

        try:
            return self._orchestrator.handle_interaction(
                sanitized_input,
                historial_conversacion=history,
            )
        except Exception as exc:
            raise ReactiveRunnerError("Orchestrator", exc) from exc

    @staticmethod
    def _validate_request(
        id_obligacion: str,
        mensaje_cliente: str,
        historial_conversacion: Sequence[ConversationTurn] | None,
    ) -> None:
        if not isinstance(id_obligacion, str):
            raise TypeError("id_obligacion debe ser str")
        if not id_obligacion:
            raise ValueError("id_obligacion no puede ser vacío")
        if not isinstance(mensaje_cliente, str):
            raise TypeError("mensaje_cliente debe ser str")
        if historial_conversacion is None:
            return
        if isinstance(historial_conversacion, (str, bytes, bytearray)) or not isinstance(
            historial_conversacion, Sequence
        ):
            raise TypeError("historial_conversacion debe ser una secuencia de turnos sanitizados")
        if any(not isinstance(turn, (str, Mapping)) for turn in historial_conversacion):
            raise TypeError("cada turno del historial debe ser str o Mapping sanitizado")
