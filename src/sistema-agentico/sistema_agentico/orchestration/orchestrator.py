"""Deterministic orchestration for one already-sanitized interaction.

``handle_interaction`` owns the fixed sequence and returns ``InteractionOutcome``:
a final response is represented explicitly, while an escalation never fabricates an
``AgentResponse``.  Input sanitization and proactivo/reactivo runners intentionally
remain outside this component.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from sistema_agentico.context import ContextEscalationRequired
from sistema_agentico.conversational import ConversationTurn
from sistema_agentico.types import (
    AgentResponse,
    ClienteObligacionContext,
    EligibilityResult,
    EscalationCase,
    EscalationReason,
    InputMode,
    SanitizedInput,
    ScoreInfo,
    TraceRecord,
    ValidationResult,
    WhitelistItem,
)

__all__ = ["InteractionOutcome", "InteractionStatus", "Orchestrator"]

_logger = logging.getLogger(__name__)

_MAX_EXCEPTION_DETAIL_CHARS = 500


def _safe_exception_detail(exc: Exception) -> str:
    """Return a length-bounded ``str(exc)`` for logging.

    This exception path is not expected to carry secrets (there is no API key or
    credential handling this deep in the orchestration), but some HTTP/LLM client
    errors embed request/response fragments in their ``__str__``. Truncating
    defensively avoids ever dumping an unbounded payload into logs.
    """
    detail = str(exc)
    if len(detail) > _MAX_EXCEPTION_DETAIL_CHARS:
        return detail[:_MAX_EXCEPTION_DETAIL_CHARS] + "...(truncated)"
    return detail


class InteractionStatus(Enum):
    """The only two externally observable terminal states of an interaction."""

    FINAL = "final"
    ESCALATED = "escalated"


@dataclass(frozen=True)
class InteractionOutcome:
    """Typed terminal result of :meth:`Orchestrator.handle_interaction`.

    ``response`` is populated only for a customer-safe final answer.  Escalations
    instead expose their deterministic ``escalation_reason`` and, when delivery did
    not fail, the ``EscalationCase`` returned by the manager.  ``trace_logged`` is
    observational only: an unavailable logger never changes the terminal result.
    """

    trace_id: str
    status: InteractionStatus
    response: AgentResponse | None
    escalation_reason: EscalationReason | None
    escalation_case: EscalationCase | None
    trace_logged: bool

    def __post_init__(self) -> None:
        if self.status is InteractionStatus.FINAL:
            if self.response is None or self.escalation_reason is not None:
                raise ValueError("un resultado final requiere respuesta y no razón de escalamiento")
        elif self.status is InteractionStatus.ESCALATED:
            if self.response is not None or self.escalation_reason is None:
                raise ValueError("un escalamiento requiere razón y no AgentResponse")
        else:  # defensive in case a non-enum value is supplied at runtime
            raise ValueError("status debe ser InteractionStatus")


@dataclass
class _InteractionState:
    """Mutable assembly state used only while building one immutable trace record."""

    contexto: ClienteObligacionContext
    eligibility: EligibilityResult
    whitelist: list[WhitelistItem]
    response: AgentResponse | None
    validation: ValidationResult
    steps: list[str]


class Orchestrator:
    """Execute the fixed non-LLM-controlled Component 2 sequence.

    Existing component interfaces do not accept a ``trace_id`` parameter.  This
    orchestrator therefore propagates it through the complete ``TraceRecord``, the
    escalation call and an optional safe observer; it never infers trace state from
    an LLM or exposes unsanitized input to callbacks.
    """

    def __init__(
        self,
        context_builder: Any,
        eligibility_engine: Any,
        nba_ranker: Any,
        conversational_agent: Any,
        output_guardrail: Any,
        escalation_manager: Any,
        trace_logger: Any,
        *,
        trace_observer: Callable[[str, str], None] | None = None,
    ) -> None:
        self._context_builder = context_builder
        self._eligibility_engine = eligibility_engine
        self._nba_ranker = nba_ranker
        self._conversational_agent = conversational_agent
        self._output_guardrail = output_guardrail
        self._escalation_manager = escalation_manager
        self._trace_logger = trace_logger
        self._trace_observer = trace_observer

    def handle_interaction(
        self,
        sanitized_input: SanitizedInput,
        *,
        historial_conversacion: Sequence[ConversationTurn] = (),
    ) -> InteractionOutcome:
        """Run context → eligibility → NBA → agent → output validation once per attempt.

        The caller must pass an already-sanitized ``SanitizedInput``.  The evaluation
        date/clock belongs to the injected ``EligibilityEngine`` so this method never
        introduces implicit wall-clock nondeterminism into eligibility decisions.
        """
        if not isinstance(sanitized_input, SanitizedInput):
            raise TypeError("sanitized_input debe ser SanitizedInput")

        trace_id = str(uuid4())
        history = list(historial_conversacion)
        state = self._initial_state(sanitized_input)
        self._record_step(trace_id, state, "interaction_started")

        if sanitized_input.bloqueado:
            state.validation = ValidationResult(
                valido=False,
                motivo_rechazo="Entrada bloqueada por riesgo de manipulación.",
                escalar=True,
                razon_escalamiento=EscalationReason.MANIPULACION_DETECTADA,
            )
            self._record_step(trace_id, state, "input_blocked")
            return self._finalize(
                sanitized_input, trace_id, history, state, EscalationReason.MANIPULACION_DETECTADA
            )

        try:
            state.contexto = self._context_builder.build(sanitized_input.id_obligacion)
            self._record_step(trace_id, state, "context_built")

            state.eligibility = self._eligibility_engine.evaluate(state.contexto)
            self._record_step(trace_id, state, "eligibility_evaluated")
            if self._has_eligible_offers(state.eligibility):
                state.whitelist = self._nba_ranker.rank(state.eligibility, state.contexto)
                self._record_step(trace_id, state, "nba_ranked")
            else:
                state.whitelist = []
                self._record_step(trace_id, state, "nba_skipped_no_eligible_offers")

            state.response = self._conversational_agent.generate(
                sanitized_input.mensaje_sanitizado,
                state.whitelist,
                state.contexto,
                history,
            )
            self._record_step(trace_id, state, "agent_generated_attempt_0")
            state.validation = self._validate_response(
                state.response, state.whitelist, state.contexto
            )
            self._record_step(trace_id, state, "output_validated_attempt_0")

            direct_reason = self._escalation_manager.resolve_escalation_reason(
                state.validation, intento_num=0
            )
            if state.validation.valido:
                return self._finalize(sanitized_input, trace_id, history, state, None)
            if direct_reason is not None:
                return self._finalize(sanitized_input, trace_id, history, state, direct_reason)

            # The manager permits exactly one ordinary correction after attempt zero.
            state.response = self._conversational_agent.retry_with_correction(
                state.response,
                state.validation.motivo_rechazo or "Respuesta no autorizada.",
                state.whitelist,
            )
            self._record_step(trace_id, state, "agent_generated_attempt_1")
            state.validation = self._validate_response(
                state.response, state.whitelist, state.contexto
            )
            self._record_step(trace_id, state, "output_validated_attempt_1")
            if state.validation.valido:
                return self._finalize(sanitized_input, trace_id, history, state, None)

            retry_reason = self._escalation_manager.resolve_escalation_reason(
                state.validation, intento_num=1
            ) or EscalationReason.REINTENTO_FALLIDO
            return self._finalize(sanitized_input, trace_id, history, state, retry_reason)
        except ContextEscalationRequired:
            state.validation = ValidationResult(
                valido=False,
                motivo_rechazo="El contexto requiere revisión por información contradictoria.",
                escalar=True,
                razon_escalamiento=EscalationReason.INFO_CONTRADICTORIA,
            )
            self._record_step(trace_id, state, "context_escalation_required")
            return self._finalize(
                sanitized_input, trace_id, history, state, EscalationReason.INFO_CONTRADICTORIA
            )
        except Exception as exc:  # Any component failure becomes a safe technical escalation.
            state.validation = ValidationResult(
                valido=False,
                motivo_rechazo="Fallo técnico durante la orquestación.",
                escalar=True,
                razon_escalamiento=EscalationReason.FALLO_TECNICO,
            )
            self._record_step(trace_id, state, f"technical_failure:{type(exc).__name__}")
            # Diagnosticable desde los logs del contenedor sin reproducir el fallo
            # manualmente. Nunca se registra `sanitized_input.mensaje_sanitizado`
            # (contenido de cliente, ya enmascarado por PII pero de todos modos no
            # es un hábito a mantener) ni ningún secreto. `str(exc)` se trunca de
            # forma defensiva: algunos errores de clientes HTTP/LLM pueden incrustar
            # cabeceras o payloads en su representación de texto.
            _logger.error(
                "Fallo técnico en Orchestrator.handle_interaction: "
                "exception_type=%s trace_id=%s detail=%s",
                type(exc).__name__,
                trace_id,
                _safe_exception_detail(exc),
            )
            return self._finalize(
                sanitized_input, trace_id, history, state, EscalationReason.FALLO_TECNICO
            )

    def _validate_response(
        self,
        response: AgentResponse,
        whitelist: list[WhitelistItem],
        contexto: ClienteObligacionContext,
    ) -> ValidationResult:
        """Validate ``response`` treating an explicit retry signal like an invalid one.

        ``ConversationalAgent`` sets ``requiere_reintento=True`` when it could not
        reliably extract ``ofertas_mencionadas`` from the LLM text (e.g. the model
        described a canonical offer in prose instead of its exact ``id_opcion``).
        In that case ``ofertas_mencionadas`` is empty, which ``OutputGuardrail``
        would otherwise accept as a trivially valid subset of the whitelist. That
        would silently skip the single-correction retry this failure mode needs
        (Requirements 6.6, 7.2, 8.2, 8.3), so this check runs before deferring to
        ``OutputGuardrail`` and never leaks the raw LLM text into the rejection
        reason handed to the correction prompt.
        """
        if response.requiere_reintento:
            return ValidationResult(
                valido=False,
                motivo_rechazo=(
                    "La respuesta mencionó una oferta sin usar su identificador "
                    "exacto de la whitelist."
                ),
                escalar=False,
                razon_escalamiento=None,
            )
        return self._output_guardrail.validate(response, whitelist, contexto)

    @staticmethod
    def _has_eligible_offers(eligibility: EligibilityResult) -> bool:
        return bool(eligibility.opciones_elegibles or eligibility.acuerdo_elegible)

    def _initial_state(self, sanitized_input: SanitizedInput) -> _InteractionState:
        contexto = self._placeholder_context(sanitized_input.id_obligacion)
        return _InteractionState(
            contexto=contexto,
            eligibility=self._placeholder_eligibility(),
            whitelist=[],
            response=None,
            validation=ValidationResult(
                valido=False,
                motivo_rechazo="Interacción aún no generada.",
                escalar=False,
                razon_escalamiento=None,
            ),
            steps=[],
        )

    @staticmethod
    def _placeholder_context(id_obligacion: str) -> ClienteObligacionContext:
        return ClienteObligacionContext(
            id_obligacion=id_obligacion,
            id_cliente="not-available",
            dias_mora=0,
            exposicion=0.0,
            segmento=None,
            canal_gestion=None,
            score=ScoreInfo(0.5, 5, "not-available", "not-available", True),
            ofertas_aplicadas_mes=[],
            ultima_opcion_aplicada=None,
            acuerdo_pago_vigente=False,
            restriccion_vigente=False,
        )

    @staticmethod
    def _placeholder_eligibility() -> EligibilityResult:
        return EligibilityResult(
            opciones_elegibles=[],
            acuerdo_elegible=None,
            motivo_no_elegible="No disponible por salida temprana de la interacción.",
        )

    def _record_step(self, trace_id: str, state: _InteractionState, step: str) -> None:
        state.steps.append(step)
        if self._trace_observer is not None:
            try:
                self._trace_observer(trace_id, step)
            except Exception:
                # Observability is never a control-flow dependency.
                pass

    def _finalize(
        self,
        sanitized_input: SanitizedInput,
        trace_id: str,
        history: Sequence[ConversationTurn],
        state: _InteractionState,
        escalation_reason: EscalationReason | None,
    ) -> InteractionOutcome:
        if escalation_reason is not None and state.validation.razon_escalamiento is None:
            state.validation = ValidationResult(
                valido=False,
                motivo_rechazo=state.validation.motivo_rechazo or "Interacción escalada.",
                escalar=True,
                razon_escalamiento=escalation_reason,
            )
        self._record_step(trace_id, state, "interaction_finalized")
        record = self._build_trace_record(sanitized_input, trace_id, state, escalation_reason)

        escalation_case: EscalationCase | None = None
        if escalation_reason is not None:
            try:
                escalation_case = self._escalation_manager.escalate_with_outcome(
                    state.contexto,
                    escalation_reason,
                    trace_id,
                    history,
                    trace_record=record,
                    modo=sanitized_input.mode,
                ).case
            except Exception:
                # The terminal state remains escalated even when downstream delivery fails.
                pass

        trace_logged = True
        try:
            self._trace_logger.append(record)
        except Exception:
            trace_logged = False

        if escalation_reason is not None:
            return InteractionOutcome(
                trace_id=trace_id,
                status=InteractionStatus.ESCALATED,
                response=None,
                escalation_reason=escalation_reason,
                escalation_case=escalation_case,
                trace_logged=trace_logged,
            )
        return InteractionOutcome(
            trace_id=trace_id,
            status=InteractionStatus.FINAL,
            response=state.response,
            escalation_reason=None,
            escalation_case=None,
            trace_logged=trace_logged,
        )

    @staticmethod
    def _build_trace_record(
        sanitized_input: SanitizedInput,
        trace_id: str,
        state: _InteractionState,
        escalation_reason: EscalationReason | None,
    ) -> TraceRecord:
        response = state.response
        return TraceRecord(
            trace_id=trace_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            id_obligacion=sanitized_input.id_obligacion,
            modo=sanitized_input.mode,
            score_info=state.contexto.score,
            reglas_evaluadas={
                "pipeline_steps": list(state.steps),
                "input_bloqueado": sanitized_input.bloqueado,
                "riesgo_inyeccion": sanitized_input.riesgo_inyeccion,
                "pii_tipos_detectados": list(sanitized_input.pii_detectada),
                "escalation_reason": escalation_reason.value if escalation_reason else None,
            },
            eligibility_result=state.eligibility,
            whitelist=list(state.whitelist),
            ofertas_mencionadas=list(response.ofertas_mencionadas) if response else [],
            respuesta_agente=response.texto if response else "",
            agente_responsable="conversational_agent",
            validation_result=state.validation,
            prompt_version=response.prompt_version if response else "not-available",
            version_modelo_propension=state.contexto.score.version_modelo,
        )
