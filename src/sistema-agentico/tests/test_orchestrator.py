"""Focused unit coverage for deterministic Component 2 orchestration."""
from __future__ import annotations

import logging

import pytest

from sistema_agentico.escalation import EscalationManager, InMemoryEscalationCaseSink
from sistema_agentico.guardrails import OutputGuardrail
from sistema_agentico.orchestration import InteractionStatus, Orchestrator
from sistema_agentico.types import (
    AgentResponse,
    ClienteObligacionContext,
    EligibilityResult,
    EscalationReason,
    InputMode,
    OfertaElegible,
    SanitizedInput,
    ScoreInfo,
    TipoOferta,
    WhitelistItem,
)


def _context() -> ClienteObligacionContext:
    return ClienteObligacionContext(
        id_obligacion="OBL-1", id_cliente="CLI-SYN-1", dias_mora=15,
        exposicion=100_000.0, segmento="synthetic", canal_gestion="chat",
        score=ScoreInfo(0.8, 8, "score-v1", "2025-01-01", False),
        ofertas_aplicadas_mes=[], ultima_opcion_aplicada=None,
        acuerdo_pago_vigente=False, restriccion_vigente=False,
    )


def _eligibility() -> EligibilityResult:
    offer = OfertaElegible(TipoOferta.OPCION_PAGO, "ampliacion_plazo", {}, "Preaprobada.")
    return EligibilityResult([offer], None, None)


class _ContextBuilder:
    def __init__(self) -> None:
        self.calls = 0

    def build(self, id_obligacion: str) -> ClienteObligacionContext:
        self.calls += 1
        assert id_obligacion == "OBL-1"
        return _context()


class _EligibilityEngine:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, context: ClienteObligacionContext) -> EligibilityResult:
        self.calls += 1
        assert context == _context()
        return _eligibility()


class _Ranker:
    def __init__(self) -> None:
        self.calls = 0

    def rank(self, eligibility: EligibilityResult, context: ClienteObligacionContext) -> list[WhitelistItem]:
        self.calls += 1
        assert eligibility == _eligibility()
        return [WhitelistItem(eligibility.opciones_elegibles[0], 1, "Autorizada.")]


class _Agent:
    def __init__(self, generated: AgentResponse, corrected: AgentResponse | None = None) -> None:
        self.generated = generated
        self.corrected = corrected
        self.generate_calls = 0
        self.retry_calls = 0

    def generate(self, *args: object) -> AgentResponse:
        self.generate_calls += 1
        return self.generated

    def retry_with_correction(self, *args: object) -> AgentResponse:
        self.retry_calls += 1
        assert self.corrected is not None
        return self.corrected


class _TraceLogger:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.records = []

    def append(self, record: object) -> None:
        if self.fail:
            raise OSError("trace store unavailable")
        self.records.append(record)


def _input(*, blocked: bool = False) -> SanitizedInput:
    return SanitizedInput(
        mode=InputMode.REACTIVO, id_obligacion="OBL-1", mensaje_sanitizado="Necesito ayuda.",
        pii_detectada=[], riesgo_inyeccion=0.99 if blocked else 0.0, bloqueado=blocked,
    )


def _orchestrator(agent: _Agent, trace_logger: _TraceLogger, sink: InMemoryEscalationCaseSink | None = None) -> tuple[Orchestrator, _ContextBuilder, _EligibilityEngine, _Ranker]:
    context, eligibility, ranker = _ContextBuilder(), _EligibilityEngine(), _Ranker()
    return (
        Orchestrator(context, eligibility, ranker, agent, OutputGuardrail(),
                     EscalationManager(case_sink=sink), trace_logger),
        context, eligibility, ranker,
    )


def test_normal_flow_generates_once_and_records_complete_trace() -> None:
    agent = _Agent(AgentResponse("Oferta ampliacion_plazo", ["ampliacion_plazo"], False, "prompt-v1"))
    logger = _TraceLogger()
    orchestrator, context, eligibility, ranker = _orchestrator(agent, logger)

    outcome = orchestrator.handle_interaction(_input())

    assert outcome.status is InteractionStatus.FINAL
    assert outcome.response == agent.generated
    assert outcome.trace_logged is True
    assert context.calls == eligibility.calls == ranker.calls == agent.generate_calls == 1
    assert agent.retry_calls == 0
    assert len(logger.records) == 1
    trace = logger.records[0]
    assert trace.trace_id == outcome.trace_id
    assert trace.respuesta_agente == "Oferta ampliacion_plazo"
    assert trace.reglas_evaluadas["pipeline_steps"] == [
        "interaction_started", "context_built", "eligibility_evaluated", "nba_ranked",
        "agent_generated_attempt_0", "output_validated_attempt_0", "interaction_finalized",
    ]


def test_blocked_input_escalates_without_context_or_agent_and_is_traced() -> None:
    agent = _Agent(AgentResponse("unused", [], False, "prompt-v1"))
    logger, sink = _TraceLogger(), InMemoryEscalationCaseSink()
    orchestrator, context, eligibility, ranker = _orchestrator(agent, logger, sink)

    outcome = orchestrator.handle_interaction(_input(blocked=True))

    assert outcome.status is InteractionStatus.ESCALATED
    assert outcome.escalation_reason is EscalationReason.MANIPULACION_DETECTADA
    assert outcome.escalation_case is not None
    assert context.calls == eligibility.calls == ranker.calls == agent.generate_calls == 0
    assert logger.records[0].validation_result.razon_escalamiento is EscalationReason.MANIPULACION_DETECTADA
    assert logger.records[0].reglas_evaluadas["input_bloqueado"] is True
    assert sink.cases() == (outcome.escalation_case,)


def test_ordinary_invalid_response_is_corrected_once_then_finalized() -> None:
    agent = _Agent(
        AgentResponse("Oferta no_autorizada", ["no_autorizada"], False, "prompt-v1"),
        AgentResponse("Oferta ampliacion_plazo", ["ampliacion_plazo"], False, "prompt-v1"),
    )
    logger = _TraceLogger()
    orchestrator, _, _, _ = _orchestrator(agent, logger)

    outcome = orchestrator.handle_interaction(_input())

    assert outcome.status is InteractionStatus.FINAL
    assert outcome.response == agent.corrected
    assert agent.generate_calls == 1
    assert agent.retry_calls == 1
    assert logger.records[0].ofertas_mencionadas == ["ampliacion_plazo"]
    assert logger.records[0].validation_result.valido is True


def test_trace_logger_failure_does_not_change_a_final_customer_outcome() -> None:
    agent = _Agent(AgentResponse("Oferta ampliacion_plazo", ["ampliacion_plazo"], False, "prompt-v1"))
    orchestrator, _, _, _ = _orchestrator(agent, _TraceLogger(fail=True))

    outcome = orchestrator.handle_interaction(_input())

    assert outcome.status is InteractionStatus.FINAL
    assert outcome.response == agent.generated
    assert outcome.trace_logged is False


def test_unreliable_extraction_signal_is_corrected_once_then_finalized() -> None:
    """Regression for task 18.1.

    ``requiere_reintento=True`` (set when ``ConversationalAgent`` could not
    reliably extract ``ofertas_mencionadas``, e.g. the LLM described a canonical
    offer in prose) must trigger the same single-correction retry as an
    unauthorized-offer rejection, not an immediate technical escalation.
    """
    agent = _Agent(
        AgentResponse("La opción es ampliación de plazo.", [], True, "prompt-v1"),
        AgentResponse("Oferta ampliacion_plazo", ["ampliacion_plazo"], False, "prompt-v1"),
    )
    logger = _TraceLogger()
    orchestrator, _, _, _ = _orchestrator(agent, logger)

    outcome = orchestrator.handle_interaction(_input())

    assert outcome.status is InteractionStatus.FINAL
    assert outcome.response == agent.corrected
    assert agent.generate_calls == 1
    assert agent.retry_calls == 1
    assert logger.records[0].validation_result.valido is True


def test_unreliable_extraction_signal_on_retry_escalates_reintento_fallido() -> None:
    """If the correction attempt also cannot be trusted, escalate (never loop)."""
    agent = _Agent(
        AgentResponse("La opción es ampliación de plazo.", [], True, "prompt-v1"),
        AgentResponse("Sigue mencionando ampliación de plazo.", [], True, "prompt-v1"),
    )
    logger, sink = _TraceLogger(), InMemoryEscalationCaseSink()
    orchestrator, _, _, _ = _orchestrator(agent, logger, sink)

    outcome = orchestrator.handle_interaction(_input())

    assert outcome.status is InteractionStatus.ESCALATED
    assert outcome.escalation_reason is EscalationReason.REINTENTO_FALLIDO
    assert agent.generate_calls == 1
    assert agent.retry_calls == 1
    assert sink.cases() == (outcome.escalation_case,)


def test_technical_failure_logs_exception_type_and_trace_id_without_message_content(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Regression for task 18.2: technical failures must be diagnosable from logs."""

    class _RaisingContextBuilder:
        def build(self, id_obligacion: str) -> ClienteObligacionContext:
            raise RuntimeError("boom: llm client unavailable")

    agent = _Agent(AgentResponse("unused", [], False, "prompt-v1"))
    orchestrator = Orchestrator(
        _RaisingContextBuilder(), _EligibilityEngine(), _Ranker(), agent,
        OutputGuardrail(), EscalationManager(), _TraceLogger(),
    )

    secret_message = "Necesito ayuda con mi cedula 123456789."
    sanitized = SanitizedInput(
        mode=InputMode.REACTIVO, id_obligacion="OBL-1", mensaje_sanitizado=secret_message,
        pii_detectada=[], riesgo_inyeccion=0.0, bloqueado=False,
    )

    with caplog.at_level(logging.ERROR, logger="sistema_agentico.orchestration.orchestrator"):
        outcome = orchestrator.handle_interaction(sanitized)

    assert outcome.status is InteractionStatus.ESCALATED
    assert outcome.escalation_reason is EscalationReason.FALLO_TECNICO
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == logging.ERROR
    assert "RuntimeError" in record.message
    assert outcome.trace_id in record.message
    assert "boom: llm client unavailable" in record.message
    assert secret_message not in record.message
