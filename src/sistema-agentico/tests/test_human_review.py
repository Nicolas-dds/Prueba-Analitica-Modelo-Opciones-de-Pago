"""Focused tests for the HumanReviewInterface (Requirements 12.1--12.3)."""
from __future__ import annotations

from pathlib import Path

import pytest

from sistema_agentico.escalation import InMemoryEscalationCaseSink
from sistema_agentico.human_review import HumanReviewInputError, HumanReviewInterface
from sistema_agentico.synthetic_data.profiles import generate_profile
from sistema_agentico.trace import JsonlTraceLogger
from sistema_agentico.types import (
    ClienteObligacionContext,
    EligibilityResult,
    EscalationCase,
    EscalationReason,
    GoldenScenario,
    InputMode,
    ScoreInfo,
    TraceRecord,
    ValidationResult,
)


def _context() -> ClienteObligacionContext:
    return ClienteObligacionContext(
        id_obligacion="OBL-1", id_cliente="CLI-1", dias_mora=12, exposicion=500_000.0,
        segmento="masivo", canal_gestion="telefono",
        score=ScoreInfo(0.5, 5, "score-v1", "2025-01-01", False),
        ofertas_aplicadas_mes=[], ultima_opcion_aplicada=None,
        acuerdo_pago_vigente=False, restriccion_vigente=False,
    )


def _trace(trace_id: str, context: ClienteObligacionContext | None = None) -> TraceRecord:
    context = context or _context()
    return TraceRecord(
        trace_id=trace_id, timestamp="2025-01-01T00:00:00+00:00", id_obligacion=context.id_obligacion,
        modo=InputMode.REACTIVO, score_info=context.score, reglas_evaluadas={"regla": "aplicada"},
        eligibility_result=EligibilityResult([], None, "Sin oferta para esta prueba."), whitelist=[],
        ofertas_mencionadas=[], respuesta_agente="Respuesta rechazada", agente_responsable="conversational_agent",
        validation_result=ValidationResult(False, "Oferta inválida", True, EscalationReason.FUERA_DE_ALCANCE),
        prompt_version="test-v1", version_modelo_propension=context.score.version_modelo,
    )


def _case(trace_id: str, context: ClienteObligacionContext | None = None) -> EscalationCase:
    context = context or _context()
    return EscalationCase(
        trace_id=trace_id, id_obligacion=context.id_obligacion, razon=EscalationReason.FUERA_DE_ALCANCE,
        contexto=context, historial_conversacion=[], trace_completo=_trace(trace_id, context),
    )


def _review(tmp_path: Path, cases: list[EscalationCase]) -> tuple[HumanReviewInterface, JsonlTraceLogger]:
    logger = JsonlTraceLogger(tmp_path / "traces.jsonl")
    sink = InMemoryEscalationCaseSink()
    for case in cases:
        logger.append(case.trace_completo)
        sink.deliver(case)
    return HumanReviewInterface(trace_logger=logger, case_source=sink), logger


def test_get_pending_cases_returns_all_unresolved_cases_in_trace_order(tmp_path: Path) -> None:
    review, _ = _review(tmp_path, [_case("trace-b"), _case("trace-a")])

    assert [case.trace_id for case in review.get_pending_cases()] == ["trace-a", "trace-b"]


def test_resolve_case_appends_linked_event_without_mutating_original(tmp_path: Path) -> None:
    review, logger = _review(tmp_path, [_case("trace-1")])
    original = logger.get_trace("trace-1")

    review.resolve_case("trace-1", "Se acuerda contacto en dos días.", "gestor.17")

    assert logger.get_trace("trace-1") == original
    history = logger.get_history(original.id_obligacion)
    event = next(record for record in history if record.trace_id != original.trace_id)
    assert event.trace_id != original.trace_id
    assert event.reglas_evaluadas["trace_id_origen"] == original.trace_id
    assert event.agente_responsable == "human:gestor.17"
    assert event.respuesta_agente == "Se acuerda contacto en dos días."
    assert review.get_pending_cases() == []


class _FailingAppendLogger:
    def __init__(self, original: TraceRecord) -> None:
        self.original = original

    def get_trace(self, trace_id: str) -> TraceRecord:
        assert trace_id == self.original.trace_id
        return self.original

    def append(self, record: TraceRecord) -> None:
        del record
        raise OSError("trace storage unavailable")


def test_resolution_closes_case_when_event_append_fails_but_original_is_intact() -> None:
    case = _case("trace-1")
    sink = InMemoryEscalationCaseSink()
    sink.deliver(case)
    review = HumanReviewInterface(trace_logger=_FailingAppendLogger(case.trace_completo), case_source=sink)  # type: ignore[arg-type]

    review.resolve_case("trace-1", "Gestionado por llamada.", "gestor-1")

    assert review.get_pending_cases() == []
    assert review.resolution_logging_error("trace-1") == "OSError: trace storage unavailable"
    assert case.trace_completo == _trace("trace-1")


def test_feedback_sanitizes_free_text_and_keeps_trace_origin_outside_scenario(tmp_path: Path) -> None:
    context = generate_profile(seed=7, index=7)
    case = _case("trace-1", context)
    review, _ = _review(tmp_path, [case])
    scenarios: list[GoldenScenario] = []
    review = HumanReviewInterface(
        trace_logger=JsonlTraceLogger(tmp_path / "traces.jsonl"),
        case_source=_source(case),
        golden_scenario_sink=scenarios,
    )
    scenario = GoldenScenario(
        scenario_id="SYN-GOLD-001", descripcion="Caso de Juan Pérez", contexto=context,
        mensaje_cliente="Mi cédula es 12345678", resultado_esperado={"nota": "Llame al 3001234567"},
    )

    review.feedback_to_golden_dataset("trace-1", scenario)

    assert scenarios[0].descripcion == "Caso de [NOMBRE]"
    assert scenarios[0].mensaje_cliente == "Mi cédula es [CEDULA]"
    assert scenarios[0].resultado_esperado == {"nota": "Llame al [TELEFONO]"}
    assert review.feedback_origin_trace_id("SYN-GOLD-001") == "trace-1"
    assert "trace_id_origen" not in scenarios[0].resultado_esperado


def test_feedback_rejects_non_synthetic_context_and_invalid_resolution_input(tmp_path: Path) -> None:
    review, _ = _review(tmp_path, [_case("trace-1")])
    real_scenario = GoldenScenario(
        scenario_id="scenario-1", descripcion="Caso", contexto=_context(), mensaje_cliente=None,
        resultado_esperado={},
    )

    with pytest.raises(HumanReviewInputError, match="sintético"):
        review.feedback_to_golden_dataset("trace-1", real_scenario)
    with pytest.raises(HumanReviewInputError, match="agente_id"):
        review.resolve_case("trace-1", "Solución", "gestor con espacios")


class _source:
    def __init__(self, case: EscalationCase) -> None:
        self._case = case

    def cases(self) -> tuple[EscalationCase, ...]:
        return (self._case,)
