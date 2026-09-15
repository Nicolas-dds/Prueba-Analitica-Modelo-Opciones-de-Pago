"""Focused tests for escalation policy and independent case delivery (Requirement 9)."""
from __future__ import annotations

import pytest

from sistema_agentico.escalation import (
    EscalationManager,
    EscalationTrigger,
    InMemoryEscalationCaseSink,
    UnsupportedEscalationTriggerError,
    map_trigger_to_reason,
)
from sistema_agentico.types import (
    ClienteObligacionContext,
    EligibilityResult,
    EscalationReason,
    InputMode,
    ScoreInfo,
    TraceRecord,
    ValidationResult,
)


def _context() -> ClienteObligacionContext:
    return ClienteObligacionContext(
        id_obligacion="OBL-1",
        id_cliente="CLI-1",
        dias_mora=12,
        exposicion=500_000.0,
        segmento="masivo",
        canal_gestion="telefono",
        score=ScoreInfo(0.5, 5, "score-v1", "2025-01-01", False),
        ofertas_aplicadas_mes=[],
        ultima_opcion_aplicada=None,
        acuerdo_pago_vigente=False,
        restriccion_vigente=False,
    )


def _trace(trace_id: str = "trace-1") -> TraceRecord:
    context = _context()
    return TraceRecord(
        trace_id=trace_id,
        timestamp="2025-01-01T00:00:00+00:00",
        id_obligacion=context.id_obligacion,
        modo=InputMode.REACTIVO,
        score_info=context.score,
        reglas_evaluadas={"regla": "aplicada"},
        eligibility_result=EligibilityResult([], None, "Sin oferta para esta prueba."),
        whitelist=[],
        ofertas_mencionadas=[],
        respuesta_agente="Respuesta rechazada",
        agente_responsable="conversational_agent",
        validation_result=ValidationResult(False, "Oferta inválida", False, None),
        prompt_version="test-v1",
        version_modelo_propension="score-v1",
    )


class _TraceLogger:
    def __init__(self, record: TraceRecord | None = None, error: Exception | None = None) -> None:
        self.record = record
        self.error = error

    def get_trace(self, trace_id: str) -> TraceRecord:
        if self.error is not None:
            raise self.error
        assert self.record is not None
        assert trace_id == self.record.trace_id
        return self.record


class _FailingSink:
    def deliver(self, case: object) -> None:
        del case
        raise RuntimeError("cola humana no disponible")


def test_trigger_mapping_is_total_and_rejects_unknown_values() -> None:
    expected = {trigger.value for trigger in EscalationTrigger}
    assert set(EscalationReason(reason).value for reason in expected) == expected
    assert {map_trigger_to_reason(trigger).value for trigger in EscalationTrigger} == expected
    with pytest.raises(UnsupportedEscalationTriggerError):
        map_trigger_to_reason("technical_failure")  # type: ignore[arg-type]


def test_should_escalate_allows_one_unauthorized_offer_retry_then_escalates() -> None:
    manager = EscalationManager()
    invalid_offer = ValidationResult(False, "Oferta fuera de whitelist", False, None)

    assert manager.should_escalate(invalid_offer, intento_num=0) is False
    assert manager.resolve_escalation_reason(invalid_offer, intento_num=0) is None
    assert manager.should_escalate(invalid_offer, intento_num=1) is True
    assert manager.resolve_escalation_reason(invalid_offer, intento_num=1) is EscalationReason.REINTENTO_FALLIDO


@pytest.mark.parametrize(
    "reason",
    [
        EscalationReason.MANIPULACION_DETECTADA,
        EscalationReason.INFO_CONTRADICTORIA,
        EscalationReason.DIFICULTAD_SEVERA,
        EscalationReason.FUERA_DE_ALCANCE,
        EscalationReason.FALLO_TECNICO,
    ],
)
def test_should_escalate_direct_reasons_without_retry(reason: EscalationReason) -> None:
    validation = ValidationResult(False, "Señal directa", True, reason)
    manager = EscalationManager()
    assert manager.should_escalate(validation, intento_num=0) is True
    assert manager.resolve_escalation_reason(validation, intento_num=0) is reason


def test_escalate_delivers_full_trace_and_sanitized_history() -> None:
    sink = InMemoryEscalationCaseSink()
    manager = EscalationManager(trace_logger=_TraceLogger(_trace()), case_sink=sink)

    outcome = manager.escalate_with_outcome(
        _context(),
        EscalationReason.DIFICULTAD_SEVERA,
        "trace-1",
        ["Mi cédula es 12345678", {"cliente": "me llamo Juan Pérez"}],
    )

    assert outcome.trace_available is True
    assert outcome.delivery_succeeded is True
    assert outcome.case.trace_completo == _trace()
    assert outcome.case.historial_conversacion == ["Mi cédula es [CEDULA]", {"cliente": "me llamo [NOMBRE]"}]
    assert sink.cases() == (outcome.case,)


def test_delivery_failure_preserves_case_and_reports_outcome() -> None:
    manager = EscalationManager(
        trace_logger=_TraceLogger(error=RuntimeError("logger caído")),
        case_sink=_FailingSink(),
    )

    outcome = manager.escalate_with_outcome(
        _context(), EscalationReason.MANIPULACION_DETECTADA, "blocked-trace"
    )

    assert outcome.case.trace_id == "blocked-trace"
    assert outcome.case.razon is EscalationReason.MANIPULACION_DETECTADA
    assert outcome.trace_available is False
    assert "logger caído" in outcome.trace_error  # type: ignore[operator]
    assert outcome.delivery_succeeded is False
    assert "cola humana no disponible" in outcome.delivery_error  # type: ignore[operator]
    assert outcome.case.trace_completo.reglas_evaluadas["escalation_trace_status"] == "unavailable"
