"""Focused safety coverage for ``GoldenDatasetEvaluator.run``."""
from __future__ import annotations

from dataclasses import replace

from sistema_agentico.evaluation import EvaluationExecution, GoldenDatasetEvaluator
from sistema_agentico.orchestration import InteractionOutcome, InteractionStatus
from sistema_agentico.types import (
    AgentResponse,
    ClienteObligacionContext,
    EligibilityResult,
    EscalationReason,
    GoldenScenario,
    InputMode,
    OfertaElegible,
    ScoreInfo,
    TraceRecord,
    TipoOferta,
    ValidationResult,
    WhitelistItem,
)


def _context(identifier: str) -> ClienteObligacionContext:
    return ClienteObligacionContext(
        id_obligacion=identifier,
        id_cliente=f"SYN-CLIENTE-{identifier}",
        dias_mora=10,
        exposicion=1000.0,
        segmento="synthetic",
        canal_gestion="test",
        score=ScoreInfo(0.6, 6, "SYN-model-v1", "2026-01-01", False),
        ofertas_aplicadas_mes=[],
        ultima_opcion_aplicada=None,
        acuerdo_pago_vigente=False,
        restriccion_vigente=False,
    )


def _scenario(
    identifier: str,
    *,
    mode: InputMode = InputMode.REACTIVO,
    escalation: dict[str, object] | None = None,
) -> GoldenScenario:
    return GoldenScenario(
        scenario_id=identifier,
        descripcion="Escenario sintético de evaluación.",
        contexto=_context(identifier),
        mensaje_cliente="Necesito opciones" if mode is InputMode.REACTIVO else None,
        resultado_esperado={
            "modo_esperado": mode.value,
            "escalamiento_esperado": escalation or {"requerido": False, "razones_aceptadas": []},
        },
    )


def _final(offers: list[str]) -> InteractionOutcome:
    return InteractionOutcome(
        trace_id="trace-test",
        status=InteractionStatus.FINAL,
        response=AgentResponse("Respuesta", offers, False, "prompt-test"),
        escalation_reason=None,
        escalation_case=None,
        trace_logged=True,
    )


def _trace(outcome: InteractionOutcome, whitelist_ids: list[str]) -> TraceRecord:
    offer = OfertaElegible(TipoOferta.OPCION_PAGO, "oferta-base", {}, "Elegible en prueba")
    whitelist = [
        WhitelistItem(
            OfertaElegible(TipoOferta.OPCION_PAGO, offer_id, {}, "Autorizada en prueba"),
            index,
            "Justificación de prueba",
        )
        for index, offer_id in enumerate(whitelist_ids, start=1)
    ]
    context = _context("trace")
    return TraceRecord(
        trace_id=outcome.trace_id,
        timestamp="2026-01-01T00:00:00+00:00",
        id_obligacion=context.id_obligacion,
        modo=InputMode.REACTIVO,
        score_info=context.score,
        reglas_evaluadas={},
        eligibility_result=EligibilityResult([offer], None, None),
        whitelist=whitelist,
        ofertas_mencionadas=list(outcome.response.ofertas_mencionadas) if outcome.response else [],
        respuesta_agente=outcome.response.texto if outcome.response else "",
        agente_responsable="conversational_agent",
        validation_result=ValidationResult(True, None, False, None),
        prompt_version="prompt-test",
        version_modelo_propension=context.score.version_modelo,
    )


class _Pipeline:
    def __init__(self, executions: dict[str, EvaluationExecution | Exception]) -> None:
        self.executions = executions
        self.calls: list[tuple[str, str]] = []

    def execute_reactive(self, id_obligacion: str, mensaje_cliente: str) -> EvaluationExecution:
        self.calls.append(("reactivo", id_obligacion))
        return self._result(id_obligacion)

    def execute_proactive(self, id_obligacion: str) -> EvaluationExecution:
        self.calls.append(("proactivo", id_obligacion))
        return self._result(id_obligacion)

    def _result(self, id_obligacion: str) -> EvaluationExecution:
        result = self.executions[id_obligacion]
        if isinstance(result, Exception):
            raise result
        return result


def _evaluation_outcome(*, approved: bool) -> object:
    """Create the minimal valid input required by the promotion gate."""
    from sistema_agentico.types import EvaluationOutcome

    return EvaluationOutcome(
        scenario_id="SYN-gate",
        metrica="ofertas_invalidas",
        umbral_aceptacion=0.0,
        resultado=0.0 if approved else 1.0,
        aprobado=approved,
        oportunidad_mejora=None if approved else "Escenario rechazado.",
    )


def test_gate_allows_promotion_only_when_every_outcome_is_approved() -> None:
    evaluator = GoldenDatasetEvaluator()

    assert evaluator.gate([_evaluation_outcome(approved=True), _evaluation_outcome(approved=True)]) is True
    assert evaluator.gate([_evaluation_outcome(approved=True), _evaluation_outcome(approved=False)]) is False


def test_gate_fails_closed_for_empty_or_malformed_evaluations() -> None:
    evaluator = GoldenDatasetEvaluator()

    assert evaluator.gate([]) is False
    assert evaluator.gate([_evaluation_outcome(approved=True), object()]) is False  # type: ignore[list-item]


def test_run_accepts_audited_final_response_without_invalid_offers() -> None:
    scenario = _scenario("SYN-valid")
    outcome = _final(["ampliacion_plazo"])
    pipeline = _Pipeline(
        {scenario.contexto.id_obligacion: EvaluationExecution(outcome, _trace(outcome, ["ampliacion_plazo"]))}
    )

    result = GoldenDatasetEvaluator().run([scenario], pipeline)

    assert result == [
        result[0].__class__(
            scenario_id="SYN-valid",
            metrica="ofertas_invalidas",
            umbral_aceptacion=0.0,
            resultado=0.0,
            aprobado=True,
            oportunidad_mejora=None,
        )
    ]
    assert pipeline.calls == [("reactivo", scenario.contexto.id_obligacion)]


def test_run_fails_non_negotiable_metric_when_final_offer_is_not_whitelisted() -> None:
    scenario = _scenario("SYN-invalid")
    outcome = _final(["oferta_no_autorizada"])
    pipeline = _Pipeline(
        {scenario.contexto.id_obligacion: EvaluationExecution(outcome, _trace(outcome, ["ampliacion_plazo"]))}
    )

    result = GoldenDatasetEvaluator().run([scenario], pipeline)[0]

    assert (result.metrica, result.umbral_aceptacion, result.resultado, result.aprobado) == (
        "ofertas_invalidas", 0.0, 1.0, False
    )
    assert result.oportunidad_mejora is not None
    assert "resultado=1" in result.oportunidad_mejora


def test_run_fails_when_expected_escalation_does_not_occur_even_without_invalid_offers() -> None:
    scenario = _scenario(
        "SYN-escalation",
        escalation={"requerido": True, "razones_aceptadas": [EscalationReason.INFO_CONTRADICTORIA.value]},
    )
    outcome = _final([])
    pipeline = _Pipeline({scenario.contexto.id_obligacion: EvaluationExecution(outcome, _trace(outcome, []))})

    result = GoldenDatasetEvaluator().run([scenario], pipeline)[0]

    assert result.resultado == 0.0
    assert result.aprobado is False
    assert result.oportunidad_mejora is not None
    assert "escalamiento esperado=sí" in result.oportunidad_mejora


def test_run_isolates_pipeline_exceptions_and_routes_declared_proactive_mode() -> None:
    failed = _scenario("SYN-failure")
    succeeding = _scenario("SYN-proactive", mode=InputMode.PROACTIVO)
    safe_outcome = replace(_final([]), trace_id="trace-proactive")
    pipeline = _Pipeline(
        {
            failed.contexto.id_obligacion: RuntimeError("runner unavailable"),
            succeeding.contexto.id_obligacion: EvaluationExecution(safe_outcome, _trace(safe_outcome, [])),
        }
    )

    results = GoldenDatasetEvaluator().run([failed, succeeding], pipeline)

    assert [(result.scenario_id, result.resultado, result.aprobado) for result in results] == [
        ("SYN-failure", 1.0, False),
        ("SYN-proactive", 0.0, True),
    ]
    assert results[0].oportunidad_mejora is not None
    assert "RuntimeError: runner unavailable" in results[0].oportunidad_mejora
    assert pipeline.calls == [("reactivo", failed.contexto.id_obligacion), ("proactivo", succeeding.contexto.id_obligacion)]
