"""Deterministic golden-scenario evaluation over the real runner boundaries.

The evaluator deliberately does not judge generated prose.  Its release-critical
measurement compares structured offer identifiers from the terminal response with
the whitelist preserved in the interaction trace.  Escalation expectations are
checked separately, so a safely escalated interaction with no final response does
not become an invalid-offer failure.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from sistema_agentico.orchestration import InteractionOutcome, InteractionStatus
from sistema_agentico.types import EvaluationOutcome, GoldenScenario, InputMode, TraceRecord

__all__ = [
    "EvaluationExecution",
    "GoldenDatasetEvaluator",
    "GoldenScenarioPipeline",
    "RunnerPipelineAdapter",
]


@dataclass(frozen=True)
class EvaluationExecution:
    """Auditable terminal result returned by a golden-scenario pipeline.

    ``trace_record`` is the preferred authorization evidence.  Lightweight test
    pipelines may instead provide ``authorized_offer_ids``.  When neither is
    available, a final response that names offers fails closed because its
    authorization cannot be verified; a response naming no offers remains safe.
    """

    outcome: InteractionOutcome
    trace_record: TraceRecord | None = None
    authorized_offer_ids: frozenset[str] | None = None


class GoldenScenarioPipeline(Protocol):
    """Small injectable boundary for executing both declared scenario modes."""

    def execute_reactive(self, id_obligacion: str, mensaje_cliente: str) -> EvaluationExecution:
        """Run raw reactive input through ``ReactiveRunner`` and return audit evidence."""

    def execute_proactive(self, id_obligacion: str) -> EvaluationExecution:
        """Run one proactive item through ``ProactiveBatchRunner`` and return evidence."""


class RunnerPipelineAdapter:
    """Adapt the production reactive/proactive runners to ``GoldenScenarioPipeline``.

    Passing the same ``TraceLogger`` used by the orchestrator makes the evaluator
    compare the terminal response IDs with the persisted whitelist.  A trace lookup
    failure is intentionally non-fatal here: the evaluator then fails closed only
    if the terminal response mentions one or more offers.
    """

    def __init__(
        self,
        reactive_runner: Any,
        proactive_runner: Any,
        *,
        trace_logger: Any | None = None,
    ) -> None:
        self._reactive_runner = reactive_runner
        self._proactive_runner = proactive_runner
        self._trace_logger = trace_logger

    def execute_reactive(self, id_obligacion: str, mensaje_cliente: str) -> EvaluationExecution:
        outcome = self._reactive_runner.run(id_obligacion, mensaje_cliente)
        return EvaluationExecution(outcome=outcome, trace_record=self._trace_for(outcome))

    def execute_proactive(self, id_obligacion: str) -> EvaluationExecution:
        items = self._proactive_runner.run([id_obligacion])
        if len(items) != 1:
            raise RuntimeError("ProactiveBatchRunner debe retornar exactamente un resultado por escenario")
        item = items[0]
        if item.outcome is None:
            raise RuntimeError(item.error or "El runner proactivo no produjo un resultado")
        return EvaluationExecution(outcome=item.outcome, trace_record=self._trace_for(item.outcome))

    def _trace_for(self, outcome: InteractionOutcome) -> TraceRecord | None:
        if self._trace_logger is None:
            return None
        try:
            trace = self._trace_logger.get_trace(outcome.trace_id)
        except Exception:
            return None
        return trace if isinstance(trace, TraceRecord) and trace.trace_id == outcome.trace_id else None


class GoldenDatasetEvaluator:
    """Evaluate structured release-safety evidence for each golden scenario.

    ``run`` returns exactly one ``ofertas_invalidas`` outcome per supplied scenario.
    Its non-negotiable threshold is always zero.  A scenario passes only when that
    count is zero *and* its declarative escalation expectation is satisfied:

    * ``requerido=True`` requires escalation, optionally restricted to
      ``razones_aceptadas``.
    * ``requerido=False`` requires no escalation, except an escalation matching a
      declared accepted reason or the declared conditional reason is allowed.

    The evaluator does not mutate scenarios and captures each pipeline error as a
    failed outcome so later scenarios still execute.
    """

    INVALID_OFFERS_METRIC = "ofertas_invalidas"
    INVALID_OFFERS_THRESHOLD = 0.0

    def run(
        self,
        scenarios: list[GoldenScenario],
        agent_pipeline: GoldenScenarioPipeline,
    ) -> list[EvaluationOutcome]:
        outcomes: list[EvaluationOutcome] = []
        for scenario in scenarios:
            try:
                execution = self._execute(scenario, agent_pipeline)
                invalid_offers, evidence_note = self._invalid_offer_count(execution)
                offers_ok, offers_note = self._matches_offer_expectation(scenario, execution)
                escalation_ok, escalation_note = self._matches_escalation_expectation(
                    scenario, execution.outcome
                )
                approved = invalid_offers == 0 and offers_ok and escalation_ok
                improvement = None
                if not approved:
                    details = [
                        f"ofertas inválidas esperadas=0, resultado={invalid_offers}",
                    ]
                    if evidence_note:
                        details.append(evidence_note)
                    if not offers_ok:
                        details.append(offers_note)
                    if not escalation_ok:
                        details.append(escalation_note)
                    improvement = "; ".join(details)
                outcomes.append(
                    EvaluationOutcome(
                        scenario_id=scenario.scenario_id,
                        metrica=self.INVALID_OFFERS_METRIC,
                        umbral_aceptacion=self.INVALID_OFFERS_THRESHOLD,
                        resultado=float(invalid_offers),
                        aprobado=approved,
                        oportunidad_mejora=improvement,
                    )
                )
            except Exception as exc:
                outcomes.append(
                    EvaluationOutcome(
                        scenario_id=scenario.scenario_id,
                        metrica=self.INVALID_OFFERS_METRIC,
                        umbral_aceptacion=self.INVALID_OFFERS_THRESHOLD,
                        resultado=1.0,
                        aprobado=False,
                        oportunidad_mejora=(
                            "Ejecución del pipeline falló "
                            f"({type(exc).__name__}: {exc}); ofertas inválidas esperadas=0."
                        ),
                    )
                )
        return outcomes

    @staticmethod
    def gate(outcomes: list[EvaluationOutcome]) -> bool:
        """Allow promotion only when a non-empty evaluation fully passes.

        The gate is fail-closed: an empty evaluation cannot demonstrate release
        safety, and malformed entries are treated as failed evaluations.  This
        method only inspects its input; it neither mutates outcomes nor invokes
        pipeline, network, or LLM dependencies.
        """
        return bool(outcomes) and all(
            isinstance(outcome, EvaluationOutcome) and outcome.aprobado is True
            for outcome in outcomes
        )

    @staticmethod
    def _execute(
        scenario: GoldenScenario,
        agent_pipeline: GoldenScenarioPipeline,
    ) -> EvaluationExecution:
        expected_mode = GoldenDatasetEvaluator._expected_mode(scenario)
        if expected_mode is InputMode.REACTIVO:
            if not isinstance(scenario.mensaje_cliente, str):
                raise ValueError("un escenario reactivo requiere mensaje_cliente str")
            execution = agent_pipeline.execute_reactive(
                scenario.contexto.id_obligacion, scenario.mensaje_cliente
            )
        else:
            execution = agent_pipeline.execute_proactive(scenario.contexto.id_obligacion)
        if not isinstance(execution, EvaluationExecution):
            raise TypeError("el pipeline debe retornar EvaluationExecution")
        return execution

    @staticmethod
    def _expected_mode(scenario: GoldenScenario) -> InputMode:
        value = scenario.resultado_esperado.get("modo_esperado")
        if isinstance(value, InputMode):
            return value
        try:
            return InputMode(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("resultado_esperado.modo_esperado debe ser proactivo o reactivo") from exc

    @staticmethod
    def _invalid_offer_count(execution: EvaluationExecution) -> tuple[int, str | None]:
        response = execution.outcome.response
        if response is None:
            # An escalation has no customer-facing final response, therefore no final
            # offer was delivered.  Escalation correctness is evaluated independently.
            return 0, None

        mentioned = set(response.ofertas_mencionadas)
        trace = execution.trace_record
        if trace is not None and trace.trace_id == execution.outcome.trace_id:
            authorized = {item.oferta.id_opcion for item in trace.whitelist}
            return len(mentioned - authorized), None
        if execution.authorized_offer_ids is not None:
            return len(mentioned - execution.authorized_offer_ids), None
        if not mentioned:
            return 0, None
        # No raw-text parsing is attempted.  Without a whitelist, each named ID is
        # conservatively counted as unverified/invalid rather than assumed safe.
        return len(mentioned), "No hubo whitelist auditable para verificar los IDs finales."

    @staticmethod
    def _matches_offer_expectation(
        scenario: GoldenScenario,
        execution: EvaluationExecution,
    ) -> tuple[bool, str]:
        """Check only structured, deterministic offer expectations.

        ``debe_incluir`` and ``debe_excluir`` apply to terminal response IDs.
        ``debe_ser_whitelist_vacia`` is checked only against auditable authorization
        evidence, never response prose.  Other catalogue annotations describe E2E
        behavior for task 15 and are intentionally not inferred here.
        """
        expected = scenario.resultado_esperado.get("ofertas_esperadas", {})
        if not isinstance(expected, Mapping):
            raise ValueError("ofertas_esperadas debe ser un mapping")
        mentioned = set(execution.outcome.response.ofertas_mencionadas) if execution.outcome.response else set()
        required = {str(offer_id) for offer_id in expected.get("debe_incluir", [])}
        prohibited = {str(offer_id) for offer_id in expected.get("debe_excluir", [])}
        missing = required - mentioned
        present_prohibited = prohibited & mentioned
        details: list[str] = []
        if missing:
            details.append(f"ofertas esperadas ausentes={sorted(missing)}; actual={sorted(mentioned)}")
        if present_prohibited:
            details.append(f"ofertas esperadas excluidas presentes={sorted(present_prohibited)}")

        if expected.get("debe_ser_whitelist_vacia"):
            if execution.trace_record is not None and execution.trace_record.trace_id == execution.outcome.trace_id:
                authorized = {item.oferta.id_opcion for item in execution.trace_record.whitelist}
            else:
                authorized = execution.authorized_offer_ids
            if authorized is None:
                details.append("se esperaba whitelist vacía, pero no hubo evidencia auditable de whitelist")
            elif authorized:
                details.append(f"se esperaba whitelist vacía; actual={sorted(authorized)}")
        return not details, "; ".join(details)

    @staticmethod
    def _matches_escalation_expectation(
        scenario: GoldenScenario,
        outcome: InteractionOutcome,
    ) -> tuple[bool, str]:
        expected = scenario.resultado_esperado.get("escalamiento_esperado", {})
        if not isinstance(expected, Mapping):
            raise ValueError("escalamiento_esperado debe ser un mapping")
        required = bool(expected.get("requerido", False))
        actual_escalated = outcome.status is InteractionStatus.ESCALATED
        actual_reason = outcome.escalation_reason.value if outcome.escalation_reason else None
        accepted = {str(reason) for reason in expected.get("razones_aceptadas", [])}
        conditional = expected.get("condicional")
        conditional_reason = (
            str(conditional["razon"])
            if isinstance(conditional, Mapping) and conditional.get("razon") is not None
            else None
        )

        if required:
            ok = actual_escalated and (not accepted or actual_reason in accepted)
            if ok:
                return True, ""
            expected_reason = f" con razón en {sorted(accepted)}" if accepted else ""
            return False, f"escalamiento esperado=sí{expected_reason}; actual={actual_reason or 'no escalado'}"

        if not actual_escalated:
            return True, ""
        allowed_reasons = accepted | ({conditional_reason} if conditional_reason else set())
        if actual_reason in allowed_reasons:
            return True, ""
        expected_text = "no escalado"
        if allowed_reasons:
            expected_text += f" o razón en {sorted(allowed_reasons)}"
        return False, f"escalamiento esperado={expected_text}; actual={actual_reason or 'escalado sin razón'}"
