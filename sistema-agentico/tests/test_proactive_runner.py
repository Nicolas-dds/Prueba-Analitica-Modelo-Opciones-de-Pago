"""Focused tests for independent prioritized proactive batch processing."""
from __future__ import annotations

from sistema_agentico.orchestration import (
    InteractionOutcome,
    InteractionStatus,
    ProactiveBatchRunner,
)
from sistema_agentico.types import AgentResponse, InputMode, RawInput, SanitizedInput


class _InputGuardrail:
    def __init__(self, failing_ids: set[str] | None = None) -> None:
        self.failing_ids = failing_ids or set()
        self.raw_inputs: list[RawInput] = []

    def process(self, raw_input: RawInput) -> SanitizedInput:
        self.raw_inputs.append(raw_input)
        if raw_input.id_obligacion in self.failing_ids:
            raise RuntimeError("guardrail unavailable")
        return SanitizedInput(
            mode=raw_input.mode,
            id_obligacion=raw_input.id_obligacion,
            mensaje_sanitizado=None,
            pii_detectada=[],
            riesgo_inyeccion=0.0,
            bloqueado=False,
        )


class _Orchestrator:
    def __init__(self, failing_ids: set[str] | None = None) -> None:
        self.failing_ids = failing_ids or set()
        self.inputs: list[SanitizedInput] = []

    def handle_interaction(self, sanitized_input: SanitizedInput) -> InteractionOutcome:
        self.inputs.append(sanitized_input)
        if sanitized_input.id_obligacion in self.failing_ids:
            raise RuntimeError("component failure")
        return InteractionOutcome(
            trace_id=f"trace-{sanitized_input.id_obligacion}",
            status=InteractionStatus.FINAL,
            response=AgentResponse("Gestión proactiva", [], False, "prompt-v1"),
            escalation_reason=None,
            escalation_case=None,
            trace_logged=True,
        )


def test_runner_preserves_order_and_continues_after_an_orchestrator_failure() -> None:
    guardrail = _InputGuardrail()
    orchestrator = _Orchestrator(failing_ids={"OBL-2"})

    results = ProactiveBatchRunner(guardrail, orchestrator).run(["OBL-1", "OBL-2", "OBL-3"])

    assert [result.id_obligacion for result in results] == ["OBL-1", "OBL-2", "OBL-3"]
    assert [raw.id_obligacion for raw in guardrail.raw_inputs] == ["OBL-1", "OBL-2", "OBL-3"]
    assert all(raw.mode is InputMode.PROACTIVO for raw in guardrail.raw_inputs)
    assert all(raw.mensaje_crudo is None for raw in guardrail.raw_inputs)
    assert [item.id_obligacion for item in orchestrator.inputs] == ["OBL-1", "OBL-2", "OBL-3"]
    assert results[0].completed is True
    assert results[0].outcome is not None
    assert results[0].outcome.trace_id == "trace-OBL-1"
    assert results[1].completed is False
    assert results[1].error == "Orchestrator failed: RuntimeError: component failure"
    assert results[2].completed is True
    assert results[2].outcome is not None
    assert results[2].outcome.trace_id == "trace-OBL-3"


def test_guardrail_failure_is_local_and_does_not_bypass_later_items() -> None:
    guardrail = _InputGuardrail(failing_ids={"OBL-2"})
    orchestrator = _Orchestrator()

    results = ProactiveBatchRunner(guardrail, orchestrator).run(["OBL-1", "OBL-2", "OBL-3"])

    assert [raw.id_obligacion for raw in guardrail.raw_inputs] == ["OBL-1", "OBL-2", "OBL-3"]
    assert [item.id_obligacion for item in orchestrator.inputs] == ["OBL-1", "OBL-3"]
    assert results[1].outcome is None
    assert results[1].error == "InputGuardrail failed: RuntimeError: guardrail unavailable"
    assert results[0].completed is results[2].completed is True
