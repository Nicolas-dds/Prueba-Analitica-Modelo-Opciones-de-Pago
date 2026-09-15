"""Human review of escalated cases (Requirements 12.1--12.3).

Resolution state is deliberately owned by :class:`HumanReviewInterface`, not by
``EscalationCase`` or ``TraceRecord``: both are immutable audit inputs.  Resolving a
case attempts to append a separate event with a fresh UUID and ``trace_id_origen``.
An append failure does not reopen the case, provided the original record is still
unchanged; the failure is retained for operational inspection.

Golden-scenario feedback uses an injected append-compatible sink.  The interface has
no dependency on the evaluator implementation from task 14.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import re
from typing import Any, Protocol
from uuid import uuid4

from sistema_agentico.guardrails.pii import mask_pii
from sistema_agentico.synthetic_data.profiles import (
    SYNTHETIC_MODEL_VERSION_PREFIX,
    is_synthetic_context,
)
from sistema_agentico.trace import TraceLogger
from sistema_agentico.types import EscalationCase, GoldenScenario, TraceRecord, ValidationResult

__all__ = [
    "CaseAlreadyResolvedError",
    "EscalationCaseSource",
    "GoldenScenarioSink",
    "HumanReviewInputError",
    "HumanReviewInterface",
    "OriginalTraceIntegrityError",
    "UnknownEscalationCaseError",
]


class HumanReviewInputError(ValueError):
    """Raised when a human-review operation receives invalid controlled input."""


class UnknownEscalationCaseError(KeyError):
    """Raised when the requested trace does not belong to the injected case source."""


class CaseAlreadyResolvedError(ValueError):
    """Raised when a resolution would create a duplicate human decision event."""


class OriginalTraceIntegrityError(RuntimeError):
    """Raised if the original trace changed while a resolution was being recorded."""


class EscalationCaseSource(Protocol):
    """Read-only source of escalated cases, compatible with the in-memory sink."""

    def cases(self) -> Sequence[EscalationCase]:
        """Return delivered escalation cases."""


class GoldenScenarioSink(Protocol):
    """Minimal append-compatible abstraction for task 14's dataset implementation."""

    def append(self, scenario: GoldenScenario) -> None:
        """Store one validated, synthetic golden scenario."""


@dataclass(frozen=True)
class _ResolutionStatus:
    """Private state separated from immutable case and trace data."""

    resolution_event_trace_id: str
    logging_error: str | None


_ACTOR_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_MAX_RESOLUTION_LENGTH = 10_000


class HumanReviewInterface:
    """Query, resolve, and convert escalated cases into safe golden feedback.

    ``case_source`` is normally ``InMemoryEscalationCaseSink``.  ``golden_scenario_sink``
    can be a list or a future evaluator-owned collection; it intentionally avoids an
    import of task 14's evaluator.
    """

    def __init__(
        self,
        *,
        trace_logger: TraceLogger,
        case_source: EscalationCaseSource,
        golden_scenario_sink: GoldenScenarioSink | None = None,
    ) -> None:
        self._trace_logger = trace_logger
        self._case_source = case_source
        self._golden_scenario_sink: GoldenScenarioSink = (
            golden_scenario_sink if golden_scenario_sink is not None else []
        )
        self._resolved: dict[str, _ResolutionStatus] = {}
        self._feedback_origins: dict[str, str] = {}

    def get_pending_cases(self) -> list[EscalationCase]:
        """Return every unresolved case in stable, deterministic order (Requirement 12.1)."""
        pending = [case for case in self._case_source.cases() if case.trace_id not in self._resolved]
        return sorted(pending, key=lambda case: (case.trace_id, case.id_obligacion, case.razon.value))

    def resolve_case(self, trace_id: str, resolucion: str, agente_id: str) -> None:
        """Close a case and append a separate human-resolution trace event.

        The source record is never modified.  If appending the new event fails, this
        still closes the case after an integrity comparison confirms that the original
        remains intact, as mandated by Requirement 12.2.
        """
        trace_id = _require_nonempty_text(trace_id, "trace_id")
        resolution = _validate_resolution(resolucion)
        actor_id = _validate_actor_id(agente_id)
        case = self._get_case(trace_id)
        if trace_id in self._resolved:
            raise CaseAlreadyResolvedError(f"El caso '{trace_id}' ya fue resuelto.")

        original = self._get_original_trace(case, trace_id)
        original_snapshot = deepcopy(original)
        event = _build_resolution_event(original, resolution, actor_id)
        logging_error: str | None = None
        try:
            self._trace_logger.append(event)
        except Exception as exc:  # logging is explicitly non-blocking for resolution
            logging_error = f"{type(exc).__name__}: {exc}"

        if original != original_snapshot:
            raise OriginalTraceIntegrityError(
                f"El TraceRecord original '{trace_id}' cambió durante la resolución; el caso no fue cerrado."
            )
        self._resolved[trace_id] = _ResolutionStatus(event.trace_id, logging_error)

    def feedback_to_golden_dataset(self, trace_id: str, scenario: GoldenScenario) -> None:
        """Sanitize and append a synthetic scenario, retaining origin privately (Requirement 12.3)."""
        trace_id = _require_nonempty_text(trace_id, "trace_id")
        self._get_case(trace_id)
        if not isinstance(scenario, GoldenScenario):
            raise HumanReviewInputError("scenario debe ser un GoldenScenario.")
        if not is_synthetic_context(scenario.contexto) or not scenario.contexto.score.version_modelo.startswith(
            SYNTHETIC_MODEL_VERSION_PREFIX
        ):
            raise HumanReviewInputError(
                "El escenario debe usar un contexto sintético marcado con los prefijos SYN- requeridos."
            )

        sanitized = _sanitize_scenario(scenario)
        self._golden_scenario_sink.append(sanitized)
        self._feedback_origins[sanitized.scenario_id] = trace_id

    def resolution_logging_error(self, trace_id: str) -> str | None:
        """Return the non-blocking event logging failure for an already resolved case."""
        trace_id = _require_nonempty_text(trace_id, "trace_id")
        status = self._resolved.get(trace_id)
        if status is None:
            raise UnknownEscalationCaseError(f"El caso '{trace_id}' no está resuelto.")
        return status.logging_error

    def feedback_origin_trace_id(self, scenario_id: str) -> str:
        """Return private traceability metadata without embedding it in the scenario."""
        scenario_id = _require_nonempty_text(scenario_id, "scenario_id")
        try:
            return self._feedback_origins[scenario_id]
        except KeyError as exc:
            raise KeyError(f"No existe feedback para scenario_id='{scenario_id}'.") from exc

    def _get_case(self, trace_id: str) -> EscalationCase:
        matches = [case for case in self._case_source.cases() if case.trace_id == trace_id]
        if not matches:
            raise UnknownEscalationCaseError(f"No existe un caso escalado con trace_id='{trace_id}'.")
        return matches[0]

    def _get_original_trace(self, case: EscalationCase, trace_id: str) -> TraceRecord:
        original = self._trace_logger.get_trace(trace_id)
        if original.trace_id != trace_id or original.id_obligacion != case.id_obligacion:
            raise HumanReviewInputError("El TraceRecord original no corresponde al caso escalado.")
        return original


def _require_nonempty_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HumanReviewInputError(f"{name} debe ser un texto no vacío.")
    return value.strip()


def _validate_resolution(value: object) -> str:
    resolution = _require_nonempty_text(value, "resolucion")
    if len(resolution) > _MAX_RESOLUTION_LENGTH:
        raise HumanReviewInputError(f"resolucion no puede exceder {_MAX_RESOLUTION_LENGTH} caracteres.")
    return mask_pii(resolution)[0] or ""


def _validate_actor_id(value: object) -> str:
    actor_id = _require_nonempty_text(value, "agente_id")
    if not _ACTOR_ID_PATTERN.fullmatch(actor_id):
        raise HumanReviewInputError(
            "agente_id debe contener únicamente letras, números, puntos, guiones o guiones bajos."
        )
    return actor_id


def _build_resolution_event(original: TraceRecord, resolution: str, actor_id: str) -> TraceRecord:
    """Build a new append-only event without retaining mutable references to the source."""
    rules = deepcopy(original.reglas_evaluadas)
    rules["trace_id_origen"] = original.trace_id
    rules["human_resolution"] = resolution
    rules["resolution_event"] = "case_resolved"
    return TraceRecord(
        trace_id=str(uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        id_obligacion=original.id_obligacion,
        modo=original.modo,
        score_info=deepcopy(original.score_info),
        reglas_evaluadas=rules,
        eligibility_result=deepcopy(original.eligibility_result),
        whitelist=deepcopy(original.whitelist),
        ofertas_mencionadas=[],
        respuesta_agente=resolution,
        agente_responsable=f"human:{actor_id}",
        validation_result=ValidationResult(
            valido=True,
            motivo_rechazo=None,
            escalar=False,
            razon_escalamiento=None,
        ),
        prompt_version="human-review-v1",
        version_modelo_propension=original.version_modelo_propension,
    )


def _sanitize_scenario(scenario: GoldenScenario) -> GoldenScenario:
    """Return an anonymized copy; original free-text values never enter the dataset."""
    return replace(
        scenario,
        descripcion=_sanitize_text(scenario.descripcion),
        mensaje_cliente=(
            _sanitize_text(scenario.mensaje_cliente) if scenario.mensaje_cliente is not None else None
        ),
        resultado_esperado=_sanitize_value(scenario.resultado_esperado),
    )


def _sanitize_text(value: str) -> str:
    return mask_pii(value)[0] or ""


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, str):
        return _sanitize_text(value)
    if isinstance(value, Mapping):
        return {_sanitize_text(str(key)): _sanitize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_value(item) for item in value)
    return deepcopy(value)
