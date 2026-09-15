"""Política y entrega de escalamiento determinista (Requirements 9.1--9.6).

``intento_num`` usa una convención de índice base cero: ``0`` es la generación
inicial y ``1`` es el único reintento autorizado. Por tanto, una validación de
oferta no autorizada en el intento 0 permite la corrección, y la misma
validación en el intento 1 se convierte en ``REINTENTO_FALLIDO``.

El manager no invoca LLMs ni implementa la interfaz humana. Solo entrega un
caso a un ``EscalationCaseSink`` inyectable. La obtención de trazabilidad y la
entrega son operaciones independientes: ambas se intentan y sus resultados se
exponen mediante ``EscalationOutcome``.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol

from sistema_agentico.conversational import ConversationTurn
from sistema_agentico.guardrails.pii import mask_pii
from sistema_agentico.trace import TraceLogger
from sistema_agentico.types import (
    ClienteObligacionContext,
    EligibilityResult,
    EscalationCase,
    EscalationReason,
    InputMode,
    ScoreInfo,
    TraceRecord,
    ValidationResult,
)

__all__ = [
    "EscalationCaseSink",
    "EscalationManager",
    "EscalationOutcome",
    "EscalationTrigger",
    "InMemoryEscalationCaseSink",
    "UnsupportedEscalationTriggerError",
    "map_trigger_to_reason",
]


class EscalationTrigger(Enum):
    """Señales estables que el futuro orquestador puede traducir sin ambigüedad."""

    OFERTA_NO_AUTORIZADA = "oferta_no_autorizada"
    MANIPULACION_DETECTADA = "manipulacion_detectada"
    INFO_CONTRADICTORIA = "info_contradictoria"
    DIFICULTAD_SEVERA = "dificultad_severa"
    FUERA_DE_ALCANCE = "fuera_de_alcance"
    FALLO_TECNICO = "fallo_tecnico"
    REINTENTO_FALLIDO = "reintento_fallido"


_TRIGGER_TO_REASON: Mapping[EscalationTrigger, EscalationReason] = {
    EscalationTrigger.OFERTA_NO_AUTORIZADA: EscalationReason.OFERTA_NO_AUTORIZADA,
    EscalationTrigger.MANIPULACION_DETECTADA: EscalationReason.MANIPULACION_DETECTADA,
    EscalationTrigger.INFO_CONTRADICTORIA: EscalationReason.INFO_CONTRADICTORIA,
    EscalationTrigger.DIFICULTAD_SEVERA: EscalationReason.DIFICULTAD_SEVERA,
    EscalationTrigger.FUERA_DE_ALCANCE: EscalationReason.FUERA_DE_ALCANCE,
    EscalationTrigger.FALLO_TECNICO: EscalationReason.FALLO_TECNICO,
    EscalationTrigger.REINTENTO_FALLIDO: EscalationReason.REINTENTO_FALLIDO,
}


class UnsupportedEscalationTriggerError(ValueError):
    """La señal no pertenece al contrato total y determinista del manager."""


def map_trigger_to_reason(trigger: EscalationTrigger) -> EscalationReason:
    """Mapea una señal tipada a su motivo de escalamiento de forma total.

    Los valores no tipados se rechazan explícitamente: no se hace inferencia por
    texto que pudiera clasificar un caso en una razón incorrecta.
    """
    if not isinstance(trigger, EscalationTrigger):
        raise UnsupportedEscalationTriggerError(
            f"Señal de escalamiento no soportada: {trigger!r}. "
            "Use un valor de EscalationTrigger."
        )
    return _TRIGGER_TO_REASON[trigger]


class EscalationCaseSink(Protocol):
    """Adaptador mínimo para entregar casos; no sustituye HumanReviewInterface."""

    def deliver(self, case: EscalationCase) -> None:
        """Entrega un caso para revisión humana o su cola intermedia."""


class InMemoryEscalationCaseSink:
    """Sink local inyectable para pruebas y ejecución autocontenida."""

    def __init__(self) -> None:
        self._cases: list[EscalationCase] = []

    def deliver(self, case: EscalationCase) -> None:
        self._cases.append(case)

    def cases(self) -> tuple[EscalationCase, ...]:
        """Vista inmutable de los casos entregados, en orden de entrega."""
        return tuple(self._cases)


@dataclass(frozen=True)
class EscalationOutcome:
    """Resultado no bloqueante de enriquecer y entregar un caso de escalamiento."""

    case: EscalationCase
    trace_available: bool
    delivery_succeeded: bool
    trace_error: str | None = None
    delivery_error: str | None = None


_IMMEDIATE_REASONS = frozenset(
    {
        EscalationReason.MANIPULACION_DETECTADA,
        EscalationReason.INFO_CONTRADICTORIA,
        EscalationReason.DIFICULTAD_SEVERA,
        EscalationReason.FUERA_DE_ALCANCE,
        EscalationReason.FALLO_TECNICO,
        EscalationReason.REINTENTO_FALLIDO,
    }
)


class EscalationManager:
    """Aplica la política de reintento único y conserva la entrega independiente.

    ``trace_logger`` es opcional porque las rutas de manipulación o contradicción
    pueden escalar antes de que exista un registro append-only. Si falta o falla,
    se crea un ``TraceRecord`` marcador inmutable y ``EscalationOutcome`` informa
    el error; nunca se modifica un registro existente.
    """

    def __init__(
        self,
        *,
        trace_logger: TraceLogger | None = None,
        case_sink: EscalationCaseSink | None = None,
    ) -> None:
        self._trace_logger = trace_logger
        self._case_sink = case_sink

    def should_escalate(self, validation: ValidationResult, intento_num: int) -> bool:
        """Indica si una validación requiere escalar bajo la convención 0/1.

        Razones severas/directas se escalan en cualquier intento. Una invalidación
        normal del ``OutputGuardrail`` no trae razón (su contrato actual): permite
        corrección en el intento 0 y escala en el intento 1 como reintento fallido.
        """
        return self.resolve_escalation_reason(validation, intento_num) is not None

    def resolve_escalation_reason(
        self, validation: ValidationResult, intento_num: int
    ) -> EscalationReason | None:
        """Resuelve la razón final para que el orquestador no duplique la política."""
        if not isinstance(intento_num, int) or isinstance(intento_num, bool) or intento_num < 0:
            raise ValueError("intento_num debe ser un entero >= 0 (0=inicial, 1=reintento)")
        if validation.valido:
            return None

        reason = validation.razon_escalamiento
        if reason is not None and not isinstance(reason, EscalationReason):
            raise ValueError("razon_escalamiento debe ser EscalationReason o None")
        if reason in _IMMEDIATE_REASONS:
            return reason
        if reason not in (None, EscalationReason.OFERTA_NO_AUTORIZADA):
            raise ValueError("razon_escalamiento inválida para una validación rechazada")
        if intento_num >= 1:
            return EscalationReason.REINTENTO_FALLIDO
        return None

    def escalate(
        self,
        contexto: ClienteObligacionContext,
        razon: EscalationReason,
        trace_id: str,
        historial_conversacion: Sequence[ConversationTurn] | None = None,
        *,
        trace_record: TraceRecord | None = None,
        modo: InputMode = InputMode.REACTIVO,
    ) -> EscalationCase:
        """Crea y entrega un caso, devolviéndolo aun si la entrega falla.

        Para observar el estado de las dos operaciones independientes use
        :meth:`escalate_with_outcome`; esta forma conserva la firma de diseño que
        devuelve directamente ``EscalationCase``.
        """
        return self.escalate_with_outcome(
            contexto,
            razon,
            trace_id,
            historial_conversacion,
            trace_record=trace_record,
            modo=modo,
        ).case

    def escalate_with_outcome(
        self,
        contexto: ClienteObligacionContext,
        razon: EscalationReason,
        trace_id: str,
        historial_conversacion: Sequence[ConversationTurn] | None = None,
        *,
        trace_record: TraceRecord | None = None,
        modo: InputMode = InputMode.REACTIVO,
    ) -> EscalationOutcome:
        """Crea el caso y reporta separadamente trazabilidad y entrega.

        La consulta de ``TraceLogger`` se intenta aun si no existe sink; la entrega
        se intenta aun si la consulta falla. Un trace explícito tiene precedencia,
        útil para el futuro orquestador antes de hacer ``append``.
        """
        if not isinstance(razon, EscalationReason):
            raise ValueError("razon debe ser un EscalationReason")
        if not isinstance(trace_id, str) or not trace_id.strip():
            raise ValueError("trace_id debe ser un texto no vacío")
        if not isinstance(modo, InputMode):
            raise ValueError("modo debe ser un InputMode")

        resolved_trace, trace_available, trace_error = self._resolve_trace(
            contexto, trace_id, trace_record, modo
        )
        case = EscalationCase(
            trace_id=trace_id,
            id_obligacion=contexto.id_obligacion,
            razon=razon,
            contexto=contexto,
            historial_conversacion=_sanitize_history(historial_conversacion or ()),
            trace_completo=resolved_trace,
        )

        delivery_succeeded = False
        delivery_error: str | None = None
        if self._case_sink is not None:
            try:
                self._case_sink.deliver(case)
                delivery_succeeded = True
            except Exception as exc:  # no perder el caso por una frontera externa
                delivery_error = f"{type(exc).__name__}: {exc}"

        return EscalationOutcome(
            case=case,
            trace_available=trace_available,
            delivery_succeeded=delivery_succeeded,
            trace_error=trace_error,
            delivery_error=delivery_error,
        )

    def _resolve_trace(
        self,
        contexto: ClienteObligacionContext,
        trace_id: str,
        trace_record: TraceRecord | None,
        modo: InputMode,
    ) -> tuple[TraceRecord, bool, str | None]:
        if trace_record is not None:
            self._validate_trace(trace_record, contexto, trace_id)
            return trace_record, True, None
        if self._trace_logger is not None:
            try:
                record = self._trace_logger.get_trace(trace_id)
                self._validate_trace(record, contexto, trace_id)
                return record, True, None
            except Exception as exc:  # entregar aun con logger no disponible/inconsistente
                return (
                    _placeholder_trace(contexto, trace_id, modo),
                    False,
                    f"{type(exc).__name__}: {exc}",
                )
        return _placeholder_trace(contexto, trace_id, modo), False, "TraceLogger no configurado"

    @staticmethod
    def _validate_trace(
        record: TraceRecord, contexto: ClienteObligacionContext, trace_id: str
    ) -> None:
        if record.trace_id != trace_id:
            raise ValueError("el TraceRecord entregado no corresponde al trace_id solicitado")
        if record.id_obligacion != contexto.id_obligacion:
            raise ValueError("el TraceRecord entregado no corresponde al contexto de obligación")


def _sanitize_history(history: Sequence[ConversationTurn]) -> list[Any]:
    """Copia y enmascara turnos para que el caso nunca reintroduzca PII en texto."""
    sanitized: list[Any] = []
    for turn in history:
        if isinstance(turn, str):
            sanitized.append(mask_pii(turn)[0])
        elif isinstance(turn, Mapping):
            sanitized.append(_sanitize_value(dict(turn)))
        else:
            raise TypeError("cada turno debe ser texto o un mapeo serializable")
    return sanitized


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, str):
        return mask_pii(value)[0]
    if isinstance(value, Mapping):
        return {key: _sanitize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_value(item) for item in value)
    return value


def _placeholder_trace(
    contexto: ClienteObligacionContext, trace_id: str, modo: InputMode
) -> TraceRecord:
    """Construye un marcador válido sin fingir que contiene una interacción completa."""
    return TraceRecord(
        trace_id=trace_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        id_obligacion=contexto.id_obligacion,
        modo=modo,
        score_info=contexto.score,
        reglas_evaluadas={"escalation_trace_status": "unavailable"},
        eligibility_result=EligibilityResult(
            opciones_elegibles=[],
            acuerdo_elegible=None,
            motivo_no_elegible="Trazabilidad no disponible al crear el escalamiento.",
        ),
        whitelist=[],
        ofertas_mencionadas=[],
        respuesta_agente="",
        agente_responsable="escalation_manager",
        validation_result=ValidationResult(
            valido=False,
            motivo_rechazo="TraceRecord no disponible al momento del escalamiento.",
            escalar=True,
            razon_escalamiento=EscalationReason.FALLO_TECNICO,
        ),
        prompt_version="not-available",
        version_modelo_propension=contexto.score.version_modelo,
    )
