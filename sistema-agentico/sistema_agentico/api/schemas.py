"""Contratos Pydantic de request/response de la API HTTP (tarea 17.3).

Estos modelos son deliberadamente más angostos que los tipos internos del pipeline
(`sistema_agentico.types`): nunca exponen `id_cliente`, `ClienteObligacionContext`,
`EscalationCase`, `TraceRecord` ni ningún otro detalle interno de reglas de negocio o
PII. Sólo se serializa lo explícitamente listado en la tarea 17.3:

- `trace_id`, `status`, `respuesta` (texto/ofertas_mencionadas/prompt_version),
  `escalation_reason` (como valor string del enum, nunca el `EscalationCase` completo)
  y `trace_logged`.

Reutilizan la forma de `InteractionOutcome`
(`sistema_agentico.orchestration.orchestrator`) mediante `resultado_from_outcome`,
en vez de duplicar la lógica de mapeo en cada endpoint.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from sistema_agentico.orchestration.orchestrator import InteractionOutcome, InteractionStatus

__all__ = [
    "HealthResponse",
    "InteraccionResultado",
    "ProactivaItemResultado",
    "ProactivaRequest",
    "ReactivaRequest",
    "RespuestaAgenteSchema",
    "resultado_from_outcome",
]


class HealthResponse(BaseModel):
    """Respuesta de `GET /health` (mismo contrato que `modelo-propension`)."""

    status: str
    timestamp: str


class RespuestaAgenteSchema(BaseModel):
    """Subconjunto público de `AgentResponse`: sin campos internos adicionales."""

    texto: str
    ofertas_mencionadas: list[str]
    prompt_version: str


class InteraccionResultado(BaseModel):
    """Forma de respuesta común a ambos endpoints, derivada de `InteractionOutcome`.

    Deliberadamente NO incluye `escalation_case` (puede contener
    `ClienteObligacionContext`/`TraceRecord` con detalle interno de reglas): sólo se
    expone `escalation_reason` como su valor string.
    """

    trace_id: str
    status: Literal["final", "escalated"]
    respuesta: RespuestaAgenteSchema | None
    escalation_reason: str | None
    trace_logged: bool


class ReactivaRequest(BaseModel):
    """Cuerpo de `POST /interacciones/reactiva`.

    `id_obligacion` y `mensaje_cliente` vacíos se rechazan con 422 (validación de
    Pydantic), consistente con la validación de `ReactiveRunner`/`RawInput`.
    """

    id_obligacion: str = Field(min_length=1)
    mensaje_cliente: str = Field(min_length=1)
    historial_conversacion: list[str | dict[str, Any]] | None = None


class ProactivaRequest(BaseModel):
    """Cuerpo de `POST /interacciones/proactiva`."""

    ids_obligacion: list[str] = Field(default_factory=list)


class ProactivaItemResultado(BaseModel):
    """Resultado de un ítem del lote proactivo.

    El orden de la lista de `ProactivaItemResultado` devuelta por el endpoint refleja
    el orden de PROCESAMIENTO real (priorizado por score de propensión descendente,
    tarea 20.5), no necesariamente el orden de `ids_obligacion` en la petición.
    """

    id_obligacion: str
    completado: bool
    resultado: InteraccionResultado | None
    error: str | None


def resultado_from_outcome(outcome: InteractionOutcome) -> InteraccionResultado:
    """Mapea un `InteractionOutcome` interno al contrato público sin PII/detalle interno."""
    respuesta: RespuestaAgenteSchema | None = None
    if outcome.response is not None:
        respuesta = RespuestaAgenteSchema(
            texto=outcome.response.texto,
            ofertas_mencionadas=list(outcome.response.ofertas_mencionadas),
            prompt_version=outcome.response.prompt_version,
        )
    return InteraccionResultado(
        trace_id=outcome.trace_id,
        status="final" if outcome.status is InteractionStatus.FINAL else "escalated",
        respuesta=respuesta,
        escalation_reason=outcome.escalation_reason.value if outcome.escalation_reason else None,
        trace_logged=outcome.trace_logged,
    )
