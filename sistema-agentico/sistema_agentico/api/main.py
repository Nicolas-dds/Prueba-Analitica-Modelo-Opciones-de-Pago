"""API HTTP de `sistema-agentico` (tarea 17.3).

Expone el `Orchestrator`/`ReactiveRunner`/`ProactiveBatchRunner` ya implementados
como un servicio HTTP nuevo, reutilizando `InteractionOutcome` (ver
`sistema_agentico.api.schemas.resultado_from_outcome`) en lugar de re-implementar la
lógica de negocio. Fuera del alcance original de `design.md` (tarea 17 de
`.kiro/specs/sistema-agentico-cobranza/tasks.md`).

SEGURIDAD (obligatorio reportar, no omitir en silencio): este prototipo NO tiene
autenticación ni autorización configuradas. Cualquier cliente con acceso de red al
contenedor/puerto puede invocar ambos endpoints de interacción. Es aceptable para el
alcance de empaquetamiento local/demo de la tarea 17, pero un despliegue real
requeriría, como mínimo, autenticación (p.ej. API key o JWT) y control de acceso de
red (p.ej. sólo accesible desde la red interna/gateway).
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException

from sistema_agentico.api.dependencies import (
    get_proactive_runner,
    get_reactive_runner,
    get_score_batch_prioritizer,
)
from sistema_agentico.api.schemas import (
    HealthResponse,
    InteraccionResultado,
    ProactivaItemResultado,
    ProactivaRequest,
    ReactivaRequest,
    resultado_from_outcome,
)
from sistema_agentico.orchestration import ProactiveBatchRunner, ReactiveRunner, ReactiveRunnerError
from sistema_agentico.orchestration.score_batch_prioritizer import ScoreBatchPrioritizer

app = FastAPI(
    title="Sistema Agéntico de Gestión de Cartera en Mora",
    description=(
        "Expone el Orchestrator/ReactiveRunner/ProactiveBatchRunner del sistema "
        "agéntico como una API HTTP (tarea 17.3, empaquetamiento fuera del alcance "
        "original de design.md). Prototipo sin autenticación/autorización."
    ),
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Chequeo de disponibilidad del servicio (mismo contrato que `modelo-propension`)."""
    return HealthResponse(status="ok", timestamp=datetime.now(timezone.utc).isoformat())


@app.post("/interacciones/reactiva", response_model=InteraccionResultado)
def crear_interaccion_reactiva(
    request: ReactivaRequest,
    reactive_runner: ReactiveRunner = Depends(get_reactive_runner),
) -> InteraccionResultado:
    """Ejecuta una interacción reactiva (mensaje de cliente) de punta a punta.

    `id_obligacion`/`mensaje_cliente` vacíos ya fueron rechazados con 422 por la
    validación de `ReactivaRequest`. Un fallo controlado del `InputGuardrail` o del
    `Orchestrator` (``ReactiveRunnerError``) se traduce a 502: refleja que la falla
    ocurrió en un componente/dependencia interna del pipeline, no que la petición del
    cliente HTTP en sí sea inválida (422/400) ni que el proceso de la API haya
    colapsado de forma genérica (500).
    """
    try:
        outcome = reactive_runner.run(
            request.id_obligacion,
            request.mensaje_cliente,
            historial_conversacion=request.historial_conversacion,
        )
    except ReactiveRunnerError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Fallo en un componente del pipeline ({exc.component}).",
        ) from exc
    return resultado_from_outcome(outcome)


@app.post("/interacciones/proactiva", response_model=list[ProactivaItemResultado])
def crear_interacciones_proactivas(
    request: ProactivaRequest,
    proactive_runner: ProactiveBatchRunner = Depends(get_proactive_runner),
    score_batch_prioritizer: ScoreBatchPrioritizer = Depends(get_score_batch_prioritizer),
) -> list[ProactivaItemResultado]:
    """Prioriza por score de propensión y ejecuta el lote proactivo resultante (tarea 20.5).

    `request.ids_obligacion` se reordena primero con `ScoreBatchPrioritizer.prioritize`
    (mayor a menor score) antes de pasarse a `ProactiveBatchRunner.run` — cierra el
    mismo gap de RF-13 ya resuelto para `run_demo.py` en la tarea 20.3, para que la API
    HTTP se comporte igual. La respuesta refleja ese orden de procesamiento real (por
    score descendente), no necesariamente el orden de `ids_obligacion` en la petición.
    Ningún id se descarta por la priorización (ver docstring de `ScoreBatchPrioritizer`
    para el detalle de doble consulta al servicio de propensión y de prioridad mínima
    ante un score no resoluble).

    `ProactiveBatchRunner.run` ya captura los fallos por obligación (una falla no
    detiene el resto del lote), por lo que aquí no hay un caso de error a nivel de
    lote completo que traducir a un código HTTP distinto de 200: cada ítem expone su
    propio `completado`/`error`.
    """
    prioritized_ids = score_batch_prioritizer.prioritize(request.ids_obligacion)
    items = proactive_runner.run(prioritized_ids)
    return [
        ProactivaItemResultado(
            id_obligacion=item.id_obligacion,
            completado=item.completed,
            resultado=resultado_from_outcome(item.outcome) if item.outcome is not None else None,
            error=item.error,
        )
        for item in items
    ]


if __name__ == "__main__":
    import uvicorn

    from sistema_agentico.settings import SISTEMA_AGENTICO_HOST, SISTEMA_AGENTICO_PORT

    uvicorn.run(app, host=SISTEMA_AGENTICO_HOST, port=SISTEMA_AGENTICO_PORT)
