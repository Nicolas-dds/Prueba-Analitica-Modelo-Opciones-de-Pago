"""Dependencias compartidas de la API HTTP (construcción/caché del pipeline).

Sigue la misma convención que `src/modelo-propension/api/dependencies.py`:
`@lru_cache` para construir el pipeline una única vez por proceso.
"""
from __future__ import annotations

from functools import lru_cache

from sistema_agentico.orchestration import ProactiveBatchRunner, ReactiveRunner
from sistema_agentico.orchestration.score_batch_prioritizer import ScoreBatchPrioritizer
from sistema_agentico.pipeline_factory import Pipeline, build_pipeline

__all__ = ["get_proactive_runner", "get_reactive_runner", "get_score_batch_prioritizer"]


@lru_cache
def _get_pipeline() -> Pipeline:
    """Construye (y cachea) el pipeline completo para la vida del proceso."""
    return build_pipeline()


def get_reactive_runner() -> ReactiveRunner:
    """`ReactiveRunner` compartido, inyectable como dependencia de FastAPI."""
    return _get_pipeline().reactive_runner


def get_proactive_runner() -> ProactiveBatchRunner:
    """`ProactiveBatchRunner` compartido, inyectable como dependencia de FastAPI."""
    return _get_pipeline().proactive_runner


def get_score_batch_prioritizer() -> ScoreBatchPrioritizer:
    """`ScoreBatchPrioritizer` inyectable como dependencia de FastAPI (tarea 20.5).

    Reutiliza el mismo `propension_client`/`fallback_strategy` del pipeline cacheado
    (`_get_pipeline()`) en vez de construir una segunda instancia de cliente — mismo
    patrón que `get_reactive_runner`/`get_proactive_runner`. Se construye una instancia
    nueva de `ScoreBatchPrioritizer` en cada llamada (no se cachea con `@lru_cache`):
    la clase en sí no tiene estado propio más allá de las referencias que recibe en
    `__init__`, así que no hay costo real de reconstrucción por request.
    """
    pipeline = _get_pipeline()
    return ScoreBatchPrioritizer(pipeline.propension_client, pipeline.fallback_strategy)
