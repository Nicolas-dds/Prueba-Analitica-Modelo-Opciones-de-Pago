"""`ScoreBatchPrioritizer`: ordena un lote de obligaciones candidatas por score de propensión.

Ver tarea 20 de `tasks.md` y `requirements.md` Requirement 1, AC1 (RF-13):
"obligación priorizada por lote". Cierra un gap real entre esa acceptance criteria y la
implementación previa: `ProactiveBatchRunner.run(prioritized_obligation_ids)` siempre
confió en que quien construye ese iterable ya lo entrega ordenado por prioridad de
contacto — pero ningún componente del sistema hacía realmente esa priorización. El score
de propensión existía y se usaba, pero únicamente dentro de `NBARanker` (Component 5),
donde decide el ORDEN DE LAS OFERTAS dentro de la whitelist de UN cliente ya seleccionado
— nunca decide A QUÉ CLIENTE contactar primero dentro de un lote. Esta clase es el
componente explícito que faltaba para esa segunda decisión.

`ScoreBatchPrioritizer` NO reemplaza ni modifica `ProactiveBatchRunner`: su contrato
(`run(prioritized_obligation_ids: Iterable[str])`) permanece exactamente igual, para no
romper a llamadores que ya priorizan su propia lista por su cuenta (p.ej. un proceso batch
externo con su propia lógica de negocio). `ScoreBatchPrioritizer.prioritize` simplemente
produce un `list[str]` con la misma forma de entrada que `ProactiveBatchRunner.run` ya
acepta, para usarse como paso previo opcional cuando nadie más priorizó el lote.

Doble consulta al servicio de propensión (limitación conocida, ver tarea 20.2)
--------------------------------------------------------------------------------
Este componente consulta `propension_client.get_score` UNA VEZ por candidato,
únicamente para decidir el orden de contacto del lote. Cuando `ProactiveBatchRunner`
procese después cada obligación individualmente, `Orchestrator.handle_interaction` invoca
a `ContextBuilder.build`, que vuelve a consultar el score de esa misma obligación — porque
`ContextBuilder` no tiene (ni este componente le inyecta) ningún caché o mecanismo que
vincule esa segunda consulta con la que ya se hizo aquí. El resultado es una llamada HTTP
adicional por obligación priorizada frente al flujo sin priorización. Es una ineficiencia
aceptada para este prototipo (RNF-13 prioriza simplicidad y acotar tiempo de desarrollo
sobre eficiencia de red) — señalada aquí explícitamente, no oculta. Construir un caché
compartido entre ambos puntos de consulta queda fuera del alcance de esta tarea (scope
creep del gap original, que es únicamente "no existe priorización", no "la priorización
es ineficiente").

Decisión de diseño — prioridad de un id sin score resoluble
---------------------------------------------------------------
Si `propension_client.get_score` falla para un id individual (`ScoreNotFoundError`,
`PropensionServiceUnavailableError`, o cualquier otra excepción no anticipada del
cliente):

- Si se inyectó un `fallback_strategy` (`FallbackScoreStrategy`, tarea 5.3), se reutiliza
  exactamente la misma filosofía de degradación controlada ya establecida en
  `ContextBuilder._resolve_score`: se invoca `fallback_strategy.resolve_fallback(id_obligacion)`
  (nunca lanza) y se usa su `.score` como criterio de orden. Esto mantiene consistencia:
  el mismo id, resuelto por el mismo fallback, produce el mismo score degradado tanto en
  la priorización como luego en `ContextBuilder`.
- Si NO se inyectó `fallback_strategy`, el id se trata como **prioridad mínima**: se le
  asigna un score-sentinela (`_SENTINEL_MIN_PRIORITY_SCORE = -1.0`, fuera del rango real
  `[0, 1]` de un score válido, únicamente para efectos de orden — nunca se expone como un
  `ScoreInfo` real) de forma que quede ordenado estrictamente después de cualquier
  candidato con score real o degradado. El id **nunca se descarta**: sigue apareciendo en
  la lista devuelta, solo al final. Ocultar/eliminar un id cuyo score no se pudo obtener
  violaría el propósito mismo de esta clase (priorizar el lote completo, no un subconjunto).

Un fallo de consulta para UN id nunca aborta la priorización de los demás: cada llamada a
`get_score` se captura individualmente, no el ciclo completo.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from sistema_agentico.context.fallback_score_strategy import FallbackScoreStrategy
from sistema_agentico.context.propension_client import DEFAULT_TIMEOUT_SECONDS
from sistema_agentico.types import ScoreInfo

__all__ = ["ScoreBatchPrioritizer"]

# Nunca es un score real (los scores válidos siempre están en [0, 1]): sentinela usado
# únicamente como clave de orden para un id cuyo score no se pudo resolver y para el cual
# no hay `fallback_strategy` inyectado. Garantiza que ese id ordene estrictamente después
# de cualquier score real o degradado (que están siempre en [0, 1]).
_SENTINEL_MIN_PRIORITY_SCORE = -1.0


class PropensionClient(Protocol):
    """Misma forma estructural que el `PropensionClient` usado por `ContextBuilder`.

    Definido aquí (en vez de importado desde `context.context_builder`) para no acoplar
    este componente de orquestación al módulo interno de `ContextBuilder` — únicamente se
    necesita la forma (`get_score(id_obligacion, timeout_seconds) -> ScoreInfo`), cumplida
    estructuralmente por `HttpPropensionClient` y `FakePropensionClient` sin necesidad de
    heredar de nada de este módulo.
    """

    def get_score(self, id_obligacion: str, timeout_seconds: float) -> ScoreInfo: ...


class ScoreBatchPrioritizer:
    """Ordena un lote de `id_obligacion` candidatos de mayor a menor score de propensión.

    Ver docstring del módulo para el contexto completo (gap de RF-13, doble consulta
    conocida, y la decisión de prioridad mínima para ids sin score resoluble).
    """

    def __init__(
        self,
        propension_client: PropensionClient,
        fallback_strategy: FallbackScoreStrategy | None = None,
    ) -> None:
        """
        Args:
            propension_client: cliente del score de propensión (`HttpPropensionClient` o
                `FakePropensionClient`), consultado una vez por candidato.
            fallback_strategy: estrategia opcional de degradación controlada (tarea 5.3).
                Si se inyecta, un id cuyo score no se pudo obtener usa el score degradado
                de `fallback_strategy.resolve_fallback` en vez de tratarse como prioridad
                mínima. Si es `None`, ese id siempre se trata como prioridad mínima (ver
                docstring del módulo).
        """
        self._propension_client = propension_client
        self._fallback_strategy = fallback_strategy

    def prioritize(
        self,
        id_obligacion_candidates: Iterable[str],
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> list[str]:
        """Devuelve los ids candidatos ordenados de mayor a menor score de propensión.

        Esta lista resultante tiene exactamente la misma forma (`list[str]` de
        `id_obligacion`) que `ProactiveBatchRunner.run` ya acepta como
        `prioritized_obligation_ids` — no se requiere ningún cambio en `ProactiveBatchRunner`
        para consumir el resultado de este método.

        Args:
            id_obligacion_candidates: ids candidatos, en cualquier orden. Un iterable
                vacío produce una lista vacía, sin error.
            timeout_seconds: timeout pasado a cada llamada `propension_client.get_score`
                (Requirement 14.3). Por defecto, el mismo timeout corto por defecto que
                usa `HttpPropensionClient`.

        Returns:
            Lista de `id_obligacion`, ordenada de mayor a menor score. Ante empate exacto
            de score, se preserva el orden relativo original de entrada (tiebreak
            determinista por índice de entrada, no por orden de iteración del `sort`).
            Ningún id se omite del resultado, incluso si su score no se pudo obtener.
        """
        candidates = list(id_obligacion_candidates)
        if not candidates:
            return []

        # (score, índice_original, id_obligacion): el índice original es la clave de
        # tiebreak explícita — nunca se depende implícitamente de la estabilidad de
        # `sorted`/`reverse=True` para el orden de empates.
        scored: list[tuple[float, int, str]] = [
            (self._resolve_score(id_obligacion, timeout_seconds), index, id_obligacion)
            for index, id_obligacion in enumerate(candidates)
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [id_obligacion for _, _, id_obligacion in scored]

    def _resolve_score(self, id_obligacion: str, timeout_seconds: float) -> float:
        """Resuelve el score de un único candidato; nunca propaga una excepción.

        Un fallo de `propension_client.get_score` para este id NO debe abortar la
        priorización de los demás candidatos (ver docstring del módulo) — por eso el
        `try/except` envuelve únicamente esta llamada individual, no el ciclo completo de
        `prioritize`.
        """
        try:
            return self._propension_client.get_score(id_obligacion, timeout_seconds).score
        except Exception:  # noqa: BLE001 - defensivo: un fallo individual no debe abortar el lote
            if self._fallback_strategy is not None:
                try:
                    return self._fallback_strategy.resolve_fallback(id_obligacion).score
                except Exception:  # noqa: BLE001 - defensivo: el fallback nunca debería lanzar, pero no confiar ciegamente
                    return _SENTINEL_MIN_PRIORITY_SCORE
            return _SENTINEL_MIN_PRIORITY_SCORE
