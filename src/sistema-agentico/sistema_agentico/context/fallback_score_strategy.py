"""`FallbackScoreStrategy`: estrategia de degradación controlada del score de propensión.

Ver Component 3 (`ContextBuilder`) de `design.md`, sección "Contrato de Integración con
el Modelo de Propensión" (algoritmo `obtener_score_con_fallback`) y Requirement 3.2 de
`requirements.md` (RF-22, RNF-08).

Esta clase implementa ÚNICAMENTE la rama de fallback del algoritmo `obtener_score_con_fallback`
del diseño: dado que `ContextBuilder.build` (tarea 5.4) es quien invoca primero al
`PropensionClient` (real o fake) y captura la excepción cuando la llamada falla, esta clase
recibe la responsabilidad de decidir, para ese caso, entre:

  (a) un score reciente cacheado (si existe y no está vencido), marcado `degradado=True`; o
  (b) un score neutro conservador (`SCORE_NEUTRO_CONSERVADOR`, `decil=5`,
      `version_modelo="fallback-neutro"`), también marcado `degradado=True`, cuando no hay
      caché válido.

Decisión de diseño — alcance de la excepción que dispara el fallback: `requirements.md`
Requirement 3.2 (RF-22) redacta el disparador como "timeout, error 5xx o `ConnectionError`",
es decir, exactamente `PropensionServiceUnavailableError` (ver `fake_propension_client.py`).
`ScoreNotFoundError` (404, obligación sin score) es semánticamente un caso distinto: no es
que el servicio esté degradado, sino que la obligación consultada no tiene score. Sin
embargo, ambos casos comparten la misma postcondición útil para el resto del pipeline: nunca
propagar la excepción y siempre producir un `ScoreInfo` válido para no bloquear la
evaluación de elegibilidad (más aún, una obligación sin score en absoluto se beneficia tanto
o más de un score neutro conservador que degrada la calidad de las ofertas ofrecidas, en
lugar de tumbar la interacción). Por eso `resolve_fallback` no distingue internamente el
tipo de fallo — es agnóstica a la excepción concreta — y queda a criterio de
`ContextBuilder.build` (tarea 5.4) invocarla tanto ante `PropensionServiceUnavailableError`
como, opcionalmente, ante `ScoreNotFoundError`.

Postcondición (heredada del algoritmo de `design.md`): `resolve_fallback` SIEMPRE retorna un
`ScoreInfo` válido con `degradado=True`; nunca propaga una excepción, incluso ante errores
internos inesperados (defensive programming, Requirement 3.3).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

from sistema_agentico.types import ScoreInfo

__all__ = [
    "MAX_ANTIGUEDAD_CACHE_DIAS",
    "SCORE_NEUTRO_CONSERVADOR",
    "FallbackScoreStrategy",
]

# Ver `design.md`, algoritmo `obtener_score_con_fallback`.
MAX_ANTIGUEDAD_CACHE_DIAS = 30
SCORE_NEUTRO_CONSERVADOR = 0.3
DECIL_NEUTRO = 5
VERSION_MODELO_FALLBACK_NEUTRO = "fallback-neutro"


@dataclass(frozen=True)
class _CacheEntry:
    """Entrada interna del caché de scores recientes: el score y la fecha en que se registró."""

    score_info: ScoreInfo
    fecha_registro: date


class FallbackScoreStrategy:
    """Implementa la rama de fallback de `obtener_score_con_fallback` (`design.md`).

    Mantiene un caché en memoria (`dict`) de los últimos scores obtenidos exitosamente por
    obligación (poblado explícitamente vía `record_score`, típicamente por `ContextBuilder`
    cada vez que el `PropensionClient` responde 200). Ante un fallo de la llamada real,
    `ContextBuilder.build` (tarea 5.4) invoca `resolve_fallback(id_obligacion)`, que resuelve
    entre caché vigente y score neutro conservador siguiendo el algoritmo del diseño.

    El caché es intencionalmente en memoria (no persistente a disco): es adecuado para el
    prototipo local y autocontenido (RNF-13) y para pruebas deterministas; se pierde al
    reiniciar el proceso, lo cual solo implica que se recurrirá con más frecuencia al score
    neutro conservador (comportamiento seguro por diseño, nunca incorrecto).
    """

    def __init__(
        self,
        *,
        max_antiguedad_cache_dias: int = MAX_ANTIGUEDAD_CACHE_DIAS,
        score_neutro_conservador: float = SCORE_NEUTRO_CONSERVADOR,
        clock: Callable[[], date] | None = None,
    ) -> None:
        """
        Args:
            max_antiguedad_cache_dias: antigüedad máxima (en días) que puede tener una
                entrada de caché para considerarse vigente (`MAX_ANTIGUEDAD_CACHE` del
                algoritmo de `design.md`). Configurable para facilitar pruebas.
            score_neutro_conservador: valor de `score` a usar cuando no hay caché vigente
                (`SCORE_NEUTRO_CONSERVADOR` del algoritmo). Debe estar en `[0, 1]`.
            clock: función sin argumentos que retorna la fecha "actual" (`date`), usada
                tanto para registrar la fecha de cada entrada de caché como para calcular
                su antigüedad. Por defecto `date.today`; inyectable para pruebas
                deterministas de vencimiento de caché sin depender del reloj real.
        """
        if max_antiguedad_cache_dias < 0:
            raise ValueError("max_antiguedad_cache_dias no puede ser negativo")
        if not (0.0 <= score_neutro_conservador <= 1.0):
            raise ValueError("score_neutro_conservador debe estar en [0, 1]")

        self._max_antiguedad_cache_dias = max_antiguedad_cache_dias
        self._score_neutro_conservador = score_neutro_conservador
        self._clock: Callable[[], date] = clock if clock is not None else date.today
        self._cache: dict[str, _CacheEntry] = {}

    def record_score(
        self,
        id_obligacion: str,
        score_info: ScoreInfo,
        fecha: date | None = None,
    ) -> None:
        """Registra `score_info` como el último score exitoso conocido para `id_obligacion`.

        Pensado para que `ContextBuilder` lo invoque cada vez que el `PropensionClient`
        (real o fake) responde exitosamente (200), de forma que ese score quede disponible
        como caché reciente ante un fallo posterior de la llamada real.

        Args:
            id_obligacion: identificador de la obligación.
            score_info: score obtenido exitosamente (se almacena tal cual, incluyendo su
                `degradado`, aunque en la práctica un score recién obtenido de la API real
                tendrá `degradado=False`).
            fecha: fecha en la que se registra la entrada de caché; por defecto la fecha
                actual según `clock`. Parametrizable para pruebas.
        """
        if not id_obligacion:
            raise ValueError("id_obligacion no puede ser vacío")
        fecha_registro = fecha if fecha is not None else self._clock()
        self._cache[id_obligacion] = _CacheEntry(score_info=score_info, fecha_registro=fecha_registro)

    def get_cached_score(self, id_obligacion: str) -> ScoreInfo | None:
        """Retorna el score cacheado para `id_obligacion` si existe y no está vencido.

        Vigencia: `antiguedad_dias = (fecha_actual - fecha_registro).days <=
        max_antiguedad_cache_dias` (equivalente a `cache.antiguedad_dias <=
        MAX_ANTIGUEDAD_CACHE` del algoritmo de `design.md`).

        Returns:
            Una copia del `ScoreInfo` cacheado con `degradado=True` si hay una entrada
            vigente; `None` si no hay entrada registrada o si está vencida.
        """
        entry = self._cache.get(id_obligacion)
        if entry is None:
            return None

        antiguedad_dias = (self._clock() - entry.fecha_registro).days
        if antiguedad_dias < 0 or antiguedad_dias > self._max_antiguedad_cache_dias:
            return None

        cached = entry.score_info
        return ScoreInfo(
            score=cached.score,
            decil=cached.decil,
            version_modelo=cached.version_modelo,
            fecha_calificacion=cached.fecha_calificacion,
            degradado=True,
        )

    def resolve_fallback(self, id_obligacion: str) -> ScoreInfo:
        """Resuelve la rama de fallback de `obtener_score_con_fallback` (`design.md`).

        Se asume que el llamador (`ContextBuilder.build`, tarea 5.4) ya intentó la llamada
        real al `PropensionClient` y esta falló (`PropensionServiceUnavailableError` y,
        opcionalmente según su criterio, `ScoreNotFoundError` — ver docstring del módulo).
        Este método NO realiza ninguna llamada al `PropensionClient`: solo decide entre
        caché vigente y score neutro conservador.

        Postcondición (Requirement 3.2, 3.3): SIEMPRE retorna un `ScoreInfo` válido con
        `degradado=True`. Nunca propaga una excepción, ni siquiera ante un error interno
        inesperado al consultar el caché (defensive programming): en ese caso degrada
        directamente al score neutro conservador.

        Returns:
            El `ScoreInfo` cacheado (si hay uno vigente para `id_obligacion`) o, en su
            defecto, el score neutro conservador (`SCORE_NEUTRO_CONSERVADOR`, `decil=5`,
            `version_modelo="fallback-neutro"`), siempre con `degradado=True`.
        """
        try:
            cached = self.get_cached_score(id_obligacion)
            if cached is not None:
                return cached
        except Exception:  # noqa: BLE001 - defensivo: el fallback nunca debe propagar (Requirement 3.3)
            pass

        return self._score_neutro()

    def _score_neutro(self) -> ScoreInfo:
        """Construye el `ScoreInfo` neutro conservador (rama final del algoritmo)."""
        return ScoreInfo(
            score=self._score_neutro_conservador,
            decil=DECIL_NEUTRO,
            version_modelo=VERSION_MODELO_FALLBACK_NEUTRO,
            fecha_calificacion=self._clock().isoformat(),
            degradado=True,
        )
