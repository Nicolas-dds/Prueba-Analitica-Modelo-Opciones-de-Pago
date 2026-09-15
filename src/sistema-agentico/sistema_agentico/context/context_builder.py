"""`ContextBuilder`: construcción del contexto completo de una obligación (Component 3).

Ver Component 3 de `design.md` y Requirements 3.3, 3.4, 3.5 de `requirements.md`.

`ContextBuilder.build(id_obligacion)` combina tres fuentes:

1. El score de propensión (`PropensionClient.get_score`, tarea 5.1/5.2), degradando de
   forma controlada vía `FallbackScoreStrategy` (tarea 5.3) ante `ScoreNotFoundError` o
   `PropensionServiceUnavailableError`.
2. El perfil crudo de cliente/obligación e historial de gestión (`ObligacionProfileProvider`,
   definido en este módulo — ver "Decisión de diseño: fuente de perfil" más abajo).
3. Reglas de detección de datos contradictorios (Requirement 3.5), que deciden si se
   retorna un `ClienteObligacionContext` o si se escala la interacción sin retornarlo.

Decisión de diseño: fuente de perfil (`ObligacionProfileProvider`)
--------------------------------------------------------------------
`design.md` no define un componente/repositorio separado para el "Perfil Cliente /
Obligación" e "Historial de Gestión" (aparecen únicamente como fuentes de datos en el
diagrama de arquitectura); tampoco existen aún en este spec. Para no bloquear la tarea 5.4
ni sobre-diseñar un repositorio real (prototipo, RNF-13), se define aquí una abstracción
mínima:

- `ObligacionProfileData`: DTO crudo con TODOS los campos `Optional`, a diferencia de
  `ClienteObligacionContext` (que exige campos completos y válidos). Modela el dato tal
  como llega de la fuente, que puede venir incompleto o contradictorio — exactamente el
  caso que Requirement 3.5 exige detectar.
- `ObligacionProfileProvider` (Protocol): `get_profile(id_obligacion) -> ObligacionProfileData`,
  puede lanzar `ObligacionProfileNotFoundError`.
- `InMemoryObligacionProfileProvider`: implementación por defecto en memoria/dict, con un
  factory `with_default_synthetic_profiles` que reutiliza `synthetic_data.profiles`
  (tarea 4.1) para poblarse sin necesidad de un repositorio real, consistente con el
  cliente de propensión fake (`FakePropensionClient.with_default_synthetic_scores`).

`ContextBuilder` depende de esta abstracción (inyectable) en lugar de acoplarse a una
fuente concreta, igual que ya depende de `PropensionClient` (Protocol).

Decisión de diseño: señal de escalamiento controlada (`ContextEscalationRequired`)
------------------------------------------------------------------------------------
`EscalationManager` (Component 7, tarea 10) todavía no existe. Requirement 3.5 exige que
`ContextBuilder` "escale la interacción... sin retornar un `ClienteObligacionContext` al
`Orchestrator`", y Requirement 3.3 exige que nunca se propague una "excepción no
controlada". Estas dos afirmaciones NO son contradictorias: `ContextEscalationRequired`
es una excepción deliberada, tipada y documentada — el canal de señal CONTROLADO que
Requirement 3.5 requiere — no un fallo inesperado. El futuro `Orchestrator` (tarea 11)
deberá capturar específicamente este tipo (`except ContextEscalationRequired as exc`) y
enrutar `exc.id_obligacion` / `exc.razon` (siempre `EscalationReason.INFO_CONTRADICTORIA`)
/ `exc.detalle` hacia `EscalationManager.escalate`. Cualquier OTRA excepción inesperada
(bug, forma de dato no anticipada) se captura de forma defensiva dentro de `build` y se
traduce también a `ContextEscalationRequired` (ver `build`), de modo que jamás escapa una
excepción no controlada hacia quien invoque a `ContextBuilder` — satisfaciendo
Requirement 3.3 incluso en el peor caso.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from sistema_agentico.context.fake_propension_client import (
    PropensionServiceUnavailableError,
    ScoreNotFoundError,
)
from sistema_agentico.context.fallback_score_strategy import FallbackScoreStrategy
from sistema_agentico.context.propension_client import DEFAULT_TIMEOUT_SECONDS
from sistema_agentico.types import ClienteObligacionContext, EscalationReason, OfertaAplicada, ScoreInfo

__all__ = [
    "ContextBuilder",
    "ContextEscalationRequired",
    "InMemoryObligacionProfileProvider",
    "ObligacionProfileData",
    "ObligacionProfileNotFoundError",
    "ObligacionProfileProvider",
    "PropensionClient",
]


class PropensionClient(Protocol):
    """Ver `design.md`, Component 3. Cumplido estructuralmente por `HttpPropensionClient`
    y `FakePropensionClient` (tareas 5.1/5.2)."""

    def get_score(self, id_obligacion: str, timeout_seconds: float) -> ScoreInfo: ...


class ContextEscalationRequired(Exception):
    """Señal controlada de escalamiento emitida por `ContextBuilder.build` (Requirement 3.5).

    NO es una "excepción no controlada" en el sentido de Requirement 3.3: es el mecanismo
    deliberado mediante el cual `ContextBuilder` comunica a su llamador (el futuro
    `Orchestrator`, tarea 11) que debe escalar la interacción sin recibir un
    `ClienteObligacionContext`. El `Orchestrator` deberá capturar específicamente este
    tipo (nunca dejarlo propagar sin manejar) e invocar a `EscalationManager.escalate`
    con los datos que transporta.

    Attributes:
        id_obligacion: obligación para la cual se requiere escalamiento.
        razon: siempre `EscalationReason.INFO_CONTRADICTORIA` en este módulo (único motivo
            de escalamiento que `ContextBuilder` puede producir, Requirement 3.5).
        detalle: explicación legible (no PII) de la contradicción/campo ausente detectado,
            o del error inesperado que forzó el escalamiento defensivo (ver `build`).
    """

    def __init__(self, id_obligacion: str, razon: EscalationReason, detalle: str) -> None:
        self.id_obligacion = id_obligacion
        self.razon = razon
        self.detalle = detalle
        super().__init__(
            f"Escalamiento requerido para id_obligacion={id_obligacion!r}: "
            f"razon={razon.value}, detalle={detalle}"
        )


# ---------------------------------------------------------------------------
# Fuente de perfil/historial (`ObligacionProfileProvider`) — ver docstring del módulo.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObligacionProfileData:
    """DTO crudo (no validado) de los campos de perfil/historial de una obligación.

    A diferencia de `ClienteObligacionContext` (`types.py`), TODOS los campos son
    `Optional` deliberadamente: modela el dato tal como llega de la fuente (perfil
    cliente/obligación + historial de gestión), que puede venir incompleto o
    contradictorio. `ContextBuilder._detectar_contradiccion` es responsable de detectar
    esos casos (Requirement 3.5) antes de construir el `ClienteObligacionContext` final.
    """

    id_cliente: Optional[str]
    dias_mora: Optional[int]
    exposicion: Optional[float]
    segmento: Optional[str]
    canal_gestion: Optional[str]
    ofertas_aplicadas_mes: Optional[list[OfertaAplicada]]
    ultima_opcion_aplicada: Optional[OfertaAplicada]
    acuerdo_pago_vigente: Optional[bool]
    restriccion_vigente: Optional[bool]

    @classmethod
    def faltante(cls) -> "ObligacionProfileData":
        """Perfil completamente ausente (todos los campos `None`).

        Usado cuando `ObligacionProfileProvider.get_profile` lanza
        `ObligacionProfileNotFoundError` o falla inesperadamente: se trata de forma
        uniforme como "campos requeridos ausentes" (Requirement 3.5) en lugar de
        propagar la excepción.
        """
        return cls(
            id_cliente=None,
            dias_mora=None,
            exposicion=None,
            segmento=None,
            canal_gestion=None,
            ofertas_aplicadas_mes=None,
            ultima_opcion_aplicada=None,
            acuerdo_pago_vigente=None,
            restriccion_vigente=None,
        )


class ObligacionProfileNotFoundError(Exception):
    """No existe perfil/historial registrado para `id_obligacion`."""

    def __init__(self, id_obligacion: str) -> None:
        self.id_obligacion = id_obligacion
        super().__init__(f"No existe perfil para id_obligacion={id_obligacion!r}")


class ObligacionProfileProvider(Protocol):
    """Abstracción inyectable de la fuente de perfil/historial de una obligación.

    Ver "Decisión de diseño: fuente de perfil" en el docstring del módulo. Cumplida
    estructuralmente por `InMemoryObligacionProfileProvider` (implementación por
    defecto de este módulo).
    """

    def get_profile(self, id_obligacion: str) -> ObligacionProfileData: ...


class InMemoryObligacionProfileProvider:
    """Implementación en memoria/dict de `ObligacionProfileProvider` (RNF-13).

    No realiza ninguna llamada de red ni acceso a filesystem: resuelve cada consulta
    contra un diccionario interno, poblado en el constructor o mediante
    `register_profile`. Análoga a `FakePropensionClient` (tarea 5.2) en su rol dentro
    del prototipo: permite ejecutar `ContextBuilder` y escribir tests deterministas sin
    depender de un repositorio real de perfiles.
    """

    def __init__(self, profiles: dict[str, ObligacionProfileData] | None = None) -> None:
        self._profiles: dict[str, ObligacionProfileData] = dict(profiles) if profiles else {}

    def register_profile(self, id_obligacion: str, profile_data: ObligacionProfileData) -> None:
        """Registra (o sobrescribe) el perfil crudo que se retornará para `id_obligacion`."""
        if not id_obligacion:
            raise ValueError("id_obligacion no puede ser vacío")
        self._profiles[id_obligacion] = profile_data

    def get_profile(self, id_obligacion: str) -> ObligacionProfileData:
        """Retorna el perfil registrado para `id_obligacion`.

        Raises:
            ObligacionProfileNotFoundError: si `id_obligacion` no tiene perfil registrado.
        """
        try:
            return self._profiles[id_obligacion]
        except KeyError:
            raise ObligacionProfileNotFoundError(id_obligacion) from None

    @classmethod
    def with_default_synthetic_profiles(
        cls, n: int = 20, seed: int | None = 0
    ) -> "InMemoryObligacionProfileProvider":
        """Construye un provider pre-poblado con `n` perfiles sintéticos (RNF-13).

        Reutiliza `synthetic_data.profiles.generate_profiles` (tarea 4.1) con el mismo
        `seed` que `FakePropensionClient.with_default_synthetic_scores`, de forma que
        ambos, instanciados con el mismo `n`/`seed` por defecto, produzcan datos
        consistentes bajo los mismos `id_obligacion` sintéticos (prefijo `SYN-OBL-`)
        para ejecución local sin dependencias externas.
        """
        from sistema_agentico.synthetic_data.profiles import generate_profiles

        perfiles = generate_profiles(n, seed=seed)
        data = {
            perfil.id_obligacion: ObligacionProfileData(
                id_cliente=perfil.id_cliente,
                dias_mora=perfil.dias_mora,
                exposicion=perfil.exposicion,
                segmento=perfil.segmento,
                canal_gestion=perfil.canal_gestion,
                ofertas_aplicadas_mes=list(perfil.ofertas_aplicadas_mes),
                ultima_opcion_aplicada=perfil.ultima_opcion_aplicada,
                acuerdo_pago_vigente=perfil.acuerdo_pago_vigente,
                restriccion_vigente=perfil.restriccion_vigente,
            )
            for perfil in perfiles
        }
        return cls(profiles=data)


# ---------------------------------------------------------------------------
# `ContextBuilder`
# ---------------------------------------------------------------------------


class ContextBuilder:
    """Construye el `ClienteObligacionContext` completo de una obligación (Component 3).

    Ver `design.md`:
    ```python
    class ContextBuilder:
        def __init__(self, propension_client: PropensionClient, fallback_strategy: "FallbackScoreStrategy") -> None: ...
        def build(self, id_obligacion: str) -> ClienteObligacionContext: ...
    ```
    """

    def __init__(
        self,
        propension_client: PropensionClient,
        fallback_strategy: FallbackScoreStrategy,
        profile_provider: ObligacionProfileProvider | None = None,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """
        Args:
            propension_client: cliente del score de propensión (`HttpPropensionClient` o
                `FakePropensionClient`, tareas 5.1/5.2).
            fallback_strategy: estrategia de degradación controlada del score (tarea 5.3).
            profile_provider: fuente de perfil/historial de la obligación (ver "Decisión
                de diseño: fuente de perfil" en el docstring del módulo). Si es `None`, se
                usa `InMemoryObligacionProfileProvider.with_default_synthetic_profiles()`
                (ejecución local autocontenida, RNF-13).
            timeout_seconds: timeout pasado a `propension_client.get_score` (Requirement 14.3).
        """
        self._propension_client = propension_client
        self._fallback_strategy = fallback_strategy
        self._profile_provider: ObligacionProfileProvider = (
            profile_provider
            if profile_provider is not None
            else InMemoryObligacionProfileProvider.with_default_synthetic_profiles()
        )
        self._timeout_seconds = timeout_seconds

    def build(self, id_obligacion: str) -> ClienteObligacionContext:
        """Construye el contexto completo de `id_obligacion` (Requirements 3.3, 3.4, 3.5).

        Flujo:
        1. Resuelve el score (`_resolve_score`): API real o, ante fallo, fallback
           degradado (Requirement 3.2, ya implementado por `FallbackScoreStrategy`).
        2. Resuelve el perfil crudo (`_resolve_profile`): fuente inyectada o perfil
           "faltante" si no existe / falla inesperadamente.
        3. Detecta contradicciones (`_detectar_contradiccion`, Requirement 3.5). Si
           encuentra alguna, lanza `ContextEscalationRequired` — la señal de
           escalamiento CONTROLADA (ver docstring del módulo) — en lugar de retornar
           un contexto.
        4. Si no hay contradicción, enriquece y retorna el `ClienteObligacionContext`
           completo (Requirement 3.4: historial de ofertas y última opción aplicada).

        Cualquier excepción NO anticipada (bug, forma de dato inesperada) es capturada
        de forma defensiva y traducida también a `ContextEscalationRequired`: satisface
        Requirement 3.3 ("nunca propagar una excepción no controlada") incluso en el
        peor caso, sin dejar de cumplir la obligación de escalar de Requirement 3.5.

        Raises:
            ValueError: si `id_obligacion` es vacío (precondición, no una excepción no
                controlada en tiempo de negocio: es un error de uso de la API).
            ContextEscalationRequired: si se detecta información contradictoria o
                ausente (Requirement 3.5), o ante cualquier error inesperado.
        """
        if not id_obligacion:
            raise ValueError("id_obligacion no puede ser vacío")

        try:
            score = self._resolve_score(id_obligacion)
            profile = self._resolve_profile(id_obligacion)

            contradiccion = self._detectar_contradiccion(profile, score)
            if contradiccion is not None:
                raise ContextEscalationRequired(
                    id_obligacion, EscalationReason.INFO_CONTRADICTORIA, contradiccion
                )

            return ClienteObligacionContext(
                id_obligacion=id_obligacion,
                id_cliente=profile.id_cliente,  # type: ignore[arg-type]  # no-None garantizado por _detectar_contradiccion
                dias_mora=profile.dias_mora,  # type: ignore[arg-type]
                exposicion=profile.exposicion,  # type: ignore[arg-type]
                segmento=profile.segmento,
                canal_gestion=profile.canal_gestion,
                score=score,
                ofertas_aplicadas_mes=list(profile.ofertas_aplicadas_mes) if profile.ofertas_aplicadas_mes else [],
                ultima_opcion_aplicada=profile.ultima_opcion_aplicada,
                acuerdo_pago_vigente=bool(profile.acuerdo_pago_vigente),
                restriccion_vigente=bool(profile.restriccion_vigente),
            )
        except ContextEscalationRequired:
            raise
        except Exception as exc:  # noqa: BLE001 - defensivo (Requirement 3.3): nunca crashear, escalar en su lugar
            raise ContextEscalationRequired(
                id_obligacion,
                EscalationReason.INFO_CONTRADICTORIA,
                f"Error inesperado construyendo el contexto: {exc!r}",
            ) from exc

    def _resolve_score(self, id_obligacion: str) -> ScoreInfo:
        """Resuelve el score real o, ante fallo, delega en `FallbackScoreStrategy`.

        `ScoreNotFoundError` (404) y `PropensionServiceUnavailableError` (timeout/5xx/
        `ConnectionError`) se tratan de forma idéntica: ambas disparan
        `fallback_strategy.resolve_fallback`, siguiendo el criterio documentado en
        `fallback_score_strategy.py` (una obligación sin score se beneficia tanto o más
        de un score neutro conservador que de bloquear la interacción). Cualquier otra
        excepción inesperada del cliente también se trata como indisponibilidad
        (defensive programming, Requirement 3.3): `resolve_fallback` nunca propaga.
        """
        try:
            score = self._propension_client.get_score(id_obligacion, self._timeout_seconds)
        except (ScoreNotFoundError, PropensionServiceUnavailableError):
            return self._fallback_strategy.resolve_fallback(id_obligacion)
        except Exception:  # noqa: BLE001 - defensivo: fallo inesperado del cliente == indisponibilidad
            return self._fallback_strategy.resolve_fallback(id_obligacion)

        # Llamada exitosa: refresca el caché de fallback para futuros fallos (Requirement 3.2).
        self._fallback_strategy.record_score(id_obligacion, score)
        return score

    def _resolve_profile(self, id_obligacion: str) -> ObligacionProfileData:
        """Resuelve el perfil crudo vía `profile_provider`, o un perfil "faltante".

        `ObligacionProfileNotFoundError` y cualquier error inesperado del provider se
        traducen a `ObligacionProfileData.faltante()`, de forma que la ausencia total
        de perfil se detecte uniformemente como "campos requeridos ausentes" en
        `_detectar_contradiccion` (Requirement 3.5), sin propagar la excepción.
        """
        try:
            return self._profile_provider.get_profile(id_obligacion)
        except ObligacionProfileNotFoundError:
            return ObligacionProfileData.faltante()
        except Exception:  # noqa: BLE001 - defensivo: fallo inesperado del provider == perfil ausente
            return ObligacionProfileData.faltante()

    def _detectar_contradiccion(
        self, profile: ObligacionProfileData, score: ScoreInfo
    ) -> Optional[str]:
        """Detecta campos ausentes o mutuamente contradictorios (Requirement 3.5).

        Verificaciones implementadas (no exhaustivas, pero suficientes para ejercitar
        Requirement 3.5 en property tests posteriores — tareas 5.6/5.7):

        1. Campos requeridos ausentes: `id_cliente`, `dias_mora`, `exposicion`,
           `acuerdo_pago_vigente` o `restriccion_vigente` son `None` (o cadena vacía
           para `id_cliente`).
        2. `dias_mora` o `exposicion` presentes pero negativos en el perfil CRUDO (antes
           de construir `ClienteObligacionContext`, cuyo `__post_init__` ya rechaza
           estos valores — se detecta aquí para escalar en lugar de dejar que ese
           `ValueError` se propague sin control).
        3. `acuerdo_pago_vigente=True` sin `ultima_opcion_aplicada` que lo sustente
           (ejemplo textual de `design.md`, "Error Scenario 4").
        4. `ultima_opcion_aplicada` presente pero su `tipo_opcion` no aparece en ningún
           registro de `ofertas_aplicadas_mes` (inconsistencia interna del historial).

        El resultado ya aplicado del fallback (`score`) se incluye en la verificación
        por completitud del contrato de Requirement 3.5 ("incluyendo el resultado ya
        aplicado del fallback"): en la práctica `ScoreInfo.__post_init__` ya garantiza
        `score` válido, por lo que esta verificación es defensiva/no alcanzable en
        condiciones normales, pero se deja explícita para no asumir silenciosamente que
        el fallback nunca puede fallar de una forma no anticipada.

        Returns:
            Descripción legible (no PII) de la primera contradicción encontrada, o
            `None` si el perfil es consistente.
        """
        campos_ausentes = [
            nombre
            for nombre, valor in (
                ("id_cliente", profile.id_cliente),
                ("dias_mora", profile.dias_mora),
                ("exposicion", profile.exposicion),
                ("acuerdo_pago_vigente", profile.acuerdo_pago_vigente),
                ("restriccion_vigente", profile.restriccion_vigente),
            )
            if valor is None or (isinstance(valor, str) and not valor.strip())
        ]
        if campos_ausentes:
            return f"Campos requeridos ausentes en el perfil: {', '.join(campos_ausentes)}"

        if profile.dias_mora is not None and profile.dias_mora < 0:
            return "dias_mora negativo en el perfil crudo (dato imposible)"
        if profile.exposicion is not None and profile.exposicion < 0:
            return "exposicion negativa en el perfil crudo (dato imposible)"

        if profile.acuerdo_pago_vigente and profile.ultima_opcion_aplicada is None:
            return "acuerdo_pago_vigente=True sin evidencia de ultima_opcion_aplicada que lo sustente"

        ofertas = profile.ofertas_aplicadas_mes or []
        if profile.ultima_opcion_aplicada is not None:
            tipos_historial = {oferta.tipo_opcion for oferta in ofertas}
            if profile.ultima_opcion_aplicada.tipo_opcion not in tipos_historial:
                return "ultima_opcion_aplicada no aparece en ofertas_aplicadas_mes (historial inconsistente)"

        if not (0.0 <= score.score <= 1.0) or not (1 <= score.decil <= 10):
            # Defensivo/no alcanzable en la práctica: ScoreInfo.__post_init__ ya lo impide.
            return "score de propensión (incluido el resultado del fallback) fuera de rango válido"

        return None
