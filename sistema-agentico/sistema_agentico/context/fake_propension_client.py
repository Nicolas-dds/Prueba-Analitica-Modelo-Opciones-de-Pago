"""`FakePropensionClient`: implementación en memoria del protocolo `PropensionClient`.

Ver Component 3 (`ContextBuilder`) de `design.md` y Requirement 14.6 de `requirements.md`
("ejecución local y completamente autocontenida, sin depender de conexiones a componentes
cloud, del banco, ni de ningún servicio externo mockeado o real, para su operación en modo
prototipo" — RNF-13).

Esta implementación NO realiza ninguna llamada de red ni acceso a filesystem: todos los
scores se resuelven contra un diccionario en memoria, poblado explícitamente por quien la
instancia (tests o script de ejecución local). Esto permite:

- Ejecutar el `ContextBuilder` (tarea 5.4) y el resto del pipeline de forma local, sin
  levantar el servicio `modelo-propension` (RNF-13).
- Escribir tests deterministas que no dependen de red ni de un servicio externo (real o
  mockeado vía HTTP), registrando exactamente los scores que cada caso de prueba necesita.

Las excepciones `ScoreNotFoundError` y `PropensionServiceUnavailableError` definidas aquí
son el contrato de error único y compartido entre TODAS las implementaciones del protocolo
`PropensionClient` (esta implementación fake y `HttpPropensionClient`, el cliente HTTP real
de la tarea 5.1, que las importa desde este módulo). Ambas implementaciones deben lanzar
exactamente el mismo tipo ante obligación sin score (404) o servicio no disponible
(timeout/5xx/`ConnectionError`), de forma que la estrategia de fallback del `ContextBuilder`
(tarea 5.3) funcione idénticamente sin importar cuál implementación fue inyectada.
"""
from __future__ import annotations

from sistema_agentico.types import ScoreInfo

__all__ = [
    "FakePropensionClient",
    "PropensionClientError",
    "PropensionServiceUnavailableError",
    "ScoreNotFoundError",
]


class PropensionClientError(Exception):
    """Clase base de los errores del protocolo `PropensionClient`.

    Compartida entre `FakePropensionClient` y `HttpPropensionClient` para que el
    `ContextBuilder` pueda capturar cualquier fallo del cliente de propensión con un
    único `except PropensionClientError` si así lo requiere, además de poder distinguir
    el caso puntual (`ScoreNotFoundError` vs `PropensionServiceUnavailableError`).
    """

    def __init__(self, id_obligacion: str, mensaje: str) -> None:
        self.id_obligacion = id_obligacion
        super().__init__(mensaje)


class ScoreNotFoundError(PropensionClientError):
    """La obligación consultada no tiene un score registrado (equivalente a HTTP 404)."""

    def __init__(self, id_obligacion: str) -> None:
        super().__init__(id_obligacion, f"No existe score para id_obligacion={id_obligacion!r}")


class PropensionServiceUnavailableError(PropensionClientError):
    """El servicio de propensión no está disponible (timeout, 5xx o `ConnectionError`)."""

    def __init__(self, id_obligacion: str) -> None:
        super().__init__(id_obligacion, f"Servicio de propensión no disponible para id_obligacion={id_obligacion!r}")


class FakePropensionClient:
    """Implementación en memoria, determinista y sin dependencias externas de `PropensionClient`.

    Cumple estructuralmente el protocolo `PropensionClient` de `design.md` (método
    `get_score(id_obligacion, timeout_seconds) -> ScoreInfo`), pero no realiza ninguna
    llamada de red: resuelve cada consulta contra un diccionario `scores` en memoria,
    poblado en el constructor o mediante `register_score`.

    El parámetro `timeout_seconds` se acepta (para respetar la forma del protocolo y
    permitir inyección intercambiable con `HttpPropensionClient`) pero se ignora, ya que
    no hay ninguna operación bloqueante que pueda exceder ningún timeout.
    """

    def __init__(
        self,
        scores: dict[str, ScoreInfo] | None = None,
        *,
        simulate_unavailable: set[str] | None = None,
    ) -> None:
        """Inicializa el cliente fake.

        Args:
            scores: mapeo inicial `id_obligacion -> ScoreInfo` con el que responder a
                `get_score`. Se copia a un diccionario interno mutable; no se compromete
                la referencia recibida.
            simulate_unavailable: conjunto de `id_obligacion` para los cuales `get_score`
                debe simular indisponibilidad del servicio (`PropensionServiceUnavailableError`)
                en lugar de resolver contra `scores`, útil para ejercitar la estrategia de
                fallback del `ContextBuilder` (tarea 5.3) sin necesidad de un timeout o
                error de conexión real.
        """
        self._scores: dict[str, ScoreInfo] = dict(scores) if scores else {}
        self._simulate_unavailable: set[str] = set(simulate_unavailable) if simulate_unavailable else set()

    def register_score(self, id_obligacion: str, score_info: ScoreInfo) -> None:
        """Registra (o sobrescribe) el `ScoreInfo` que se retornará para `id_obligacion`."""
        if not id_obligacion:
            raise ValueError("id_obligacion no puede ser vacío")
        self._scores[id_obligacion] = score_info

    def simulate_unavailable_for(self, id_obligacion: str) -> None:
        """Marca `id_obligacion` para que `get_score` simule indisponibilidad del servicio."""
        if not id_obligacion:
            raise ValueError("id_obligacion no puede ser vacío")
        self._simulate_unavailable.add(id_obligacion)

    def clear_simulate_unavailable(self, id_obligacion: str) -> None:
        """Retira la simulación de indisponibilidad previamente configurada para `id_obligacion`."""
        self._simulate_unavailable.discard(id_obligacion)

    def get_score(self, id_obligacion: str, timeout_seconds: float) -> ScoreInfo:
        """Retorna el `ScoreInfo` registrado para `id_obligacion`.

        `timeout_seconds` se acepta por compatibilidad con el protocolo `PropensionClient`
        pero se ignora: esta implementación no realiza ninguna operación que pueda demorar
        ni bloquear.

        Raises:
            PropensionServiceUnavailableError: si `id_obligacion` fue marcado mediante
                `simulate_unavailable` (constructor o `simulate_unavailable_for`).
            ScoreNotFoundError: si `id_obligacion` no tiene un score registrado.
        """
        if id_obligacion in self._simulate_unavailable:
            raise PropensionServiceUnavailableError(id_obligacion)
        try:
            return self._scores[id_obligacion]
        except KeyError:
            raise ScoreNotFoundError(id_obligacion) from None

    @classmethod
    def with_default_synthetic_scores(cls, n: int = 20, seed: int | None = 0) -> "FakePropensionClient":
        """Construye un `FakePropensionClient` pre-poblado con `n` scores sintéticos.

        Convenience factory pensada como cliente de propensión por defecto para
        ejecución local (RNF-13): reutiliza `synthetic_data.profiles.generate_profiles`
        (tarea 4.1) para generar `n` `ClienteObligacionContext` sintéticos deterministas
        (dado `seed`) y registra el `ScoreInfo` de cada uno bajo su `id_obligacion`
        sintético (prefijo `SYN-OBL-`), de forma que un script de ejecución local pueda
        consultar cualquiera de esos ids sin depender del servicio `modelo-propension`
        real ni de uno mockeado por HTTP.
        """
        from sistema_agentico.synthetic_data.profiles import generate_profiles

        perfiles = generate_profiles(n, seed=seed)
        scores = {perfil.id_obligacion: perfil.score for perfil in perfiles}
        return cls(scores=scores)
