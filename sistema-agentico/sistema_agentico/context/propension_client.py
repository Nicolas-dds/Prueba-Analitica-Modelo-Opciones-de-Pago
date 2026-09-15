"""Cliente HTTP real hacia el servicio `modelo-propension` (`HttpPropensionClient`).

Ver Component 3 de `design.md` y Requirements 3.1, 14.3 de `requirements.md`.

Implementa estructuralmente el protocolo `PropensionClient` de `design.md`
(`get_score(id_obligacion, timeout_seconds) -> ScoreInfo`) invocando
`GET {base_url}/score/{id_obligacion}` del servicio de inferencia (`api/main.py` de
`modelo-propension`), cuyo contrato de respuesta es `schemas/contracts.py: ScoreOutput`
(`id_obligacion`, `score` en `[0,1]`, `decil` en `[1,10]`, `version_modelo`,
`fecha_calificacion`).

Esta clase tiene una única responsabilidad: hablar HTTP y traducir los distintos
resultados (200, 404, 5xx, timeout, error de conexión) en un `ScoreInfo` o en una
excepción tipada. NO decide ni aplica fallback: esa lógica pertenece a
`FallbackScoreStrategy` y a `ContextBuilder.build` (Requirement 3.2, RF-22), que
deben capturar `ScoreNotFoundError` / `PropensionServiceUnavailableError` lanzadas
por este cliente (y por `FakePropensionClient`, que lanza las mismas clases).

Las excepciones (`PropensionClientError`, `ScoreNotFoundError`,
`PropensionServiceUnavailableError`) se importan desde
`sistema_agentico.context.fake_propension_client`, que es donde se definen como el
contrato de error único y compartido entre todas las implementaciones del protocolo
`PropensionClient` (real y fake): así `ContextBuilder`/`FallbackScoreStrategy`
(tareas 5.3/5.4) pueden capturar exactamente el mismo tipo (y acceder a
`exc.id_obligacion`) sin importar qué implementación fue inyectada.

Se usa exclusivamente la librería estándar (`urllib.request`) para mantener
`sistema_agentico` sin dependencias externas adicionales, consistente con el
prototipo local y autocontenido (RNF-13).
"""
from __future__ import annotations

import json
import socket
import urllib.error
import urllib.parse
import urllib.request

from sistema_agentico.context.fake_propension_client import (
    PropensionClientError,
    PropensionServiceUnavailableError,
    ScoreNotFoundError,
)
from sistema_agentico.types import ScoreInfo

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_TIMEOUT_SECONDS",
    "HttpPropensionClient",
    "PropensionClientError",
    "PropensionServiceUnavailableError",
    "ScoreNotFoundError",
]

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT_SECONDS = 2.0  # timeout corto por defecto (Requirement 14.3, RNF-09)


class HttpPropensionClient:
    """Implementación concreta de `PropensionClient` contra la API real de `modelo-propension`.

    Ver `design.md`, Component 3:
    ```python
    class PropensionClient(Protocol):
        def get_score(self, id_obligacion: str, timeout_seconds: float) -> ScoreInfo: ...
    ```
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        default_timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """
        Args:
            base_url: URL base del servicio `modelo-propension` (p.ej. `http://localhost:8000`).
            default_timeout_seconds: timeout por defecto, corto, usado cuando `get_score` no
                recibe uno explícito (Requirement 14.3: "timeout configurable de corta duración").
        """
        self._base_url = base_url.rstrip("/")
        self._default_timeout_seconds = default_timeout_seconds

    def get_score(
        self,
        id_obligacion: str,
        timeout_seconds: float | None = None,
    ) -> ScoreInfo:
        """Invoca `GET {base_url}/score/{id_obligacion}` y retorna el `ScoreInfo` real.

        Args:
            id_obligacion: identificador de la obligación a calificar.
            timeout_seconds: timeout de la llamada HTTP; si es `None` se usa
                `default_timeout_seconds` (Requirement 14.3).

        Returns:
            `ScoreInfo` con `degradado=False` (una respuesta exitosa nunca es degradada).

        Raises:
            ScoreNotFoundError: si el servicio responde 404 (obligación sin score).
            PropensionServiceUnavailableError: ante timeout, error de conexión o 5xx.
        """
        timeout = timeout_seconds if timeout_seconds is not None else self._default_timeout_seconds
        # `id_obligacion` puede contener caracteres reservados en una URL (p.ej. '#', que
        # delimita un fragmento); se codifica con `quote` (safe="") para que viajen como
        # parte del path y no sean truncados/interpretados por el cliente HTTP.
        url = f"{self._base_url}/score/{urllib.parse.quote(id_obligacion, safe='')}"
        request = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise ScoreNotFoundError(id_obligacion) from exc
            raise PropensionServiceUnavailableError(id_obligacion) from exc
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as exc:
            raise PropensionServiceUnavailableError(id_obligacion) from exc

        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise PropensionServiceUnavailableError(id_obligacion) from exc

        try:
            return ScoreInfo(
                score=payload["score"],
                decil=payload["decil"],
                version_modelo=payload["version_modelo"],
                fecha_calificacion=str(payload["fecha_calificacion"]),
                degradado=False,
            )
        except KeyError as exc:
            raise PropensionServiceUnavailableError(id_obligacion) from exc
