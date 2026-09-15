"""Configuración basada en variables de entorno para `sistema-agentico`.

Ver tarea 17.2 del plan de implementación (`.kiro/specs/sistema-agentico-cobranza/tasks.md`):
carga de configuración sensible (API key del LLM, URL del servicio `modelo-propension`)
desde un archivo `.env` opcional, con `.env.example` documentado y sin commitear
secretos reales.

Este módulo NUNCA lee ni expone el valor de variables sensibles (p.ej. la API key del
LLM) en su propio namespace, en logs, en su `repr` ni en mensajes de error:
`build_chat_openai` (`conversational/chat_model_factory.py`) sigue siendo el único
punto que resuelve `OPENAI_API_KEY` desde `os.environ`, por nombre de variable, en el
momento en que se invoca. Importar este módulo sólo garantiza que, si existe un
archivo `.env` en la raíz del servicio (`src/sistema-agentico/.env`), sus pares
clave-valor ya estén cargados en `os.environ` *antes* de esa resolución.

Si el archivo `.env` no existe (p.ej. en producción/Docker, donde las variables se
inyectan directamente en el entorno del contenedor), `load_dotenv` no falla ni exige
el archivo: es el comportamiento estándar de `python-dotenv` no hacer nada en ese caso.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

__all__ = [
    "DOTENV_PATH",
    "OPENAI_API_KEY_ENV_VAR",
    "PROPENSION_SERVICE_URL",
    "RAW_DATA_DIR",
    "SISTEMA_AGENTICO_HOST",
    "SISTEMA_AGENTICO_PORT",
    "SISTEMA_AGENTICO_PROMPT_VERSION",
]

# Raíz del servicio: `src/sistema-agentico/` (carpeta padre del paquete `sistema_agentico/`).
_SERVICE_ROOT = Path(__file__).resolve().parent.parent
DOTENV_PATH = _SERVICE_ROOT / ".env"

# No falla si `DOTENV_PATH` no existe (comportamiento por defecto de python-dotenv).
# `override=False` preserva variables ya presentes en el entorno real del proceso
# (p.ej. inyectadas por Docker/CI), dándoles prioridad sobre el archivo `.env` local.
load_dotenv(dotenv_path=DOTENV_PATH, override=False)

# Nombre (no el valor) de la variable de entorno que contiene la API key del proveedor
# LLM. La resolución real del valor ocurre en
# `conversational.chat_model_factory.build_chat_openai` (parámetro `api_key_env_var`),
# que lee `os.environ` por nombre y nunca expone el valor resuelto.
OPENAI_API_KEY_ENV_VAR = "OPENAI_API_KEY"

# URL base del servicio `modelo-propension` (Requirements 3.1, 14.3). En Docker Compose
# (tarea 17.4) debe apuntar al nombre del servicio Docker, no a `localhost`.
_DEFAULT_PROPENSION_SERVICE_URL = "http://localhost:8000"
PROPENSION_SERVICE_URL = os.environ.get(
    "PROPENSION_SERVICE_URL", _DEFAULT_PROPENSION_SERVICE_URL
)

# Versión de prompt del agente conversacional (Requirement 6.1: versionado de `prompt_version`).
_DEFAULT_PROMPT_VERSION = "v1"
SISTEMA_AGENTICO_PROMPT_VERSION = os.environ.get(
    "SISTEMA_AGENTICO_PROMPT_VERSION", _DEFAULT_PROMPT_VERSION
)

# Host/puerto de la futura API HTTP de `sistema-agentico` (tarea 17.3). Puerto distinto
# al de `modelo-propension` (8000) para evitar colisión al levantar ambos servicios
# localmente o en la misma red de Docker Compose.
_DEFAULT_HOST = "0.0.0.0"
_DEFAULT_PORT = "8001"
SISTEMA_AGENTICO_HOST = os.environ.get("SISTEMA_AGENTICO_HOST", _DEFAULT_HOST)
SISTEMA_AGENTICO_PORT = int(os.environ.get("SISTEMA_AGENTICO_PORT", _DEFAULT_PORT))

# Directorio con los 5 CSV crudos usados por `modelo-propension` (ver
# `pipeline/ingest.py` de ese servicio), consumido por
# `context.csv_profile_provider.CsvObligacionProfileProvider` (tarea 19: perfil real de
# obligación, extensión fuera del alcance original de `requirements.md`/`design.md`).
# El default resuelve correctamente para desarrollo local: `src/sistema-agentico/` ->
# `../../data` == la carpeta `data/` en la raíz del repo. Se ancla a `_SERVICE_ROOT`
# (no al directorio de trabajo del proceso) para que el valor por defecto no dependa de
# desde dónde se invoque el proceso (p.ej. `pytest` corrido desde la raíz del repo en
# vez de desde `src/sistema-agentico/`). En Docker (tarea 19.3, no implementada aquí)
# este valor deberá sobrescribirse vía variable de entorno para apuntar a un volumen
# montado (p.ej. `/data`) -- NO se hardcodea esa ruta de Docker como default aquí.
_DEFAULT_RAW_DATA_DIR = str((_SERVICE_ROOT / ".." / ".." / "data").resolve())
RAW_DATA_DIR = os.environ.get("RAW_DATA_DIR", _DEFAULT_RAW_DATA_DIR)
