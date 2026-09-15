"""Fábrica del backend real de chat (LangChain ``ChatOpenAI``) para el agente.

No implementa transporte propio: delega en ``langchain_openai.ChatOpenAI`` (que a su
vez usa el SDK oficial de OpenAI) toda la comunicación de red. Esta fábrica sólo
valida la configuración de entrada y resuelve la clave de API exactamente con las
mismas reglas de seguridad que el adaptador retirado (nunca se expone en logs,
``repr`` ni mensajes de error), antes de construir el ``BaseChatModel`` inyectable en
``ConversationalAgent``.
"""
from __future__ import annotations

import math
import os
import urllib.parse

from langchain_openai import ChatOpenAI

__all__ = [
    "DEFAULT_API_KEY_ENV_VAR",
    "DEFAULT_TIMEOUT_SECONDS",
    "LLMAPIKeyMissingError",
    "LLMConfigurationError",
    "build_chat_openai",
]

DEFAULT_API_KEY_ENV_VAR = "OPENAI_API_KEY"
DEFAULT_TIMEOUT_SECONDS = 30.0


class LLMConfigurationError(ValueError):
    """La configuración no permite construir un ``BaseChatModel`` real y seguro."""


class LLMAPIKeyMissingError(LLMConfigurationError):
    """No se proporcionó una clave de API utilizable."""


def build_chat_openai(
    *,
    base_url: str,
    model: str,
    api_key: str | None = None,
    api_key_env_var: str | None = DEFAULT_API_KEY_ENV_VAR,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> ChatOpenAI:
    """Construye un ``ChatOpenAI`` validado para inyectar en ``ConversationalAgent``.

    ``api_key`` tiene prioridad sobre ``api_key_env_var``. La clave resuelta se pasa
    directamente a ``ChatOpenAI``, que la conserva como ``SecretStr`` y no la revela
    en su representación ni en los mensajes de error de transporte.
    """
    validated_base_url = _validate_base_url(base_url)
    validated_model = _validate_model(model)
    validated_timeout = _validate_timeout(timeout_seconds)
    resolved_api_key = _resolve_api_key(api_key, api_key_env_var)

    return ChatOpenAI(
        model=validated_model,
        api_key=resolved_api_key,
        base_url=validated_base_url,
        timeout=validated_timeout,
    )


def _validate_base_url(base_url: str) -> str:
    if not isinstance(base_url, str) or not base_url.strip():
        raise LLMConfigurationError("base_url debe ser una URL HTTP(S) no vacía")
    parsed = urllib.parse.urlsplit(base_url.strip())
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise LLMConfigurationError("base_url debe ser una URL HTTP(S) base válida")
    path = parsed.path.rstrip("/")
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _validate_model(model: str) -> str:
    if not isinstance(model, str) or not model.strip():
        raise LLMConfigurationError("model debe ser un identificador no vacío")
    return model.strip()


def _validate_timeout(timeout_seconds: float) -> float:
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise LLMConfigurationError("timeout_seconds debe ser un número finito mayor que cero")
    return float(timeout_seconds)


def _resolve_api_key(api_key: str | None, api_key_env_var: str | None) -> str:
    if api_key is not None:
        if isinstance(api_key, str) and api_key.strip():
            return api_key
        raise LLMAPIKeyMissingError("no se configuró una clave de API para el proveedor LLM")
    if not isinstance(api_key_env_var, str) or not api_key_env_var.strip():
        raise LLMConfigurationError("api_key_env_var debe ser el nombre de una variable válida")
    resolved_key = os.environ.get(api_key_env_var)
    if not isinstance(resolved_key, str) or not resolved_key.strip():
        raise LLMAPIKeyMissingError("no se configuró una clave de API para el proveedor LLM")
    return resolved_key
