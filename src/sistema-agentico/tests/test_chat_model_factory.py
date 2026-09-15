"""Pruebas locales de la fábrica del backend real ``ChatOpenAI`` (tarea 9.7).

No se realiza ninguna llamada de red: sólo se valida la configuración y la
resolución segura de la clave de API antes de construir el ``BaseChatModel``.
"""
from __future__ import annotations

import pytest
from langchain_openai import ChatOpenAI

from sistema_agentico.conversational import (
    LLMAPIKeyMissingError,
    LLMConfigurationError,
    build_chat_openai,
)


def test_builds_chat_openai_with_validated_configuration() -> None:
    chat_model = build_chat_openai(
        base_url="https://provider.example/v1/",
        model="modelo-prueba",
        api_key="test-api-key",
        timeout_seconds=4.5,
    )

    assert isinstance(chat_model, ChatOpenAI)
    assert chat_model.model_name == "modelo-prueba"
    assert chat_model.openai_api_base == "https://provider.example/v1"
    assert "test-api-key" not in repr(chat_model)


def test_api_key_can_be_resolved_from_an_injected_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_LLM_API_KEY", "key-from-environment")

    chat_model = build_chat_openai(
        base_url="https://provider.example/v1",
        model="modelo-prueba",
        api_key_env_var="TEST_LLM_API_KEY",
    )

    assert isinstance(chat_model, ChatOpenAI)
    assert "key-from-environment" not in repr(chat_model)


def test_missing_api_key_is_explicit_and_does_not_disclose_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MISSING_TEST_LLM_API_KEY", raising=False)

    with pytest.raises(LLMAPIKeyMissingError) as error:
        build_chat_openai(
            base_url="https://provider.example/v1",
            model="modelo-prueba",
            api_key_env_var="MISSING_TEST_LLM_API_KEY",
        )

    assert "clave de API" in str(error.value)
    assert "MISSING_TEST_LLM_API_KEY" not in str(error.value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"base_url": "provider.example", "model": "modelo", "api_key": "key"},
        {"base_url": "https://provider.example?token=secret", "model": "modelo", "api_key": "key"},
        {"base_url": "https://provider.example", "model": " ", "api_key": "key"},
        {"base_url": "https://provider.example", "model": "modelo", "api_key": "key", "timeout_seconds": 0},
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(LLMConfigurationError):
        build_chat_openai(**kwargs)  # type: ignore[arg-type]
