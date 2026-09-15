"""Pruebas unitarias del ``BaseChatModel`` fake local (tarea 9.7 / migración LangChain)."""
from __future__ import annotations

import pytest
from langchain_core.language_models.chat_models import BaseChatModel

from sistema_agentico.conversational import (
    DeterministicFakeChatModel,
    LLMResponseExhaustedError,
)


def test_fixed_response_is_reused_and_prompts_are_captured() -> None:
    model = DeterministicFakeChatModel(response="respuesta fija")

    assert model.invoke("primer prompt").content == "respuesta fija"
    assert model.invoke("segundo prompt").content == "respuesta fija"
    assert model.prompts == ["human: primer prompt", "human: segundo prompt"]
    assert model.call_count == 2
    assert isinstance(model, BaseChatModel)


def test_sequence_is_consumed_in_order_and_exhaustion_is_explicit() -> None:
    model = DeterministicFakeChatModel(responses=["primera", "segunda"])

    assert model.invoke("p1").content == "primera"
    assert model.invoke("p2").content == "segunda"
    with pytest.raises(LLMResponseExhaustedError, match="agotada"):
        model.invoke("p3")

    assert model.prompts == ["human: p1", "human: p2", "human: p3"]
    assert model.call_count == 3


def test_responder_can_return_a_deterministic_response_from_prompt() -> None:
    model = DeterministicFakeChatModel(responder=lambda prompt: f"eco:{prompt.upper()}")

    assert model.invoke("reintenta").content == "eco:HUMAN: REINTENTA"
    assert model.prompts == ["human: reintenta"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"response": "fija", "responses": ["secuencia"]},
        {"responses": []},
    ],
)
def test_requires_exactly_one_non_empty_response_mode(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        DeterministicFakeChatModel(**kwargs)  # type: ignore[arg-type]
