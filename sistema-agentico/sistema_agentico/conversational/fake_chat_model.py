"""``BaseChatModel`` fake, determinista y local para pruebas del agente conversacional.

No realiza llamadas de red, acceso a archivos ni invocaciones a proveedores externos
(RNF-13). Preserva la ergonomía de pruebas del antiguo ``FakeLLMClient`` (tarea 9.1)
sobre la interfaz real de LangChain (``BaseChatModel``), de modo que
``ConversationalAgent`` pueda inyectar indistintamente este fake o un backend real
(``ChatOpenAI``) sin cambiar su contrato público.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

__all__ = ["DeterministicFakeChatModel", "LLMResponseExhaustedError", "PromptResponder"]

PromptResponder = Callable[[str], str]
"""Función determinista que transforma el prompt renderizado en una respuesta de texto."""


class LLMResponseExhaustedError(RuntimeError):
    """La secuencia configurada de respuestas del fake ya no tiene elementos."""


class DeterministicFakeChatModel(BaseChatModel):
    """Implementación local, determinista e inspeccionable de ``BaseChatModel``.

    Configure exactamente uno de estos modos:

    - ``response``: devuelve siempre el mismo texto.
    - ``responses``: devuelve cada texto en orden y, al agotarse, lanza
      :class:`LLMResponseExhaustedError`.
    - ``responder``: calcula el texto a partir del prompt renderizado (todos los
      mensajes concatenados) mediante una función inyectada.

    Cada invocación registra el prompt renderizado en ``prompts`` antes de resolver
    la respuesta, inclusive cuando una secuencia ya está agotada. No se efectúa E/S
    de ningún tipo.
    """

    model_config = {"arbitrary_types_allowed": True}

    response: Optional[str] = None
    responses: Optional[tuple[str, ...]] = None
    responder: Optional[PromptResponder] = None

    _prompts: list[str] = PrivateAttr(default_factory=list)
    _next_response_index: int = PrivateAttr(default=0)

    def __init__(self, **data: Any) -> None:
        responses = data.get("responses")
        if responses is not None:
            data["responses"] = tuple(responses)
        super().__init__(**data)
        configured_modes = sum(
            value is not None for value in (self.response, self.responses, self.responder)
        )
        if configured_modes != 1:
            raise ValueError(
                "configure exactamente uno de response, responses o responder"
            )
        if self.responses is not None and not self.responses:
            raise ValueError("responses debe contener al menos una respuesta")

    @property
    def _llm_type(self) -> str:
        return "deterministic-fake-chat-model"

    @property
    def prompts(self) -> list[str]:
        """Prompts renderizados (todos los mensajes concatenados), en orden de llamada."""
        return list(self._prompts)

    @property
    def call_count(self) -> int:
        """Cantidad de invocaciones hechas al modelo, incluidas las agotadas."""
        return len(self._prompts)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        rendered_prompt = self._render_prompt(messages)
        self._prompts.append(rendered_prompt)

        if self.response is not None:
            text = self.response
        elif self.responses is not None:
            if self._next_response_index >= len(self.responses):
                raise LLMResponseExhaustedError(
                    "la secuencia configurada de respuestas del fake está agotada"
                )
            text = self.responses[self._next_response_index]
            self._next_response_index += 1
        else:
            assert self.responder is not None
            text = self.responder(rendered_prompt)
            if not isinstance(text, str):
                raise TypeError("responder debe retornar str")

        generation = ChatGeneration(message=AIMessage(content=text))
        return ChatResult(generations=[generation])

    @staticmethod
    def _render_prompt(messages: Sequence[BaseMessage]) -> str:
        return "\n".join(f"{message.type}: {message.content}" for message in messages)
