"""Agente Conversacional (`ConversationalAgent`) implementado sobre LangChain.

Ver Component 6 de `design.md` ("Implementation Note (LangChain)").
"""
from sistema_agentico.conversational.agent import (
    ConversationTurn,
    ConversationalAgent,
    ConversationalAgentPromptError,
    ConversationalAgentResponseError,
)
from sistema_agentico.conversational.chat_model_factory import (
    DEFAULT_API_KEY_ENV_VAR,
    DEFAULT_TIMEOUT_SECONDS,
    LLMAPIKeyMissingError,
    LLMConfigurationError,
    build_chat_openai,
)
from sistema_agentico.conversational.fake_chat_model import (
    DeterministicFakeChatModel,
    LLMResponseExhaustedError,
    PromptResponder,
)
from sistema_agentico.conversational.offer_extractor import (
    OfferMentionExtractionError,
    extract_offers_mentioned,
)

__all__ = [
    "ConversationTurn",
    "ConversationalAgent",
    "ConversationalAgentPromptError",
    "ConversationalAgentResponseError",
    "OfferMentionExtractionError",
    "extract_offers_mentioned",
    "DEFAULT_API_KEY_ENV_VAR",
    "DEFAULT_TIMEOUT_SECONDS",
    "DeterministicFakeChatModel",
    "LLMAPIKeyMissingError",
    "LLMConfigurationError",
    "LLMResponseExhaustedError",
    "PromptResponder",
    "build_chat_openai",
]
