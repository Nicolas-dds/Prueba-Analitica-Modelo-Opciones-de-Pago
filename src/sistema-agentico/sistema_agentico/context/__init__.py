"""Context Builder (`ContextBuilder`) e integración con el modelo de propensión.

Ver Component 3 de `design.md`.
"""
from sistema_agentico.context.context_builder import (
    ContextBuilder,
    ContextEscalationRequired,
    InMemoryObligacionProfileProvider,
    ObligacionProfileData,
    ObligacionProfileNotFoundError,
    ObligacionProfileProvider,
)
from sistema_agentico.context.csv_profile_provider import (
    CsvObligacionProfileProvider,
    RawDataUnavailableError,
    TRTEST_FILENAME,
)
from sistema_agentico.context.fake_propension_client import (
    FakePropensionClient,
    PropensionClientError,
    PropensionServiceUnavailableError,
    ScoreNotFoundError,
)
from sistema_agentico.context.fallback_score_strategy import (
    MAX_ANTIGUEDAD_CACHE_DIAS,
    SCORE_NEUTRO_CONSERVADOR,
    FallbackScoreStrategy,
)
from sistema_agentico.context.propension_client import (
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT_SECONDS,
    HttpPropensionClient,
)

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_TIMEOUT_SECONDS",
    "ContextBuilder",
    "ContextEscalationRequired",
    "CsvObligacionProfileProvider",
    "FakePropensionClient",
    "FallbackScoreStrategy",
    "HttpPropensionClient",
    "InMemoryObligacionProfileProvider",
    "MAX_ANTIGUEDAD_CACHE_DIAS",
    "ObligacionProfileData",
    "ObligacionProfileNotFoundError",
    "ObligacionProfileProvider",
    "PropensionClientError",
    "PropensionServiceUnavailableError",
    "RawDataUnavailableError",
    "SCORE_NEUTRO_CONSERVADOR",
    "ScoreNotFoundError",
    "TRTEST_FILENAME",
]
