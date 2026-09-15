"""Gestor Humano y Auditor (`HumanReviewInterface`).

Ver Component 10 de `design.md`.
"""
from sistema_agentico.human_review.interface import (
    CaseAlreadyResolvedError,
    EscalationCaseSource,
    GoldenScenarioSink,
    HumanReviewInputError,
    HumanReviewInterface,
    OriginalTraceIntegrityError,
    UnknownEscalationCaseError,
)

__all__ = [
    "CaseAlreadyResolvedError",
    "EscalationCaseSource",
    "GoldenScenarioSink",
    "HumanReviewInputError",
    "HumanReviewInterface",
    "OriginalTraceIntegrityError",
    "UnknownEscalationCaseError",
]
