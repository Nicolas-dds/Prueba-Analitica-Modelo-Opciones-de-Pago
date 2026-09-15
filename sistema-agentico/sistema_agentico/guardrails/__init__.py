"""Guardrail de Entrada (`InputGuardrail`) y Guardrail de Salida (`OutputGuardrail`).

Ver Component 1 y Component 7 de `design.md`.
"""
from sistema_agentico.guardrails.pii import (
    PII_TIPO_CEDULA,
    PII_TIPO_CUENTA,
    PII_TIPO_NOMBRE,
    PII_TIPO_TELEFONO,
    PII_TIPOS_SOPORTADOS,
    mask_pii,
)

__all__ = [
    "PII_TIPO_CEDULA",
    "PII_TIPO_CUENTA",
    "PII_TIPO_NOMBRE",
    "PII_TIPO_TELEFONO",
    "PII_TIPOS_SOPORTADOS",
    "mask_pii",
]
from sistema_agentico.guardrails.pii_masker import PIIMasker, PIIMaskResult, PIIType
from sistema_agentico.guardrails.injection_risk import (
    DEFAULT_INJECTION_RISK_THRESHOLD,
    FAIL_CLOSED_RISK,
    InjectionRiskClassifier,
    InjectionRiskResult,
)
from sistema_agentico.guardrails.input_guardrail import InputGuardrail
from sistema_agentico.guardrails.output_guardrail import OutputGuardrail
from sistema_agentico.types import RawInput

__all__ = [
    "PIIMasker",
    "PIIMaskResult",
    "PIIType",
    "DEFAULT_INJECTION_RISK_THRESHOLD",
    "FAIL_CLOSED_RISK",
    "InjectionRiskClassifier",
    "InjectionRiskResult",
    "InputGuardrail",
    "OutputGuardrail",
    "RawInput",
]
