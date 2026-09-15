"""Integración del Guardrail de Entrada (`InputGuardrail`), Component 1 de `design.md`.

Ver Component 1 de `design.md` y Requirement 2.5 de `requirements.md`.

Provee `InputGuardrail`, el único punto de entrada al `Orchestrator` tanto en modo
proactivo como en modo reactivo (RF-20). Compone `PIIMasker` (tarea 3.1) e
`InjectionRiskClassifier` (tarea 3.2) sobre el mensaje crudo del cliente y produce
siempre un `SanitizedInput`, sin excepción, para cualquier `RawInput` recibido.

En modo proactivo no existe mensaje de cliente (`RawInput.mensaje_crudo is None`):
no hay nada que enmascarar ni clasificar, por lo que `mensaje_sanitizado` es `None`
y los campos de riesgo/PII reflejan los valores neutros de "sin riesgo, sin PII"
(`pii_detectada=[]`, `riesgo_inyeccion=0.0`, `bloqueado=False`).

En modo reactivo, `InputGuardrail.process` sigue una secuencia fija de dos pasos
sobre `mensaje_crudo`:
1. `PIIMasker.mask`: produce el texto enmascarado que se usará como
   `mensaje_sanitizado` (Requirement 2.1). El texto crudo con PII nunca se propaga
   más allá de este punto.
2. `InjectionRiskClassifier.evaluate`: se aplica sobre el mensaje ya enmascarado por
   PII (nunca sobre el texto crudo), estimando `riesgo_inyeccion` y derivando
   `bloqueado` de forma fail-closed (Requirements 2.2, 2.3). El resultado
   (`riesgo_inyeccion`, `bloqueado`) del clasificador se propaga sin modificación al
   `SanitizedInput` final.
"""
from __future__ import annotations

from sistema_agentico.guardrails.injection_risk import (
    DEFAULT_INJECTION_RISK_THRESHOLD,
    InjectionRiskClassifier,
)
from sistema_agentico.guardrails.pii_masker import PIIMasker
from sistema_agentico.types import InputMode, RawInput, SanitizedInput


class InputGuardrail:
    """Único punto de entrada al `Orchestrator`, para ambos modos (Requirement 2.5).

    Ni el runner de modo proactivo ni el de modo reactivo pueden saltarse este
    componente: ambos deben construir un `RawInput` y pasarlo por
    `InputGuardrail.process` para obtener el `SanitizedInput` que consume
    `Orchestrator.handle_interaction` (RF-20).
    """

    def __init__(
        self,
        pii_masker: PIIMasker | None = None,
        injection_risk_classifier: InjectionRiskClassifier | None = None,
    ) -> None:
        self._pii_masker = pii_masker or PIIMasker()
        self._injection_risk_classifier = (
            injection_risk_classifier
            or InjectionRiskClassifier(threshold=DEFAULT_INJECTION_RISK_THRESHOLD)
        )

    def process(self, raw_input: RawInput) -> SanitizedInput:
        """Sanitiza PII y evalúa riesgo de inyección/jailbreak sobre `raw_input`.

        Modo proactivo (`raw_input.mensaje_crudo is None`): no hay mensaje de
        cliente que sanitizar; retorna un `SanitizedInput` con
        `mensaje_sanitizado=None` y los valores neutros de "sin riesgo, sin PII".

        Modo reactivo: aplica `PIIMasker.mask` sobre `mensaje_crudo` y luego
        `InjectionRiskClassifier.evaluate` sobre el texto ya enmascarado, propagando
        sus resultados (`riesgo_inyeccion`, `bloqueado`) al `SanitizedInput`
        producido.
        """
        if raw_input.mensaje_crudo is None:
            return SanitizedInput(
                mode=raw_input.mode,
                id_obligacion=raw_input.id_obligacion,
                mensaje_sanitizado=None,
                pii_detectada=[],
                riesgo_inyeccion=0.0,
                bloqueado=False,
            )

        mask_result = self._pii_masker.mask(raw_input.mensaje_crudo)
        risk_result = self._injection_risk_classifier.evaluate(mask_result.texto_enmascarado)

        return SanitizedInput(
            mode=raw_input.mode,
            id_obligacion=raw_input.id_obligacion,
            mensaje_sanitizado=mask_result.texto_enmascarado,
            pii_detectada=mask_result.pii_detectada,
            riesgo_inyeccion=risk_result.riesgo_inyeccion,
            bloqueado=risk_result.bloqueado,
        )
