"""Heurísticas de riesgo de inyección/jailbreak y bloqueo fail-closed (Component 1: `InputGuardrail`).

Ver Component 1 de `design.md` y Requirements 2.2, 2.3 de `requirements.md`.

Provee `InjectionRiskClassifier`, responsable de estimar `riesgo_inyeccion` (un `float`
en `[0.0, 1.0]`) sobre el mensaje del cliente y de derivar la decisión `bloqueado`
comparando dicho riesgo contra un umbral configurable (Requirement 2.2). La tarea 3.3
integra este clasificador dentro del pipeline completo de `InputGuardrail.process()`;
este módulo únicamente resuelve la responsabilidad de clasificación de riesgo y la
lógica de bloqueo, de forma aislada y testeable.

Enfoque deliberadamente basado en reglas/regex + scoring heurístico (sin modelo de ML),
adecuado para un prototipo en contexto financiero/cobranza en español y consistente con
RNF-15 (alcance acotado al tiempo disponible). Ver Property 4 de `design.md`.

Fail-closed (Requirement 2.3, RNF-02): `InjectionRiskClassifier.evaluate` nunca propaga
una excepción. Si la evaluación de heurísticas falla por cualquier motivo (entrada
inesperada, error en un patrón, etc.), se retorna de forma conservadora
`riesgo_inyeccion=1.0` y `bloqueado=True`, en lugar de permitir que una entrada de alto
riesgo llegue sin marcar al `Orchestrator`/`ConversationalAgent`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Umbral por defecto de bloqueo. En este prototipo se define como constante de módulo
# (no hay todavía un artefacto de configuración externo para `InputGuardrail`, a
# diferencia de `EligibilityEngine`/`ConversationalAgent` que sí lo requieren
# explícitamente en Requirement 14.4); queda inyectable vía el constructor de
# `InjectionRiskClassifier` para facilitar su futura externalización.
DEFAULT_INJECTION_RISK_THRESHOLD = 0.5

# Riesgo conservador asignado cuando la evaluación de heurísticas falla (fail-closed).
FAIL_CLOSED_RISK = 1.0


@dataclass(frozen=True)
class InjectionRiskResult:
    """Resultado de `InjectionRiskClassifier.evaluate`.

    `bloqueado` es una función determinista de `riesgo_inyeccion` y el umbral
    configurado: `bloqueado == (riesgo_inyeccion > threshold)` en el caso normal, o
    `bloqueado == True` (con `riesgo_inyeccion == FAIL_CLOSED_RISK`) si la evaluación
    de heurísticas falló (Property 4, Requirements 2.2, 2.3).
    """

    riesgo_inyeccion: float
    bloqueado: bool

    def __post_init__(self) -> None:
        if not (0.0 <= self.riesgo_inyeccion <= 1.0):
            raise ValueError("riesgo_inyeccion debe estar en [0.0, 1.0]")


class InjectionRiskClassifier:
    """Estima `riesgo_inyeccion` sobre texto y deriva `bloqueado` contra un umbral.

    Heurísticas de detección orientadas a un chatbot financiero/de cobranza en
    español: patrones de intento de anulación de instrucciones ("ignora las
    instrucciones anteriores", "olvida tus reglas"), suplantación de rol o
    jailbreak ("actúa como", "eres ahora", "finge que eres", "modo desarrollador"),
    solicitudes de exposición del prompt de sistema ("system prompt", "revela tus
    instrucciones") y presencia anómala de caracteres de control. Cada patrón
    coincidente suma un peso fijo a un score acumulado, acotado a `[0.0, 1.0]`.

    Es deliberadamente conservador (mismo criterio que `PIIMasker`): prioriza no
    dejar pasar un intento de manipulación (aceptando falsos positivos ocasionales)
    sobre no bloquear de más (RNF-02).
    """

    # --- Señales fuertes de anulación de instrucciones / jailbreak explícito ----
    _STRONG_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"ignora\s+(todas\s+)?(las\s+)?instrucciones\s*(anteriores|previas|del\s+sistema)?"),
        re.compile(r"olvid[a|e]\s+(todas\s+)?(tus\s+|las\s+)?(reglas|instrucciones|restricciones)"),
        re.compile(r"eres\s+ahora\s+"),
        re.compile(r"no\s+tienes\s+(ninguna\s+)?(restricci[oó]n|restricciones|l[ií]mites?)"),
        re.compile(r"sin\s+restricciones\s*(ni\s+filtros)?"),
        re.compile(r"desactiva\s+(tus\s+|los\s+)?filtros"),
        re.compile(r"modo\s+desarrollador"),
        re.compile(r"(revela|muestra(me)?|dime|cu[aá]l\s+es)\s+(tu|el|tus)\s+(system\s+)?prompt"),
        re.compile(r"(revela|muestra(me)?|dime)\s+(tus\s+)?(instrucciones|reglas)\s+(internas|del\s+sistema)"),
    )
    _STRONG_WEIGHT = 0.6

    # --- Señales moderadas de suplantación de rol / role-play ------------------
    _MODERATE_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"act[uú]a\s+como\s+"),
        re.compile(r"finge\s+que\s+eres"),
        re.compile(r"haz\s+de\s+cuenta\s+que\s+eres"),
        re.compile(r"pretend(e)?\s+(que\s+eres|to\s+be)"),
        re.compile(r"role[\s\-]?play"),
        re.compile(r"system\s*prompt"),
    )
    _MODERATE_WEIGHT = 0.35

    # --- Umbral de caracteres de control anómalos (excluyendo \n, \t, \r) -------
    _CONTROL_CHAR_THRESHOLD = 3
    _CONTROL_CHAR_WEIGHT = 0.3

    def __init__(self, threshold: float = DEFAULT_INJECTION_RISK_THRESHOLD) -> None:
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("threshold debe estar en [0.0, 1.0]")
        self._threshold = threshold

    @property
    def threshold(self) -> float:
        """Umbral configurado contra el cual se compara `riesgo_inyeccion` para bloquear."""
        return self._threshold

    def evaluate(self, texto: str | None) -> InjectionRiskResult:
        """Evalúa `texto` y retorna `riesgo_inyeccion` + `bloqueado`; nunca lanza.

        Camino normal: `riesgo_inyeccion` se calcula mediante heurísticas y
        `bloqueado = riesgo_inyeccion > self.threshold` (Requirement 2.2).

        Camino fail-closed (Requirement 2.3, RNF-02): si el cálculo de heurísticas
        lanza cualquier excepción, se captura aquí mismo y se retorna
        `InjectionRiskResult(riesgo_inyeccion=FAIL_CLOSED_RISK, bloqueado=True)` en
        lugar de propagar el error o de retornar un resultado no bloqueado. Esto
        garantiza que ningún fallo del mecanismo de evaluación pueda resultar en una
        entrada de alto riesgo sin marcar llegando al `ConversationalAgent`.
        """
        try:
            riesgo = self._score(texto)
            bloqueado = riesgo > self._threshold
            return InjectionRiskResult(riesgo_inyeccion=riesgo, bloqueado=bloqueado)
        except Exception:
            return InjectionRiskResult(riesgo_inyeccion=FAIL_CLOSED_RISK, bloqueado=True)

    def _score(self, texto: str | None) -> float:
        """Calcula el score de riesgo acumulado en `[0.0, 1.0]`; puede lanzar excepciones."""
        if not texto:
            return 0.0

        texto_normalizado = texto.lower()
        score = 0.0

        for pattern in self._STRONG_PATTERNS:
            if pattern.search(texto_normalizado):
                score += self._STRONG_WEIGHT

        for pattern in self._MODERATE_PATTERNS:
            if pattern.search(texto_normalizado):
                score += self._MODERATE_WEIGHT

        caracteres_control = sum(
            1 for c in texto if ord(c) < 32 and c not in ("\n", "\t", "\r")
        )
        if caracteres_control > self._CONTROL_CHAR_THRESHOLD:
            score += self._CONTROL_CHAR_WEIGHT

        return min(score, 1.0)
