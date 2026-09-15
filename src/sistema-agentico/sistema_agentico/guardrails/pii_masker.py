"""Enmascaramiento de PII sobre el mensaje crudo (Component 1: `InputGuardrail`).

Ver Component 1 de `design.md` y Requirement 2.1 de `requirements.md`.

Provee `PIIMasker`, responsable de detectar y enmascarar información personal
identificable (nombres, cédulas, teléfonos, números de cuenta) en el texto crudo
de un mensaje ANTES de que dicho texto sea entregado al `Orchestrator`, a cualquier
prompt de LLM o al `TraceLogger` (RNF-01). La tarea 3.3 integra `PIIMasker` dentro
del pipeline completo de `InputGuardrail.process()`; este módulo únicamente resuelve
la responsabilidad de enmascaramiento, de forma aislada y testeable.

Enfoque deliberadamente basado en reglas/regex (heurístico), sin dependencias de
NLP/NER externas: adecuado para un prototipo en contexto financiero colombiano y
consistente con RNF-15 (alcance acotado al tiempo disponible).
"""
from __future__ import annotations

import re
from dataclasses import dataclass


class PIIType:
    """Tipos de PII reconocidos por `PIIMasker` (ver `SanitizedInput.pii_detectada`)."""

    CUENTA = "CUENTA"
    TELEFONO = "TELEFONO"
    CEDULA = "CEDULA"
    NOMBRE = "NOMBRE"


def _mask_token(pii_type: str) -> str:
    return f"[PII:{pii_type}]"


@dataclass(frozen=True)
class PIIMaskResult:
    """Resultado de `PIIMasker.mask`: texto enmascarado + tipos de PII detectados.

    `pii_detectada` contiene únicamente los tipos (p.ej. `"CEDULA"`), nunca los
    valores originales de PII (Requirement 2.1, `SanitizedInput.pii_detectada`).
    """

    texto_enmascarado: str
    pii_detectada: list[str]


class PIIMasker:
    """Detecta y enmascara PII (nombres, cédulas, teléfonos, cuentas) en texto crudo.

    El enmascaramiento se aplica en un único pase secuencial, ordenado de mayor a
    menor especificidad, de forma que un mismo fragmento de texto no sea
    clasificado dos veces bajo tipos distintos (p.ej. un número de cuenta no debe
    terminar también enmascarado como cédula):

    1. `CUENTA`: números de cuenta, identificados por contexto léxico explícito
       ("cuenta", "cta", "número de cuenta") seguido de una secuencia de dígitos.
    2. `TELEFONO`: números de celular colombianos (10 dígitos, inician en 3) o
       fijos con indicativo, con o sin separadores.
    3. `CEDULA`: secuencias numéricas de 6 a 10 dígitos (con o sin puntos como
       separador de miles), una vez descartados cuentas y teléfonos.
    4. `NOMBRE`: nombres propios, identificados por frases gatillo
       ("me llamo", "mi nombre es", "soy", "habla/hablé con") o, de forma más
       general, secuencias de 2 a 4 palabras consecutivas con mayúscula inicial
       (heurística de nombre propio).

    Al tratarse de un enmascaramiento por reglas (no un modelo de NER), es
    deliberadamente conservador: prioriza no filtrar PII (aceptando falsos
    positivos ocasionales) sobre no enmascarar de más (RNF-01).
    """

    # --- CUENTA: contexto léxico + dígitos ---------------------------------
    _CUENTA_RE = re.compile(
        r"""(?ix)
        \b(?:n(?:[uú]mero)?\.?\s*de\s*cuenta|cuenta|cta\.?)\b
        (?:\s+(?:de\s+ahorros|corriente))?
        (?:\s+(?:es|no\.?|n[uú]mero|\#))?
        \s*[:\-]?\s*
        (\d[\d.\-\s]{5,20}\d)
        """,
    )

    # --- TELEFONO: celular colombiano (3XX...) o fijo con indicativo -------
    # El indicativo de fijo (1-8, opcionalmente precedido de 0) debe venir
    # explícitamente delimitado por paréntesis o separador para no confundirse
    # con una cédula de 8 dígitos sin ningún marcador de formato telefónico.
    _TELEFONO_RE = re.compile(
        r"""(?x)
        (?<!\d)
        (?:\+?57[\s.\-]?)?
        (?:
            3\d{2}[\s.\-]?\d{3}[\s.\-]?\d{4}
            |
            \(0?[1-8]\)[\s.\-]?\d{3}[\s.\-]?\d{4}
            |
            0[1-8][\s.\-]\d{3}[\s.\-]?\d{4}
        )
        (?!\d)
        """,
    )

    # --- CEDULA: 6-10 dígitos, con o sin puntos de miles --------------------
    _CEDULA_RE = re.compile(
        r"""(?x)
        (?<!\d)
        \d{1,3}(?:[.,]\d{3}){1,3}
        (?!\d)
        |
        (?<!\d)
        \d{6,10}
        (?!\d)
        """,
    )

    # --- NOMBRE: frase gatillo + nombre propio ------------------------------
    _NOMBRE_TRIGGER_RE = re.compile(
        r"""(?ix)
        \b(?:me\s+llamo|mi\s+nombre\s+es|soy|habla(?:r)?\s+con|habl[ée]\s+con)\s+
        ([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){0,3})
        """,
    )

    # --- NOMBRE: heurística genérica (2-4 palabras con mayúscula inicial) ---
    _NOMBRE_GENERIC_RE = re.compile(
        r"\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){1,3}\b"
    )

    def mask(self, texto: str) -> PIIMaskResult:
        """Enmascara PII en `texto` y retorna el texto enmascarado + tipos detectados.

        No modifica `texto` (los `str` son inmutables); retorna siempre un nuevo
        `PIIMaskResult`. Si `texto` es vacío o `None`-like, se retorna sin cambios
        y sin PII detectada.
        """
        if not texto:
            return PIIMaskResult(texto_enmascarado=texto, pii_detectada=[])

        detected: list[str] = []
        resultado = texto

        resultado, found = self._apply(self._CUENTA_RE, resultado, PIIType.CUENTA)
        if found:
            detected.append(PIIType.CUENTA)

        resultado, found = self._apply(self._TELEFONO_RE, resultado, PIIType.TELEFONO)
        if found:
            detected.append(PIIType.TELEFONO)

        resultado, found = self._apply(self._CEDULA_RE, resultado, PIIType.CEDULA)
        if found:
            detected.append(PIIType.CEDULA)

        resultado, found_trigger = self._apply(
            self._NOMBRE_TRIGGER_RE, resultado, PIIType.NOMBRE
        )
        resultado, found_generic = self._apply(
            self._NOMBRE_GENERIC_RE, resultado, PIIType.NOMBRE
        )
        if found_trigger or found_generic:
            detected.append(PIIType.NOMBRE)

        return PIIMaskResult(texto_enmascarado=resultado, pii_detectada=detected)

    @staticmethod
    def _apply(pattern: re.Pattern[str], texto: str, pii_type: str) -> tuple[str, bool]:
        """Sustituye cada match de `pattern` por el token de `pii_type`.

        Si el patrón define un grupo de captura (p.ej. `CUENTA`, que preserva la
        palabra "cuenta" y enmascara solo el número), se sustituye únicamente ese
        grupo dentro del match; si no hay grupo, se sustituye el match completo.
        """
        token = _mask_token(pii_type)

        def repl(match: re.Match[str]) -> str:
            if match.groups() and match.group(1) is not None:
                return match.group(0).replace(match.group(1), token)
            return token

        nuevo_texto, n = pattern.subn(repl, texto)
        return nuevo_texto, n > 0
