"""Enmascaramiento de PII sobre el mensaje crudo (Component 1: `InputGuardrail`).

Ver `design.md`, Component 1 ("Guardrail de Entrada") y Requirement 2.1 de
`requirements.md`: al procesar una entrada que contiene PII (nombres, cédulas,
teléfonos, cuentas), el `InputGuardrail` debe enmascarar dicha PII antes de que el
texto sea entregado al `Orchestrator`, a cualquier prompt de LLM o al `TraceLogger`
(RNF-01).

Este módulo implementa únicamente la pieza de enmascaramiento (`mask_pii`), pensada
para ser compuesta dentro de `InputGuardrail.process` junto con la detección de riesgo
de inyección/jailbreak (tarea 3.2) y la integración completa del guardrail (tarea 3.3).
No depende de ningún otro componente del sistema.

Enfoque: enmascaramiento basado en heurísticas (palabras clave de contexto + patrones
de formato típicos en Colombia) en lugar de NER. Es intencionalmente conservador:
ante la duda, prefiere enmascarar de más (falso positivo) a dejar pasar PII real
(falso negativo), en línea con Property 3 de `design.md` ("el `mensaje_sanitizado`
SHALL NOT contener ninguno de los valores originales de PII detectados").
"""
from __future__ import annotations

import re
from typing import Pattern

# ---------------------------------------------------------------------------
# Tipos de PII soportados
# ---------------------------------------------------------------------------

PII_TIPO_CEDULA = "cedula"
PII_TIPO_TELEFONO = "telefono"
PII_TIPO_CUENTA = "cuenta"
PII_TIPO_NOMBRE = "nombre"

#: Todos los tipos de PII que `mask_pii` es capaz de detectar/enmascarar.
PII_TIPOS_SOPORTADOS = (
    PII_TIPO_CEDULA,
    PII_TIPO_TELEFONO,
    PII_TIPO_CUENTA,
    PII_TIPO_NOMBRE,
)

_MASK_TOKENS = {
    PII_TIPO_CEDULA: "[CEDULA]",
    PII_TIPO_TELEFONO: "[TELEFONO]",
    PII_TIPO_CUENTA: "[CUENTA]",
    PII_TIPO_NOMBRE: "[NOMBRE]",
}

# ---------------------------------------------------------------------------
# Patrones, en orden de prioridad
#
# Se aplican secuencialmente sobre el texto (cada patrón opera sobre el resultado
# del anterior). Los patrones con contexto explícito (palabra clave: "cuenta",
# "cédula", "teléfono", "me llamo", ...) van primero para desambiguar correctamente
# entre tipos de PII que comparten formato (secuencias de dígitos). Los patrones
# "bare" (sin palabra clave) van después, de más específico/largo a más genérico,
# para evitar que un patrón corto capture solo una parte de un número más largo.
#
# Cada patrón tiene como máximo un grupo de captura: el fragmento exacto a
# enmascarar. Si no hay grupo de captura, se enmascara el match completo.
# ---------------------------------------------------------------------------

_PATTERNS: tuple[tuple[str, str, int], ...] = (
    # --- Cuenta con palabra clave de contexto ---------------------------------
    (
        PII_TIPO_CUENTA,
        r"cuenta(?:\s+(?:bancaria|de\s+ahorros|corriente))?[^\d\n]{0,20}?(\d[\d\s\-]{6,19}\d)",
        re.IGNORECASE,
    ),
    # --- Cédula con palabra clave de contexto ---------------------------------
    (
        PII_TIPO_CEDULA,
        r"(?:c[eé]dula(?:\s+de\s+ciudadan[ií]a)?|c\.?\s?c\.?|documento\s+de\s+identidad"
        r"|n[uú]mero\s+de\s+identificaci[oó]n)[^\d\n]{0,15}?(\d[\d.\s\-]{4,12}\d)",
        re.IGNORECASE,
    ),
    # --- Teléfono con palabra clave de contexto -------------------------------
    (
        PII_TIPO_TELEFONO,
        r"(?:tel[eé]fono|celular|cel\.?|whats\s*app|whatsapp|m[oó]vil)[^\d\n]{0,10}?"
        r"((?:\+?57[\s\-]?)?\d[\d\s\-]{6,10}\d)",
        re.IGNORECASE,
    ),
    # --- Teléfono celular colombiano sin contexto (10 dígitos, empieza en 3) --
    (
        PII_TIPO_TELEFONO,
        r"(?<!\d)((?:\+?57[\s\-]?)?3\d{2}[\s\-]?\d{3}[\s\-]?\d{4})(?!\d)",
        0,
    ),
    # --- Cuenta bancaria sin contexto (secuencia larga de dígitos) -----------
    (
        PII_TIPO_CUENTA,
        r"(?<!\d)(\d{11,20})(?!\d)",
        0,
    ),
    # --- Cédula con separadores de miles (formato colombiano: 1.234.567.890) --
    (
        PII_TIPO_CEDULA,
        r"(?<!\d)(\d{1,3}(?:\.\d{3}){1,3})(?!\d)",
        0,
    ),
    # --- Cédula sin contexto (secuencia corta de dígitos) ---------------------
    (
        PII_TIPO_CEDULA,
        r"(?<!\d)(\d{6,10})(?!\d)",
        0,
    ),
    # --- Nombre introducido explícitamente ("me llamo Juan Pérez") ------------
    (
        PII_TIPO_NOMBRE,
        r"(?:me\s+llamo|mi\s+nombre\s+es)\s+"
        r"([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){0,3})",
        0,
    ),
    # --- Nombre precedido de título ("señor/señora/sr./sra. Juan Pérez") ------
    (
        PII_TIPO_NOMBRE,
        r"(?:se[nñ]or(?:a)?|sr\.?|sra\.?)\s+"
        r"([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){0,3})",
        0,
    ),
    # --- Nombre genérico: 2 o más palabras consecutivas en Title Case ---------
    # Heurística de "nombres marcados" (design.md, Property 3): captura nombres
    # propios completos mencionados en el texto sin depender de un modelo de NER.
    (
        PII_TIPO_NOMBRE,
        r"\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){1,3}\b",
        0,
    ),
)

_COMPILED_PATTERNS: tuple[tuple[str, Pattern[str]], ...] = tuple(
    (tipo, re.compile(pattern, flags)) for tipo, pattern, flags in _PATTERNS
)


def mask_pii(text: str | None) -> tuple[str | None, list[str]]:
    """Enmascara PII (nombres, cédulas, teléfonos, cuentas) en `text`.

    Ver Requirement 2.1: el enmascaramiento debe ocurrir sobre el mensaje crudo antes
    de que llegue al `Orchestrator`, a cualquier prompt de LLM o al `TraceLogger`.
    Esta función es pura (no muta `text`, no tiene efectos secundarios) para poder
    invocarse de forma segura en ese punto único de entrada.

    Args:
        text: mensaje crudo del cliente. `None` se retorna sin cambios (modo
            proactivo, donde `mensaje_sanitizado` es `None` según `SanitizedInput`).

    Returns:
        Tupla `(texto_enmascarado, tipos_pii_detectados)` donde `tipos_pii_detectados`
        es la lista *ordenada y sin duplicados* de los tipos de PII enmascarados
        (nunca los valores originales), compatible con `SanitizedInput.pii_detectada`.
    """
    if text is None:
        return None, []

    masked = text
    detected: set[str] = set()

    for tipo, compiled in _COMPILED_PATTERNS:
        masked = compiled.sub(_make_replacer(tipo, detected), masked)

    return masked, sorted(detected)


def _make_replacer(tipo: str, detected: set[str]):
    """Construye la función de reemplazo usada por `Pattern.sub` para `tipo`.

    Si el patrón tiene un grupo de captura (el valor exacto a enmascarar), solo ese
    fragmento se reemplaza, preservando el resto del match (p.ej. la palabra clave de
    contexto "cuenta" en "cuenta 12345678"). Si no hay grupo de captura, se enmascara
    el match completo (patrones "bare" donde todo el match es el valor de PII).
    """

    def _replace(match: re.Match[str]) -> str:
        detected.add(tipo)
        mask_token = _MASK_TOKENS[tipo]
        if match.lastindex:
            valor = match.group(1)
            return match.group(0).replace(valor, mask_token, 1)
        return mask_token

    return _replace
