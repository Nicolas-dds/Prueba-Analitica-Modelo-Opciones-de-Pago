"""Motor de Elegibilidad (`EligibilityEngine`) y sus reglas configurables.

Ver Component 4 de `design.md`.
"""
from sistema_agentico.eligibility.engine import (
    EligibilityEngine,
    MAX_OPCIONES_ELEGIBLES,
)
from sistema_agentico.eligibility.reglas import (
    ConfiguracionReglasElegibilidadError,
    ReglasElegibilidad,
    TIPOS_OPCION_REQUERIDOS,
)

__all__ = [
    "ConfiguracionReglasElegibilidadError",
    "EligibilityEngine",
    "MAX_OPCIONES_ELEGIBLES",
    "ReglasElegibilidad",
    "TIPOS_OPCION_REQUERIDOS",
]
