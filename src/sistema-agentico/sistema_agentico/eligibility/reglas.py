"""Reglas configurables para el motor determinista de elegibilidad."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

TIPOS_OPCION_REQUERIDOS = frozenset({
    "ampliacion_plazo",
    "reduccion_cuota",
    "renegociacion_tasa",
    "reestructuracion",
})


class ConfiguracionReglasElegibilidadError(ValueError):
    """Indica una configuración YAML de elegibilidad inválida."""


@dataclass(frozen=True)
class ReglasElegibilidad:
    """Parámetros versionados que consume ``EligibilityEngine``.

    La configuración se carga desde YAML para permitir ajustar reglas sin cambios de código.
    """

    version: int
    max_opciones_pago_mes: int
    cooldown_meses_por_tipo: Mapping[str, int]
    max_dias_acuerdo: int

    def __post_init__(self) -> None:
        _validar_entero_positivo("version", self.version)
        _validar_entero_positivo("max_opciones_pago_mes", self.max_opciones_pago_mes)
        _validar_entero_positivo("max_dias_acuerdo", self.max_dias_acuerdo)
        if self.max_dias_acuerdo > 5:
            raise ConfiguracionReglasElegibilidadError(
                "max_dias_acuerdo no puede exceder 5 días"
            )
        if not isinstance(self.cooldown_meses_por_tipo, Mapping):
            raise ConfiguracionReglasElegibilidadError(
                "cooldown_meses_por_tipo debe ser un mapeo por tipo de opción"
            )
        tipos_configurados = set(self.cooldown_meses_por_tipo)
        if any(not isinstance(tipo, str) for tipo in tipos_configurados):
            raise ConfiguracionReglasElegibilidadError(
                "los tipos en cooldown_meses_por_tipo deben ser texto"
            )
        faltantes = TIPOS_OPCION_REQUERIDOS - tipos_configurados
        desconocidos = tipos_configurados - TIPOS_OPCION_REQUERIDOS
        if faltantes or desconocidos:
            detalle = []
            if faltantes:
                detalle.append(f"faltan: {', '.join(sorted(faltantes))}")
            if desconocidos:
                detalle.append(f"desconocidos: {', '.join(sorted(desconocidos))}")
            raise ConfiguracionReglasElegibilidadError(
                "cooldown_meses_por_tipo debe definir exactamente los tipos requeridos ("
                + "; ".join(detalle)
                + ")"
            )
        cooldowns: dict[str, int] = {}
        for tipo, meses in self.cooldown_meses_por_tipo.items():
            _validar_entero_positivo(f"cooldown_meses_por_tipo.{tipo}", meses)
            if not 3 <= meses <= 4:
                raise ConfiguracionReglasElegibilidadError(
                    f"cooldown_meses_por_tipo.{tipo} debe estar entre 3 y 4 meses"
                )
            cooldowns[tipo] = meses
        object.__setattr__(self, "cooldown_meses_por_tipo", MappingProxyType(cooldowns))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ReglasElegibilidad":
        """Carga y valida reglas desde un archivo YAML versionado."""
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(
                f"No se encontró el archivo de reglas de elegibilidad: {config_path}"
            )
        try:
            with config_path.open(encoding="utf-8") as archivo:
                contenido = yaml.safe_load(archivo)
        except yaml.YAMLError as error:
            raise ConfiguracionReglasElegibilidadError(
                f"YAML de reglas de elegibilidad inválido: {error}"
            ) from error
        return cls.from_mapping(contenido)

    @classmethod
    def from_mapping(cls, contenido: Any) -> "ReglasElegibilidad":
        """Construye reglas validadas desde el contenido YAML ya decodificado."""
        if not isinstance(contenido, Mapping):
            raise ConfiguracionReglasElegibilidadError(
                "la configuración de reglas debe ser un mapeo YAML no vacío"
            )
        requeridas = {
            "version",
            "max_opciones_pago_mes",
            "cooldown_meses_por_tipo",
            "max_dias_acuerdo",
        }
        claves = set(contenido)
        faltantes = requeridas - claves
        desconocidas = claves - requeridas
        if faltantes or desconocidas:
            detalle = []
            if faltantes:
                detalle.append(f"faltan: {', '.join(sorted(faltantes))}")
            if desconocidas:
                detalle.append(f"desconocidas: {', '.join(sorted(desconocidas))}")
            raise ConfiguracionReglasElegibilidadError(
                "claves de configuración inválidas (" + "; ".join(detalle) + ")"
            )
        return cls(
            version=contenido["version"],
            max_opciones_pago_mes=contenido["max_opciones_pago_mes"],
            cooldown_meses_por_tipo=contenido["cooldown_meses_por_tipo"],
            max_dias_acuerdo=contenido["max_dias_acuerdo"],
        )


def _validar_entero_positivo(nombre: str, valor: Any) -> None:
    if isinstance(valor, bool) or not isinstance(valor, int) or valor <= 0:
        raise ConfiguracionReglasElegibilidadError(
            f"{nombre} debe ser un entero positivo"
        )


__all__ = [
    "ConfiguracionReglasElegibilidadError",
    "ReglasElegibilidad",
    "TIPOS_OPCION_REQUERIDOS",
]
