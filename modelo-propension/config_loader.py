"""Carga de la configuracion parametrizable del pipeline (RNF-10)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "pipeline_config.yaml"


def load_config(config_path: Union[str, Path] = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Lee el archivo YAML de configuracion y lo retorna como diccionario.

    Todas las reglas parametrizables (umbrales de drift, metrica de
    promocion, rutas, hiperparametros) viven aqui para que puedan cambiar
    sin tocar codigo (RNF-10).

    Args:
        config_path: ruta al archivo YAML de configuracion.

    Returns:
        Diccionario con las secciones data/split/model/registry/monitoring/api.

    Raises:
        FileNotFoundError: si `config_path` no existe.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontro el archivo de configuracion: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
