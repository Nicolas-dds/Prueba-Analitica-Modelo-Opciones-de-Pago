"""Dependencias compartidas de la API (carga de configuracion y del modelo campeon)."""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional

from config_loader import load_config
from pipeline.registry import NoChampionRegisteredError, get_champion


@lru_cache
def get_config() -> dict[str, Any]:
    """Carga (y cachea) la configuracion del pipeline para la vida del proceso."""
    return load_config()


def get_champion_metadata() -> Optional[dict[str, Any]]:
    """Retorna los metadatos del modelo campeon, o None si aun no hay ninguno registrado."""
    config = get_config()
    try:
        return get_champion(config["registry"]["path"])
    except NoChampionRegisteredError:
        return None
