"""Model Registry y gobernanza simulados en filesystem (RF-09, RF-10, RNF-06).

Sustituye a un registry real (ej. MLflow Model Registry): versiona modelos
en disco, mantiene un puntero al modelo "campeon" y guarda historial para
permitir rollback. La logica de lectura/escritura si esta implementada; lo
que no existe aun es un modelo real que registrar (requiere pipeline.train).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Union

import joblib


class NoChampionRegisteredError(Exception):
    """No hay ningun modelo campeon registrado todavia."""


def _version_dir(registry_path: Union[str, Path], version: str) -> Path:
    return Path(registry_path) / version


def _champion_pointer_path(registry_path: Union[str, Path]) -> Path:
    return Path(registry_path) / "champion.json"


def _history_path(registry_path: Union[str, Path]) -> Path:
    return Path(registry_path) / "history.json"


def load_model(version: str, registry_path: Union[str, Path]) -> Any:
    """Carga el artefacto de modelo (joblib) de una version ya registrada.

    `get_champion` retorna solo los metadatos (metricas, hiperparametros);
    esta funcion trae el objeto de modelo real (con sus atributos
    category_maps_/feature_columns_) para poder llamar predict_proba.

    Raises:
        FileNotFoundError: si la version no tiene artefacto en el registry.
    """
    model_path = _version_dir(registry_path, version) / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"No existe el artefacto de modelo para la version '{version}'.")
    return joblib.load(model_path)


def register_version(
    model: Any,
    metadata: dict[str, Any],
    registry_path: Union[str, Path],
    version: str | None = None,
) -> str:
    """Serializa `model` y sus `metadata` como una nueva version en el registry.

    No decide si la version pasa a ser campeon (ver `promote`); solo la deja
    disponible para comparacion.

    Args:
        model: objeto entrenado (o cualquier artefacto serializable con joblib).
        metadata: metricas, hiperparametros, data_hash, etc.
        registry_path: carpeta raiz del registry (ver config.registry.path).
        version: identificador de version; si no se da, se genera con timestamp.

    Returns:
        El identificador de version usado.
    """
    version = version or datetime.now(timezone.utc).strftime("v%Y%m%d%H%M%S")
    version_dir = _version_dir(registry_path, version)
    version_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, version_dir / "model.joblib")
    metadata_to_store = {**metadata, "version": version}
    with (version_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata_to_store, f, indent=2, default=str)

    return version


def get_champion(registry_path: Union[str, Path]) -> dict[str, Any]:
    """Retorna los metadatos de la version campeon actual.

    Raises:
        NoChampionRegisteredError: si aun no se ha promovido ninguna version
            (esperado en un registry recien creado).
    """
    pointer_path = _champion_pointer_path(registry_path)
    if not pointer_path.exists():
        raise NoChampionRegisteredError("Aun no se ha promovido ningun modelo a campeon.")

    with pointer_path.open("r", encoding="utf-8") as f:
        pointer = json.load(f)

    metadata_path = _version_dir(registry_path, pointer["version"]) / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def promote(
    challenger_version: str,
    registry_path: Union[str, Path],
    promotion_metric: str,
    min_improvement: float = 0.0,
) -> bool:
    """Compara `challenger_version` contra el campeon actual y decide si promoverlo.

    Implementa el gate de CI/CD (RF-10): solo promueve si el challenger
    supera al campeon en `promotion_metric` por al menos `min_improvement`,
    o si todavia no existe campeon. Guarda el campeon anterior en el
    historial para permitir `rollback`.

    Returns:
        True si el challenger fue promovido, False si se mantuvo el campeon.
    """
    challenger_metadata_path = _version_dir(registry_path, challenger_version) / "metadata.json"
    if not challenger_metadata_path.exists():
        raise FileNotFoundError(
            f"La version '{challenger_version}' no esta registrada (usa register_version primero)."
        )
    with challenger_metadata_path.open("r", encoding="utf-8") as f:
        challenger_metadata = json.load(f)

    try:
        champion_metadata: dict[str, Any] | None = get_champion(registry_path)
        champion_score = champion_metadata.get("metricas", {}).get(promotion_metric, float("-inf"))
    except NoChampionRegisteredError:
        champion_metadata = None
        champion_score = float("-inf")

    challenger_score = challenger_metadata.get("metricas", {}).get(promotion_metric, float("-inf"))

    if challenger_score < champion_score + min_improvement:
        return False

    history_path = _history_path(registry_path)
    history = []
    if history_path.exists():
        with history_path.open("r", encoding="utf-8") as f:
            history = json.load(f)
    if champion_metadata is not None:
        history.append(
            {"version": champion_metadata["version"], "promoted_out_at": datetime.now(timezone.utc).isoformat()}
        )
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with _champion_pointer_path(registry_path).open("w", encoding="utf-8") as f:
        json.dump(
            {"version": challenger_version, "promoted_at": datetime.now(timezone.utc).isoformat()}, f, indent=2
        )

    return True


def rollback(registry_path: Union[str, Path]) -> str:
    """Revierte el campeon actual a la version previa registrada en el historial (RF-10).

    Returns:
        La version restaurada como campeon.

    Raises:
        NoChampionRegisteredError: si no hay historial para revertir.
    """
    history_path = _history_path(registry_path)
    if not history_path.exists():
        raise NoChampionRegisteredError("No hay historial de promociones para revertir.")

    with history_path.open("r", encoding="utf-8") as f:
        history = json.load(f)
    if not history:
        raise NoChampionRegisteredError("El historial de promociones esta vacio.")

    previous = history.pop()
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with _champion_pointer_path(registry_path).open("w", encoding="utf-8") as f:
        json.dump(
            {
                "version": previous["version"],
                "promoted_at": datetime.now(timezone.utc).isoformat(),
                "rollback": True,
            },
            f,
            indent=2,
        )

    return previous["version"]
