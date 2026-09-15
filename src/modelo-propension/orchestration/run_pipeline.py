"""Orquestacion end-to-end + gate de CI/CD simulado (RF-10).

Encadena: ingesta -> features -> split -> entrenamiento -> evaluacion OOT ->
registro -> gate de promocion. Reemplaza a un pipeline real de CI/CD
(ej. GitHub Actions) con un script Python ejecutable localmente.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from config_loader import load_config
from pipeline import features, ingest, registry, split, train


def run_full_pipeline(config_path: str | Path, month: str) -> dict[str, Any]:
    """Ejecuta el pipeline completo de entrenamiento y aplica el gate de promocion.

    Pasos (RF-05, RF-06, RF-09, RF-10):
        1. Cargar y validar datos crudos.
        2. Construir features (sin fuga de informacion).
        3. Particionar temporalmente (train/val/OOT).
        4. Tunear y entrenar el modelo challenger.
        5. Evaluar el challenger en OOT.
        6. Registrar la version challenger.
        7. Aplicar el gate de promocion (pipeline.registry.promote).

    Returns:
        Diccionario {"version", "promovido", "metricas_challenger",
        "metrica_campeon_anterior"} -- incluye la metrica del campeon
        anterior (capturada ANTES de promover, o None si no habia campeon)
        para poder comparar viejo vs. nuevo.
    """
    config = load_config(config_path)

    datasets = ingest.load_raw_data(config)
    ingest.validate_schema(datasets)
    datasets = ingest.clean_raw_data(datasets)

    features_df = features.build_features(datasets, reference_month=month, config=config, population="trtest")

    train_df, val_df, oot_df = split.temporal_split(
        features_df,
        date_column="fecha_corte",
        train_end_month=config["split"]["train_end_month"],
        val_end_month=config["split"]["val_end_month"],
    )

    feature_columns = features.get_feature_columns(features_df)

    best_params = train.tune_hyperparameters(
        train_df,
        val_df,
        feature_columns,
        config["data"]["label_column"],
        config["model"]["hyperparameters"],
        config["model"]["random_seed"],
        n_iter=config["model"].get("n_iter"),
    )
    model = train.train_model(
        train_df, feature_columns, config["data"]["label_column"], best_params, config["model"]["random_seed"]
    )
    threshold = train.select_threshold(model, val_df, feature_columns, config["data"]["label_column"])
    metrics = train.evaluate_oot(model, oot_df, feature_columns, config["data"]["label_column"], threshold)

    challenger_version = registry.register_version(
        model,
        metadata={"metricas": metrics, "hiperparametros": best_params},
        registry_path=config["registry"]["path"],
    )

    promotion_metric = config["registry"]["promotion_metric"]
    try:
        campeon_anterior = registry.get_champion(config["registry"]["path"])
        metrica_campeon_anterior = campeon_anterior.get("metricas", {}).get(promotion_metric)
    except registry.NoChampionRegisteredError:
        metrica_campeon_anterior = None

    promoted = registry.promote(
        challenger_version,
        registry_path=config["registry"]["path"],
        promotion_metric=promotion_metric,
        min_improvement=config["registry"]["min_improvement"],
    )

    return {
        "version": challenger_version,
        "promovido": promoted,
        "metricas_challenger": metrics,
        "metrica_campeon_anterior": metrica_campeon_anterior,
        "promotion_metric": promotion_metric,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Orquestador del pipeline de propension (simulacion de CI/CD).")
    parser.add_argument("--config", default=None, help="Ruta al archivo pipeline_config.yaml")
    parser.add_argument("--month", required=True, help="Mes de referencia (t) para entrenar, formato YYYY-MM")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    config_path = args.config or (Path(__file__).parent.parent / "config" / "pipeline_config.yaml")
    resultado = run_full_pipeline(config_path, args.month)

    metric_name = resultado["promotion_metric"]
    nuevo = resultado["metricas_challenger"][metric_name]
    anterior = resultado["metrica_campeon_anterior"]

    print(f"Version challenger: {resultado['version']}")
    print(f"{metric_name.upper()} campeon anterior: {anterior:.4f}" if anterior is not None else f"{metric_name.upper()} campeon anterior: no habia campeon (bootstrap)")
    print(f"{metric_name.upper()} challenger (nuevo): {nuevo:.4f}")
    print(f"Promovido: {resultado['promovido']}")
