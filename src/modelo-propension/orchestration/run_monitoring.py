"""Orquestacion del monitoreo mensual + disparo de reentrenamiento por alerta (RF-11).

Cierra el hueco senalado en README/plan: hoy `pipeline/monitor.py` no tenia
ningun punto de entrada. Este script simula el job batch que en produccion
real correria en el orquestador (Airflow/Databricks Jobs), separado del
contenedor de serving (ver README, seccion "Que falta para produccion").
"""
from __future__ import annotations

import argparse
from pathlib import Path

from config_loader import load_config
from orchestration.run_pipeline import run_full_pipeline
from pipeline import features, ingest, monitor


def run_monitoring(config_path: str | Path, month: str) -> list[str]:
    """Evalua drift de datos del mes `month` contra el mes de referencia y, si hay alertas, reentrena.

    Pasos (RF-11):
        1. Cargar datos crudos y construir features del mes de referencia
           (config.data.reference_month) y del mes actual (`month`).
        2. Calcular PSI por variable (pipeline.monitor.compute_data_drift).
        3. Evaluar alertas contra los umbrales parametrizables
           (config.monitoring).
        4. Si hay al menos una alerta, disparar el pipeline de entrenamiento
           (orchestration.run_pipeline.run_full_pipeline) para ese mes.

    Nota: el drift de predicciones y el de desempeno real (RF-11) requieren
    que `pipeline.infer_batch` este implementado (para obtener el historico
    de scores y las etiquetas ya conocidas); quedan pendientes de agregar
    aqui una vez ese modulo tenga logica real.

    Returns:
        Lista de alertas evaluadas (vacia si no se supero ningun umbral). Si
        hubo alertas, incluye ademas un mensaje con el resultado del intento
        de reentrenamiento.
    """
    config = load_config(config_path)

    datasets = ingest.load_raw_data(config)
    ingest.validate_schema(datasets)
    datasets = ingest.clean_raw_data(datasets)

    reference_features = features.build_features(
        datasets, reference_month=config["data"]["reference_month"], config=config, population="trtest"
    )
    current_features = features.build_features(datasets, reference_month=month, config=config, population="trtest")

    feature_columns = features.get_feature_columns(current_features)

    data_drift = monitor.compute_data_drift(reference_features, current_features, feature_columns)
    alerts = monitor.evaluate_alerts(data_drift, config["monitoring"])

    if not alerts:
        return alerts

    resultado = run_full_pipeline(config_path, month)
    metric_name = resultado["promotion_metric"]
    nuevo = resultado["metricas_challenger"][metric_name]
    anterior = resultado["metrica_campeon_anterior"]
    if resultado["promovido"]:
        alerts.append(
            f"Reentrenamiento disparado por alerta: version '{resultado['version']}' promovida a campeon "
            f"({metric_name}: {anterior:.4f} -> {nuevo:.4f})."
            if anterior is not None
            else f"Reentrenamiento disparado por alerta: version '{resultado['version']}' promovida a campeon "
            f"({metric_name}: {nuevo:.4f}, no habia campeon previo)."
        )
    else:
        alerts.append(
            f"Reentrenamiento disparado por alerta: el challenger ({metric_name}: {nuevo:.4f}) "
            f"no supero al campeon vigente ({metric_name}: {anterior:.4f})."
        )
    return alerts


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Orquestador del monitoreo mensual (RF-11).")
    parser.add_argument("--config", default=None, help="Ruta al archivo pipeline_config.yaml")
    parser.add_argument("--month", required=True, help="Mes a monitorear, formato YYYY-MM")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    config_path = args.config or (Path(__file__).parent.parent / "config" / "pipeline_config.yaml")
    result = run_monitoring(config_path, args.month)
    print("\n".join(result) if result else "Sin alertas: no se disparo reentrenamiento.")
