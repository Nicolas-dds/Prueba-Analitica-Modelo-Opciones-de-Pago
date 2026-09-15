"""Servicio de Inferencia Batch (RF-07, RF-12).

Reutiliza pipeline.ingest + pipeline.features.build_features (mismo codigo
que en entrenamiento, evita training-serving skew) y
pipeline.train.prepare_features_for_model (misma codificacion categorica que
uso el modelo). Persiste el resultado en `config["data"]["scores_path"]`
(un archivo por mes) para que `get_latest_score` y la priorizacion por
lotes (RF-12) lo consuman sin tener que recalcular nada.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from pipeline import features, ingest
from pipeline.train import prepare_features_for_model


def _score_file(scores_path: str, month: str) -> Path:
    return Path(scores_path) / f"{month}.parquet"


def run_batch_inference(
    month: str,
    model: Any,
    config: dict,
    model_version: str,
    population: str = "oot",
) -> pd.DataFrame:
    """Califica todas las obligaciones de `month` y arma el contrato de salida (RF-07).

    `model` debe venir de pipeline.registry.load_model (trae sus atributos
    category_maps_/feature_columns_, necesarios para codificar igual que en
    entrenamiento).

    Args:
        month: mes a calificar ("YYYY-MM").
        model: modelo ya entrenado (ver pipeline.train.train_model).
        config: configuracion parametrizable (ver config/pipeline_config.yaml).
        model_version: version del modelo usada (para el contrato de salida).
        population: "oot" (mes real de produccion, sin label) o "trtest"
            (para recalificar un mes historico, ej. pruebas/backtesting).

    Returns:
        DataFrame con columnas: id_obligacion, score, decil, version_modelo,
        fecha_calificacion (ver schemas.contracts.ScoreOutput).
    """
    datasets = ingest.load_raw_data(config)
    ingest.validate_schema(datasets)
    datasets = ingest.clean_raw_data(datasets)

    features_df = features.build_features(datasets, reference_month=month, config=config, population=population)

    category_maps = getattr(model, "category_maps_", None)
    feature_columns = getattr(model, "feature_columns_")
    X, _ = prepare_features_for_model(features_df, feature_columns, category_maps=category_maps)
    scores = model.predict_proba(X)[:, 1]

    # Decil 1 = mas prioritario (score mas alto), igual convencion que `lote`
    # en probabilidad_oblig_base_hist ("menor = mas prioritario"). Se rankea
    # antes de particionar en deciles para garantizar exactamente 10 grupos
    # aunque haya scores empatados.
    ranks = pd.Series(scores).rank(method="first")
    deciles = pd.qcut(ranks, 10, labels=list(range(10, 0, -1))).astype(int)

    result = pd.DataFrame(
        {
            "id_obligacion": features_df["id_obligacion"].to_numpy(),
            "score": scores,
            "decil": deciles.to_numpy(),
            "version_modelo": model_version,
            "fecha_calificacion": pd.Timestamp.now(tz="UTC").normalize(),
        }
    )

    scores_path = Path(config["data"]["scores_path"])
    scores_path.mkdir(parents=True, exist_ok=True)
    result.to_parquet(_score_file(config["data"]["scores_path"], month), index=False)

    return result


def get_latest_score(id_obligacion: str, scores_path: str) -> dict:
    """Lee el score mas reciente de una obligacion desde el output de la inferencia batch.

    Es la funcion que expone GET /score/{id_obligacion} para que el Context
    Builder del sistema agentico obtenga el score sin re-invocar el modelo
    en tiempo real (RF-07, RF-12). Busca el archivo de mes mas reciente en
    `scores_path` (nombrados "YYYY-MM.parquet", ordenan cronologicamente
    como string) y filtra la fila de `id_obligacion`.

    Raises:
        FileNotFoundError: si no hay ningun batch calificado todavia.
        KeyError: si la obligacion no aparece en el batch mas reciente.
    """
    archivos = sorted(Path(scores_path).glob("*.parquet"))
    if not archivos:
        raise FileNotFoundError(f"No hay ningun batch de inferencia calificado en '{scores_path}' todavia.")

    ultimo = pd.read_parquet(archivos[-1])
    fila = ultimo.loc[ultimo["id_obligacion"] == id_obligacion]
    if fila.empty:
        raise KeyError(f"La obligacion '{id_obligacion}' no aparece en el batch mas reciente ({archivos[-1].name}).")

    resultado = fila.iloc[0].to_dict()
    resultado["fecha_calificacion"] = pd.Timestamp(resultado["fecha_calificacion"]).date()
    return resultado
