"""Servicio de Inferencia / Model Serving expuesto via FastAPI (RF-09).

Este servicio es el punto de integracion que usara el Context Builder del
sistema agentico (GET /score/{id_obligacion}) y la priorizacion por lotes
(RF-12). Los contratos ya estan completos; la logica de negocio que
depende de datos/modelo reales responde 501 hasta que pipeline/ este
implementado con los datos entregados por Bancolombia.
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException

from api.dependencies import get_champion_metadata, get_config
from pipeline import infer_batch, registry
from schemas.contracts import ModelMetadata, ScoreOutput

app = FastAPI(
    title="Servicio de Inferencia - Modelo de Propension",
    description="Expone el score de propension a aceptacion de opciones de pago (RF-07, RF-09, RF-12).",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Chequeo de disponibilidad del servicio (RNF-08)."""
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/model/info", response_model=ModelMetadata)
def model_info() -> ModelMetadata:
    """Retorna los metadatos del modelo campeon actual (RF-08, RNF-04)."""
    metadata = get_champion_metadata()
    if metadata is None:
        return ModelMetadata(version="sin-modelo-registrado", metricas={}, hiperparametros={})
    return ModelMetadata(**metadata, es_campeon=True)


@app.get("/score/{id_obligacion}", response_model=ScoreOutput)
def get_score(id_obligacion: str) -> ScoreOutput:
    """Retorna el score batch mas reciente de una obligacion (RF-07, RF-12).

    Es el endpoint que consumira el Context Builder del sistema agentico.
    """
    config = get_config()
    try:
        result = infer_batch.get_latest_score(id_obligacion, scores_path=config["data"]["scores_path"])
        return ScoreOutput(**result)
    except (FileNotFoundError, KeyError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/score/batch", response_model=list[ScoreOutput])
def run_score_batch(month: str) -> list[ScoreOutput]:
    """Dispara la inferencia batch para `month` (RF-07)."""
    config = get_config()
    metadata = get_champion_metadata()
    if metadata is None:
        raise HTTPException(status_code=501, detail="No hay modelo campeon registrado todavia.")
    model = registry.load_model(metadata["version"], config["registry"]["path"])
    result_df = infer_batch.run_batch_inference(
        month=month,
        model=model,
        config=config,
        model_version=metadata["version"],
    )
    return [ScoreOutput(**row) for row in result_df.to_dict(orient="records")]


@app.post("/model/promote")
def promote_model(challenger_version: str) -> dict[str, object]:
    """Aplica el gate de promocion sobre una version ya registrada (RF-10)."""
    config = get_config()
    try:
        promoted = registry.promote(
            challenger_version,
            registry_path=config["registry"]["path"],
            promotion_metric=config["registry"]["promotion_metric"],
            min_improvement=config["registry"]["min_improvement"],
        )
        return {"version": challenger_version, "promovido": promoted}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
