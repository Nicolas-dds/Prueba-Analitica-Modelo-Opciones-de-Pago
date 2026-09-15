"""Contratos de entrada/salida compartidos entre el pipeline y la API (RF-07, RF-08, RF-09)."""
from __future__ import annotations

from datetime import date
from typing import Any, Optional

from pydantic import BaseModel, Field


class ObligacionInput(BaseModel):
    """Registro minimo de una obligacion en mora requerido para calificarla (RF-01, RF-02)."""

    id_obligacion: str = Field(..., description="Identificador unico de la obligacion")
    id_cliente: str = Field(..., description="Identificador del cliente (la unidad de analisis es la obligacion, no el cliente)")
    dias_mora: int = Field(..., ge=0, description="Dias de mora a la fecha de corte")
    exposicion: float = Field(..., ge=0, description="Saldo/exposicion de la obligacion")
    segmento: Optional[str] = Field(None, description="Segmento del cliente")
    canal_gestion: Optional[str] = Field(None, description="Gestion directa o de aliados (RF-02)")
    fecha_corte: date = Field(..., description="Mes t: hasta esta fecha se construyen las variables (RF-03)")


class ScoreOutput(BaseModel):
    """Contrato de salida de la inferencia batch (RF-07)."""

    id_obligacion: str
    score: float = Field(..., ge=0, le=1, description="Probabilidad de aceptar una opcion de pago en t+1")
    decil: int = Field(..., ge=1, le=10, description="Decil de propension dentro del batch calificado")
    version_modelo: str = Field(..., description="Version del modelo registrado que genero el score")
    fecha_calificacion: date = Field(..., description="Fecha en que se ejecuto la inferencia batch")


class ContribucionFeature(BaseModel):
    """Contribucion (SHAP) de una variable a un score individual (RF-08)."""

    feature: str
    valor: Any = Field(None, description="Valor de la variable; puede ser numerico o categorico")
    contribucion: float


class ExplicabilidadLocal(BaseModel):
    """Explicacion local de un score, consumible por el gestor y el sistema agentico (RF-08, RNF-04)."""

    id_obligacion: str
    version_modelo: str
    top_features: list[ContribucionFeature]


class ModelMetadata(BaseModel):
    """Metadatos de una version de modelo en el registry simulado (RF-09, RNF-06)."""

    version: str
    fecha_entrenamiento: Optional[date] = None
    metricas: dict[str, float] = Field(default_factory=dict)
    data_hash: Optional[str] = Field(None, description="Hash del dataset de entrenamiento, para trazabilidad (RNF-06)")
    hiperparametros: dict = Field(default_factory=dict)
    es_campeon: bool = False
