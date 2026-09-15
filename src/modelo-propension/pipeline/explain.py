"""Explicabilidad global y local del score (RF-08, RNF-04).

Usa SHAP (TreeExplainer) sobre el modelo XGBoost entrenado en pipeline.train.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap

from pipeline.train import prepare_features_for_model


def _shap_values(model: Any, X: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Prepara `X` igual que en entrenamiento (misma codificacion categorica) y calcula los valores SHAP con TreeExplainer."""
    X_prepared, _ = prepare_features_for_model(
        X, model.feature_columns_, category_maps=getattr(model, "category_maps_", None)
    )
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_prepared)
    return shap_values, X_prepared


def global_importance(model: Any, X: pd.DataFrame) -> dict[str, float]:
    """Calcula la importancia global de variables (promedio de |SHAP|) (RF-08).

    `X` debe traer, como minimo, las columnas de `model.feature_columns_` --
    tipicamente una muestra representativa (ej. el propio train_df o OOT).

    Returns:
        Diccionario {feature: importancia}, ordenado descendente, para
        reportar al gestor y al equipo de gobernanza de modelos.
    """
    shap_values, X_prepared = _shap_values(model, X)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importancia = dict(zip(X_prepared.columns, (float(v) for v in mean_abs_shap)))
    return dict(sorted(importancia.items(), key=lambda kv: kv[1], reverse=True))


def local_explanation(model: Any, row: pd.Series, top_k: int = 5) -> list[dict[str, Any]]:
    """Calcula las `top_k` variables que mas explican el score de una obligacion (RF-08).

    Returns:
        Lista de {"feature", "valor", "contribucion"} ordenada por
        |contribucion| descendente (ver schemas.contracts.ContribucionFeature),
        consumible por el sistema agentico y por el gestor humano.
    """
    row_df = row.to_frame().T
    shap_values, X_prepared = _shap_values(model, row_df)
    contribuciones = shap_values[0]
    valores = X_prepared.iloc[0]

    explicacion = [
        {"feature": feature, "valor": valores[feature], "contribucion": float(contribuciones[i])}
        for i, feature in enumerate(X_prepared.columns)
    ]
    explicacion.sort(key=lambda item: abs(item["contribucion"]), reverse=True)
    return explicacion[:top_k]
