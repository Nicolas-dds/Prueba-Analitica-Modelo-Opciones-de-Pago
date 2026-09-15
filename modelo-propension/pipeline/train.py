"""Entrenamiento, tuning y evaluacion OOT del modelo de propension (RF-06).

Algoritmo elegido: XGBoost (clasificacion binaria), con soporte nativo de
variables categoricas (enable_categorical=True) y de nulos: no se hace
imputacion manual, XGBoost aprende la mejor direccion del split para los
valores faltantes (relevante dado el % de nulos en columnas demograficas,
ver memoria del repo).
"""
from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
from xgboost import XGBClassifier


def prepare_features_for_model(
    df: pd.DataFrame,
    feature_columns: list[str],
    category_maps: dict[str, list] | None = None,
) -> tuple[pd.DataFrame, dict[str, list]]:
    """Selecciona `feature_columns` y castea las columnas de texto a categoria (requerido por enable_categorical de XGBoost).

    Asume que `df` ya paso por pipeline.ingest.clean_raw_data (sin inf/valores
    invalidos) -- esta funcion solo se encarga de la codificacion especifica
    para XGBoost, no de limpieza general de datos.

    Si se pasa `category_maps` (tipicamente el del set de entrenamiento), se
    reusan esas mismas categorias: una categoria nueva en val/OOT/inferencia
    que el modelo nunca vio en entrenamiento se vuelve NaN (manejado
    nativamente como faltante), evitando inconsistencias entre fit y predict.

    Returns:
        Tupla (X, category_maps); `category_maps` se reutiliza en
        evaluate_oot y en inferencia futura para codificar igual.
    """
    X = df[feature_columns].copy()
    categorical_cols = [c for c in feature_columns if X[c].dtype == object]
    new_category_maps: dict[str, list] = {}
    for col in categorical_cols:
        categories = category_maps.get(col) if category_maps else None
        X[col] = pd.Categorical(X[col], categories=categories) if categories else X[col].astype("category")
        new_category_maps[col] = list(X[col].cat.categories)
    return X, new_category_maps


def _best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Busca, sobre la curva precision-recall, el umbral que maximiza F1."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = int(np.argmax(f1_scores[:-1]))
    return float(f1_scores[best_idx]), float(thresholds[best_idx])


def _compute_ks(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Estadistico Kolmogorov-Smirnov entre la distribucion de scores de positivos y negativos."""
    return float(ks_2samp(y_score[y_true == 1], y_score[y_true == 0]).statistic)


def tune_hyperparameters(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    param_grid: dict[str, list[Any]],
    random_seed: int = 42,
    n_iter: int | None = None,
) -> dict[str, Any]:
    """Busca los mejores hiperparametros de XGBoost sobre `param_grid`.

    Entrena un modelo por cada combinacion y selecciona la que maximiza AUC
    en `val_df` (nunca el OOT, RF-06). AUC se usa por ser independiente de
    umbral; F1 -- la metrica de promocion -- se calcula despues sobre el OOT
    en evaluate_oot, con el umbral elegido en select_threshold.

    Si `n_iter` se especifica y es menor que el total de combinaciones, se
    muestrean `n_iter` al azar (random search) en vez de probarlas todas
    (grid search exhaustivo) -- mas eficiente cuando el espacio de
    hiperparametros crece (Bergstra & Bengio, 2012), permite agregar mas
    hiperparametros (ej. regularizacion) sin que el tiempo de busqueda
    explote combinatoriamente.

    Returns:
        Diccionario con los mejores hiperparametros encontrados.
    """
    X_train, category_maps = prepare_features_for_model(train_df, feature_columns)
    y_train = train_df[label_column]
    X_val, _ = prepare_features_for_model(val_df, feature_columns, category_maps=category_maps)
    y_val = val_df[label_column]

    keys = list(param_grid.keys())
    all_combos = list(itertools.product(*param_grid.values()))
    if n_iter is not None and n_iter < len(all_combos):
        rng = np.random.default_rng(random_seed)
        combos = [all_combos[i] for i in rng.choice(len(all_combos), size=n_iter, replace=False)]
    else:
        combos = all_combos

    best_score = -np.inf
    best_params: dict[str, Any] = {}
    for combo in combos:
        params = dict(zip(keys, combo))
        model = XGBClassifier(
            **params,
            random_state=random_seed,
            tree_method="hist",
            enable_categorical=True,
            eval_metric="logloss",
        )
        model.fit(X_train, y_train)
        score = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        if score > best_score:
            best_score = score
            best_params = params

    return best_params


def train_model(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    hyperparameters: dict[str, Any],
    random_seed: int = 42,
) -> Any:
    """Entrena el modelo XGBoost final con los hiperparametros seleccionados.

    Guarda las categorias usadas en entrenamiento como atributos
    `category_maps_`/`feature_columns_` del propio modelo (viajan con el al
    serializar via pipeline.registry), para que evaluate_oot e inferencia
    futura codifiquen exactamente igual.

    Returns:
        Modelo XGBoost entrenado (objeto serializable via pipeline.registry).
    """
    X_train, category_maps = prepare_features_for_model(train_df, feature_columns)
    y_train = train_df[label_column]

    model = XGBClassifier(
        **hyperparameters,
        random_state=random_seed,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    model.category_maps_ = category_maps
    model.feature_columns_ = feature_columns
    return model


def select_threshold(
    model: Any,
    val_df: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
) -> float:
    """Selecciona el umbral de clasificacion que maximiza F1 en `val_df` (RF-06).

    El umbral NUNCA debe elegirse sobre el propio OOT (ver evaluate_oot):
    hacerlo seria optimizar la metrica en el mismo conjunto donde se reporta,
    un sesgo optimista. Se elige aqui sobre validacion y se aplica FIJO en
    evaluate_oot.

    Returns:
        El umbral (float en [0,1]) que maximiza F1 en `val_df`.
    """
    category_maps = getattr(model, "category_maps_", None)
    X_val, _ = prepare_features_for_model(val_df, feature_columns, category_maps=category_maps)
    y_val = val_df[label_column].to_numpy()
    y_score = model.predict_proba(X_val)[:, 1]
    _, threshold = _best_f1_threshold(y_val, y_score)
    return threshold


def evaluate_oot(
    model: Any,
    oot_df: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    threshold: float,
) -> dict[str, float]:
    """Evalua el modelo sobre el conjunto out-of-time (RF-06).

    `threshold` debe venir de select_threshold(model, val_df, ...) y se
    aplica FIJO aqui -- nunca se re-optimiza sobre el propio OOT, para no
    sesgar de forma optimista el F1 reportado (la metrica de promocion).
    AUC y KS (threshold-free) se reportan como contexto adicional.

    Returns:
        Diccionario de metricas: {"f1", "f1_threshold", "auc", "ks"}.
    """
    category_maps = getattr(model, "category_maps_", None)
    X_oot, _ = prepare_features_for_model(oot_df, feature_columns, category_maps=category_maps)
    y_oot = oot_df[label_column].to_numpy()
    y_score = model.predict_proba(X_oot)[:, 1]
    y_pred = (y_score >= threshold).astype(int)

    return {
        "f1": float(f1_score(y_oot, y_pred)),
        "f1_threshold": threshold,
        "auc": float(roc_auc_score(y_oot, y_score)),
        "ks": _compute_ks(y_oot, y_score),
    }
