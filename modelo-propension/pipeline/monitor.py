"""Monitoreo de drift y desempeno (RF-11), implementado sin Evidently."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import f1_score

_PSI_EPS = 1e-4


def _psi_from_counts(
    ref_counts: pd.Series, cur_counts: pd.Series, ref_total: int, cur_total: int
) -> float:
    """PSI a partir de conteos por bucket ya alineados entre referencia y actual."""
    ref_props = np.clip(ref_counts.to_numpy() / max(ref_total, 1), _PSI_EPS, None)
    cur_props = np.clip(cur_counts.to_numpy() / max(cur_total, 1), _PSI_EPS, None)
    return float(np.sum((cur_props - ref_props) * np.log(cur_props / ref_props)))


def _psi_numeric(ref: pd.Series, cur: pd.Series, buckets: int = 10) -> float:
    """PSI para una variable numerica: buckets por deciles de la distribucion de referencia.

    Los bordes exteriores se extienden a +-inf para que valores de `cur` por
    fuera del rango visto en `ref` (drift real, no un error) caigan en el
    bucket extremo en vez de quedar fuera de todos los bins.
    """
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, buckets + 1)))
    if len(edges) < 3:
        # Variable casi constante en referencia: no hay forma de discretizar
        # en mas de un bucket, no es posible medir drift de forma util.
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts = pd.cut(ref, bins=edges).value_counts(sort=False)
    cur_counts = pd.cut(cur, bins=edges).value_counts(sort=False)
    return _psi_from_counts(ref_counts, cur_counts, len(ref), len(cur))


def _psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    """PSI para una variable categorica: cada categoria observada (en `ref` o `cur`) es un bucket."""
    categories = sorted(set(ref.unique()) | set(cur.unique()), key=str)
    ref_counts = ref.value_counts().reindex(categories, fill_value=0)
    cur_counts = cur.value_counts().reindex(categories, fill_value=0)
    return _psi_from_counts(ref_counts, cur_counts, len(ref), len(cur))


def compute_data_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame, features: list[str]) -> dict[str, float]:
    """Calcula el Population Stability Index (PSI) por variable.

    PSI > ~0.2 suele considerarse drift significativo (umbral parametrizable
    en config.monitoring.psi_threshold). Variables numericas se discretizan
    en deciles de la referencia; variables categoricas usan sus propias
    categorias como buckets. Si una variable no tiene datos validos (todo
    nulo) en alguno de los dos periodos, se reporta PSI=0.0 (sin evidencia
    suficiente para medir drift, en vez de fallar).

    Returns:
        Diccionario {feature: psi}.
    """
    result: dict[str, float] = {}
    for feature in features:
        ref = reference_df[feature].dropna()
        cur = current_df[feature].dropna()
        if ref.empty or cur.empty:
            result[feature] = 0.0
            continue
        if pd.api.types.is_numeric_dtype(ref):
            result[feature] = _psi_numeric(ref, cur)
        else:
            result[feature] = _psi_categorical(ref.astype(str), cur.astype(str))
    return result


def compute_prediction_drift(reference_scores: np.ndarray, current_scores: np.ndarray) -> float:
    """Calcula el estadistico Kolmogorov-Smirnov entre la distribucion de scores de referencia y la actual.

    Returns:
        Estadistico KS (0 = sin drift, 1 = drift maximo).
    """
    return float(ks_2samp(reference_scores, current_scores).statistic)


def compute_performance_drift(
    y_true: np.ndarray,
    y_score: np.ndarray,
    expected_f1: float,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compara el desempeno real (cuando llega la etiqueta en t+1) contra el esperado (RF-11).

    Usa F1 para ser consistente con la metrica de promocion en
    pipeline.registry.promote (config.registry.promotion_metric). `y_score`
    son probabilidades; se binarizan con `threshold` (idealmente el mismo
    umbral que maximizo F1 en pipeline.train.evaluate_oot, no un 0.5 fijo).

    Returns:
        Diccionario con la metrica real, la esperada y la diferencia.
    """
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    f1_real = float(f1_score(y_true, y_pred))
    return {
        "f1_real": f1_real,
        "f1_esperado": float(expected_f1),
        "diferencia": f1_real - float(expected_f1),
    }


def evaluate_alerts(drift_results: dict[str, float], thresholds: dict[str, float]) -> list[str]:
    """Convierte resultados de drift en alertas accionables segun umbrales parametrizables (RF-11, RNF-11).

    Hoy evalua PSI por variable (salida de `compute_data_drift`) contra
    `thresholds["psi_threshold"]` (config.monitoring.psi_threshold, ver
    pipeline_config.yaml) -- el disparador de reentrenamiento usado por
    `orchestration.run_monitoring`. El resto de umbrales de
    `config.monitoring` (ks_threshold, performance_drop_threshold) quedan
    definidos para cuando `compute_prediction_drift`/`compute_performance_drift`
    se integren a un job con historico de scores y etiquetas reales.

    Returns:
        Lista de mensajes de alerta (vacia si no se supero ningun umbral).
    """
    alerts: list[str] = []
    psi_threshold = thresholds.get("psi_threshold")
    if psi_threshold is None:
        return alerts

    for feature, psi in drift_results.items():
        if psi > psi_threshold:
            alerts.append(
                f"Drift de datos en '{feature}': PSI={psi:.4f} supera el umbral {psi_threshold} (RF-11)."
            )
    return alerts
