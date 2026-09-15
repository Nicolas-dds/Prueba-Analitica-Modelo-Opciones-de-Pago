"""Pruebas de pipeline.monitor con datos sinteticos pequenos (RNF-14)."""
import numpy as np
import pandas as pd

from pipeline.monitor import (
    compute_data_drift,
    compute_performance_drift,
    compute_prediction_drift,
    evaluate_alerts,
)


def test_compute_data_drift_es_cero_cuando_no_hay_cambio_de_distribucion() -> None:
    rng = np.random.default_rng(42)
    ref = pd.DataFrame({"num_feature": rng.normal(size=1000)})
    cur = ref.copy()  # misma distribucion exacta

    result = compute_data_drift(ref, cur, features=["num_feature"])

    assert result["num_feature"] < 0.01


def test_compute_data_drift_detecta_shift_fuerte_en_variable_numerica() -> None:
    rng = np.random.default_rng(42)
    ref = pd.DataFrame({"num_feature": rng.normal(loc=0, scale=1, size=1000)})
    cur = pd.DataFrame({"num_feature": rng.normal(loc=5, scale=1, size=1000)})  # shift grande

    result = compute_data_drift(ref, cur, features=["num_feature"])

    assert result["num_feature"] > 0.2  # supera el umbral tipico de drift significativo


def test_compute_data_drift_detecta_shift_en_variable_categorica() -> None:
    ref = pd.DataFrame({"cat_feature": ["A"] * 800 + ["B"] * 200})
    cur = pd.DataFrame({"cat_feature": ["A"] * 200 + ["B"] * 800})  # se invierten las proporciones

    result = compute_data_drift(ref, cur, features=["cat_feature"])

    assert result["cat_feature"] > 0.2


def test_compute_data_drift_retorna_cero_si_no_hay_datos_validos() -> None:
    ref = pd.DataFrame({"num_feature": [np.nan, np.nan]})
    cur = pd.DataFrame({"num_feature": [1.0, 2.0]})

    result = compute_data_drift(ref, cur, features=["num_feature"])

    assert result["num_feature"] == 0.0


def test_compute_prediction_drift_es_cero_para_misma_distribucion() -> None:
    rng = np.random.default_rng(1)
    scores = rng.uniform(size=500)

    ks = compute_prediction_drift(scores, scores.copy())

    assert ks == 0.0


def test_compute_prediction_drift_detecta_shift_de_scores() -> None:
    rng = np.random.default_rng(1)
    reference_scores = rng.uniform(0, 0.3, size=500)
    current_scores = rng.uniform(0.7, 1.0, size=500)

    ks = compute_prediction_drift(reference_scores, current_scores)

    assert ks > 0.9


def test_compute_performance_drift_calcula_diferencia_contra_lo_esperado() -> None:
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_score = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])  # perfectamente separable con threshold=0.5

    result = compute_performance_drift(y_true, y_score, expected_f1=0.9, threshold=0.5)

    assert result["f1_real"] == 1.0
    assert result["f1_esperado"] == 0.9
    assert abs(result["diferencia"] - 0.1) < 1e-9


def test_evaluate_alerts_sin_alertas_cuando_todo_esta_bajo_el_umbral() -> None:
    drift_results = {"feat_a": 0.05, "feat_b": 0.1}
    thresholds = {"psi_threshold": 0.2}

    alerts = evaluate_alerts(drift_results, thresholds)

    assert alerts == []


def test_evaluate_alerts_reporta_variables_que_superan_el_umbral() -> None:
    drift_results = {"feat_a": 0.05, "feat_b": 0.35}
    thresholds = {"psi_threshold": 0.2}

    alerts = evaluate_alerts(drift_results, thresholds)

    assert len(alerts) == 1
    assert "feat_b" in alerts[0]


def test_evaluate_alerts_vacio_si_no_hay_umbral_configurado() -> None:
    drift_results = {"feat_a": 0.9}

    alerts = evaluate_alerts(drift_results, thresholds={})

    assert alerts == []
