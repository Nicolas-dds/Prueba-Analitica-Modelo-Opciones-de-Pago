"""Pruebas de pipeline.train con datos sinteticos pequenos (RNF-14)."""
import numpy as np
import pandas as pd

from pipeline.train import evaluate_oot, select_threshold, train_model, tune_hyperparameters

FEATURE_COLUMNS = ["num_feature", "cat_feature"]
LABEL_COLUMN = "var_rpta_alt"
PARAM_GRID = {"max_depth": [2], "learning_rate": [0.3], "n_estimators": [20]}


def _make_df(n: int, seed: int) -> pd.DataFrame:
    """Dataset sintetico separable: la etiqueta depende fuertemente de las features."""
    rng = np.random.default_rng(seed)
    num_feature = rng.normal(size=n)
    cat_feature = rng.choice(["A", "B"], size=n)
    signal = num_feature + np.where(cat_feature == "A", 1.0, -1.0)
    label = (signal > np.median(signal)).astype(int)
    return pd.DataFrame({"num_feature": num_feature, "cat_feature": cat_feature, LABEL_COLUMN: label})


def test_train_model_y_evaluate_oot_aprenden_senal_separable() -> None:
    train_df = _make_df(400, seed=1)
    val_df = _make_df(120, seed=8)
    oot_df = _make_df(150, seed=2)

    model = train_model(train_df, FEATURE_COLUMNS, LABEL_COLUMN, {"max_depth": 3, "n_estimators": 50}, random_seed=42)
    threshold = select_threshold(model, val_df, FEATURE_COLUMNS, LABEL_COLUMN)
    metrics = evaluate_oot(model, oot_df, FEATURE_COLUMNS, LABEL_COLUMN, threshold)

    assert metrics["auc"] > 0.8  # senal fuerte y separable, debe generalizar bien
    assert 0.0 <= metrics["f1"] <= 1.0
    assert 0.0 <= metrics["ks"] <= 1.0
    assert metrics["f1_threshold"] == threshold


def test_evaluate_oot_usa_el_umbral_fijo_sin_reoptimizar() -> None:
    """Caso critico de metodologia: el umbral se aplica FIJO, nunca se re-busca sobre el propio OOT."""
    train_df = _make_df(300, seed=1)
    oot_df = _make_df(150, seed=2)
    model = train_model(train_df, FEATURE_COLUMNS, LABEL_COLUMN, {"max_depth": 3, "n_estimators": 50}, random_seed=42)

    metrics = evaluate_oot(model, oot_df, FEATURE_COLUMNS, LABEL_COLUMN, threshold=0.5)

    assert metrics["f1_threshold"] == 0.5


def test_select_threshold_retorna_valor_valido() -> None:
    train_df = _make_df(300, seed=1)
    val_df = _make_df(100, seed=7)
    model = train_model(train_df, FEATURE_COLUMNS, LABEL_COLUMN, {"max_depth": 3, "n_estimators": 50}, random_seed=42)

    threshold = select_threshold(model, val_df, FEATURE_COLUMNS, LABEL_COLUMN)

    assert 0.0 <= threshold <= 1.0


def test_tune_hyperparameters_retorna_combinacion_del_grid() -> None:
    train_df = _make_df(300, seed=3)
    val_df = _make_df(100, seed=4)

    best_params = tune_hyperparameters(train_df, val_df, FEATURE_COLUMNS, LABEL_COLUMN, PARAM_GRID, random_seed=42)

    assert best_params == {"max_depth": 2, "learning_rate": 0.3, "n_estimators": 20}


def test_evaluate_oot_maneja_categoria_nueva_como_faltante() -> None:
    """Una categoria en OOT que el modelo nunca vio en entrenamiento no debe romper la evaluacion."""
    train_df = _make_df(200, seed=5)
    model = train_model(train_df, FEATURE_COLUMNS, LABEL_COLUMN, {"max_depth": 2, "n_estimators": 10}, random_seed=42)

    oot_df = _make_df(50, seed=6)
    oot_df.loc[0, "cat_feature"] = "C"  # categoria nunca vista en train

    metrics = evaluate_oot(model, oot_df, FEATURE_COLUMNS, LABEL_COLUMN, threshold=0.5)
    assert "f1" in metrics


def test_tune_hyperparameters_con_n_iter_muestrea_un_subconjunto() -> None:
    train_df = _make_df(300, seed=3)
    val_df = _make_df(100, seed=4)
    grid = {"max_depth": [2, 3, 4], "learning_rate": [0.1, 0.3], "n_estimators": [10, 20]}  # 12 combinaciones

    best_params = tune_hyperparameters(
        train_df, val_df, FEATURE_COLUMNS, LABEL_COLUMN, grid, random_seed=42, n_iter=3
    )

    assert best_params["max_depth"] in grid["max_depth"]
    assert best_params["learning_rate"] in grid["learning_rate"]
    assert best_params["n_estimators"] in grid["n_estimators"]


def test_tune_hyperparameters_con_n_iter_muestrea_un_subconjunto() -> None:
    train_df = _make_df(300, seed=3)
    val_df = _make_df(100, seed=4)
    grid = {"max_depth": [2, 3, 4], "learning_rate": [0.1, 0.3], "n_estimators": [10, 20]}  # 12 combinaciones

    best_params = tune_hyperparameters(
        train_df, val_df, FEATURE_COLUMNS, LABEL_COLUMN, grid, random_seed=42, n_iter=3
    )

    assert best_params["max_depth"] in grid["max_depth"]
    assert best_params["learning_rate"] in grid["learning_rate"]
    assert best_params["n_estimators"] in grid["n_estimators"]
