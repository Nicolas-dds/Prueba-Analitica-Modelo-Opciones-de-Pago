"""Pruebas de pipeline.explain con datos sinteticos pequenos (RNF-14)."""
import numpy as np
import pandas as pd

from pipeline.explain import global_importance, local_explanation
from pipeline.train import train_model

FEATURE_COLUMNS = ["num_feature", "cat_feature"]
LABEL_COLUMN = "var_rpta_alt"


def _make_df(n: int, seed: int) -> pd.DataFrame:
    """Igual que en test_train.py: num_feature domina la señal, cat_feature aporta poco."""
    rng = np.random.default_rng(seed)
    num_feature = rng.normal(size=n)
    cat_feature = rng.choice(["A", "B"], size=n)
    label = (num_feature > 0).astype(int)
    return pd.DataFrame({"num_feature": num_feature, "cat_feature": cat_feature, LABEL_COLUMN: label})


def _make_model():
    train_df = _make_df(300, seed=1)
    return train_model(train_df, FEATURE_COLUMNS, LABEL_COLUMN, {"max_depth": 3, "n_estimators": 30}, random_seed=42)


def test_global_importance_identifica_la_variable_dominante() -> None:
    model = _make_model()
    X = _make_df(200, seed=2)[FEATURE_COLUMNS]

    importancia = global_importance(model, X)

    assert set(importancia.keys()) == set(FEATURE_COLUMNS)
    assert next(iter(importancia)) == "num_feature"  # la mas importante, por construccion del dataset


def test_local_explanation_retorna_top_k_ordenado_por_contribucion() -> None:
    model = _make_model()
    row = _make_df(1, seed=3).iloc[0][FEATURE_COLUMNS]

    explicacion = local_explanation(model, row, top_k=2)

    assert len(explicacion) == 2
    contribuciones = [abs(item["contribucion"]) for item in explicacion]
    assert contribuciones == sorted(contribuciones, reverse=True)
    assert {"feature", "valor", "contribucion"} <= explicacion[0].keys()


def test_local_explanation_conserva_valor_categorico() -> None:
    model = _make_model()
    row = pd.Series({"num_feature": 0.5, "cat_feature": "A"})

    explicacion = local_explanation(model, row, top_k=2)
    cat_item = next(item for item in explicacion if item["feature"] == "cat_feature")
    assert cat_item["valor"] == "A"
