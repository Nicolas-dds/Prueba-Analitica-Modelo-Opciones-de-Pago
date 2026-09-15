"""Pruebas de pipeline.infer_batch con datos sinteticos pequenos (RNF-14).

Se monkeypatchea ingest/features para aislar la logica propia de
infer_batch (scoring, deciles, persistencia) sin depender de archivos reales.
"""
import numpy as np
import pandas as pd
import pytest

from pipeline import infer_batch


class _FakeModel:
    """Modelo minimo: el score es proporcional a `num_feature` (sin XGBoost real)."""

    feature_columns_ = ["num_feature"]
    category_maps_: dict = {}

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        values = X["num_feature"].to_numpy()
        scores = (values - values.min()) / (values.max() - values.min())
        return np.column_stack([1 - scores, scores])


def _fake_features_df(n: int) -> pd.DataFrame:
    return pd.DataFrame({"id_obligacion": [f"id{i}" for i in range(n)], "num_feature": np.arange(n, dtype=float)})


def _patch_pipeline(monkeypatch, n: int = 20) -> None:
    monkeypatch.setattr(infer_batch.ingest, "load_raw_data", lambda config: {})
    monkeypatch.setattr(infer_batch.ingest, "validate_schema", lambda datasets: None)
    monkeypatch.setattr(infer_batch.ingest, "clean_raw_data", lambda datasets: datasets)
    monkeypatch.setattr(infer_batch.features, "build_features", lambda *a, **k: _fake_features_df(n))


def test_run_batch_inference_arma_contrato_de_salida_y_persiste(monkeypatch, tmp_path) -> None:
    _patch_pipeline(monkeypatch)
    config = {"data": {"scores_path": str(tmp_path)}}

    result = infer_batch.run_batch_inference(month="2024-01", model=_FakeModel(), config=config, model_version="v1")

    assert list(result.columns) == ["id_obligacion", "score", "decil", "version_modelo", "fecha_calificacion"]
    assert result["decil"].nunique() == 10
    assert result.loc[result["score"].idxmax(), "decil"] == 1  # score mas alto -> decil 1 (mas prioritario)
    assert (tmp_path / "2024-01.parquet").exists()


def test_get_latest_score_lee_el_archivo_mas_reciente(monkeypatch, tmp_path) -> None:
    _patch_pipeline(monkeypatch)
    config = {"data": {"scores_path": str(tmp_path)}}
    infer_batch.run_batch_inference(month="2023-12", model=_FakeModel(), config=config, model_version="v1")
    infer_batch.run_batch_inference(month="2024-01", model=_FakeModel(), config=config, model_version="v2")

    resultado = infer_batch.get_latest_score("id5", scores_path=str(tmp_path))

    assert resultado["version_modelo"] == "v2"  # el mas reciente (2024-01), no 2023-12


def test_get_latest_score_falla_si_no_hay_batch(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        infer_batch.get_latest_score("id0", scores_path=str(tmp_path))


def test_get_latest_score_falla_si_obligacion_no_existe(monkeypatch, tmp_path) -> None:
    _patch_pipeline(monkeypatch, n=5)
    config = {"data": {"scores_path": str(tmp_path)}}
    infer_batch.run_batch_inference(month="2024-01", model=_FakeModel(), config=config, model_version="v1")

    with pytest.raises(KeyError):
        infer_batch.get_latest_score("no-existe", scores_path=str(tmp_path))
