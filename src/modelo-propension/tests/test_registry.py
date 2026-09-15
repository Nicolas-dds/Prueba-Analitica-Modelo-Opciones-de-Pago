"""Pruebas de pipeline.registry.load_model (RNF-14)."""
import pytest

from pipeline import registry


class _DummyModel:
    def __init__(self, value: int) -> None:
        self.value = value


def test_load_model_recupera_el_objeto_registrado(tmp_path) -> None:
    model = _DummyModel(value=42)
    version = registry.register_version(model, metadata={"metricas": {"f1": 0.5}}, registry_path=tmp_path)

    loaded = registry.load_model(version, registry_path=tmp_path)

    assert loaded.value == 42


def test_load_model_falla_si_la_version_no_existe(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        registry.load_model("v-inexistente", registry_path=tmp_path)
