"""Cobertura enfocada de la bandera `USE_REAL_PROFILE` en `pipeline_factory` (tarea 19.3).

`USE_REAL_PROFILE` (igual que `USE_REAL_LLM`/`USE_REAL_PROPENSION`) se lee de
`os.environ` en tiempo de import del módulo, así que estos tests recargan
`pipeline_factory` con `importlib.reload` tras fijar la variable de entorno vía
`monkeypatch`, en vez de solo mutar el atributo del módulo ya importado.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from sistema_agentico.context import CsvObligacionProfileProvider, InMemoryObligacionProfileProvider

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATA_DIR = _REPO_ROOT / "data"


def _reload_pipeline_factory():
    # `RAW_DATA_DIR` es una constante calculada en tiempo de import de `settings.py`;
    # `pipeline_factory` la importa por nombre (`from ... import RAW_DATA_DIR`), así que
    # hay que recargar primero `settings` para que recoja el `RAW_DATA_DIR` fijado por
    # `monkeypatch.setenv` en el test, y luego `pipeline_factory` para que reimporte el
    # valor ya actualizado.
    import sistema_agentico.pipeline_factory as pipeline_factory_module
    import sistema_agentico.settings as settings_module

    importlib.reload(settings_module)
    return importlib.reload(pipeline_factory_module)


def test_use_real_profile_no_definido_usa_provider_sintetico_por_defecto(monkeypatch: pytest.MonkeyPatch) -> None:
    """Comportamiento por defecto (bandera ausente) debe ser exactamente el de antes de la tarea 19.3."""
    monkeypatch.delenv("USE_REAL_PROFILE", raising=False)
    pipeline_factory = _reload_pipeline_factory()

    assert pipeline_factory.USE_REAL_PROFILE is False

    pipeline = pipeline_factory.build_pipeline()
    profile_provider = pipeline.reactive_runner._orchestrator._context_builder._profile_provider
    assert isinstance(profile_provider, InMemoryObligacionProfileProvider)


@pytest.mark.skipif(not _DATA_DIR.is_dir(), reason="carpeta data/ del repo no disponible")
def test_use_real_profile_1_usa_csv_profile_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USE_REAL_PROFILE", "1")
    monkeypatch.setenv("RAW_DATA_DIR", str(_DATA_DIR))
    pipeline_factory = _reload_pipeline_factory()

    assert pipeline_factory.USE_REAL_PROFILE is True

    # No debe lanzar: con RAW_DATA_DIR válido, build_pipeline() construye el pipeline
    # completo (LLM/propensión siguen en modo fake por defecto) usando el provider real.
    pipeline = pipeline_factory.build_pipeline()
    profile_provider = pipeline.reactive_runner._orchestrator._context_builder._profile_provider
    assert isinstance(profile_provider, CsvObligacionProfileProvider)


def test_use_real_profile_1_sin_datos_falla_al_construir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Fail-fast: activar la bandera sin CSV disponible debe fallar ruidosamente, nunca
    caer de vuelta al provider sintético en silencio."""
    monkeypatch.setenv("USE_REAL_PROFILE", "1")
    monkeypatch.setenv("RAW_DATA_DIR", str(tmp_path))
    pipeline_factory = _reload_pipeline_factory()

    from sistema_agentico.context import RawDataUnavailableError

    with pytest.raises(RawDataUnavailableError):
        pipeline_factory.build_pipeline()
