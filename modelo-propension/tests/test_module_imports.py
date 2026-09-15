"""Verifica que todos los modulos del esqueleto importen sin errores (RNF-14)."""
import importlib

import pytest

MODULES = [
    "config_loader",
    "schemas.contracts",
    "pipeline.ingest",
    "pipeline.features",
    "pipeline.split",
    "pipeline.train",
    "pipeline.explain",
    "pipeline.registry",
    "pipeline.infer_batch",
    "pipeline.monitor",
    "orchestration.run_pipeline",
    "orchestration.run_monitoring",
    "orchestration.generate_submission",
    "api.dependencies",
    "api.main",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name: str) -> None:
    importlib.import_module(module_name)
