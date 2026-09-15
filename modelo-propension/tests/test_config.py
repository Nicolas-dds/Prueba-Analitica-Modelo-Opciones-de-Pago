"""Verifica que la configuracion parametrizable cargue y tenga las claves esperadas (RNF-10)."""
from config_loader import load_config

REQUIRED_TOP_LEVEL_KEYS = {"data", "split", "model", "registry", "monitoring", "api"}


def test_load_config_returns_expected_sections() -> None:
    config = load_config()
    assert REQUIRED_TOP_LEVEL_KEYS.issubset(config.keys())


def test_registry_section_has_promotion_settings() -> None:
    config = load_config()
    assert "promotion_metric" in config["registry"]
    assert "min_improvement" in config["registry"]
