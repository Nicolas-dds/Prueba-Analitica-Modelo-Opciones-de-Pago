"""Focused coverage for the minimum synthetic GoldenScenario catalogue."""

from sistema_agentico.evaluation import (
    MINIMUM_GOLDEN_CATEGORIES,
    build_minimum_golden_scenarios,
)
from sistema_agentico.synthetic_data.conversations import ScenarioCategoria
from sistema_agentico.synthetic_data.profiles import is_synthetic_context


def test_minimum_golden_scenarios_cover_each_category_and_keep_synthetic_markers() -> None:
    scenarios = build_minimum_golden_scenarios(seed=2026)

    assert scenarios == build_minimum_golden_scenarios(seed=2026)
    assert len(scenarios) == len(MINIMUM_GOLDEN_CATEGORIES) == 7
    assert {scenario.resultado_esperado["categoria"] for scenario in scenarios} == {
        category.value for category in ScenarioCategoria
    }

    for scenario in scenarios:
        expected = scenario.resultado_esperado
        assert scenario.scenario_id.startswith("SYN-GOLD-")
        assert is_synthetic_context(scenario.contexto)
        assert scenario.contexto.score.version_modelo.startswith("SYN-")
        assert scenario.mensaje_cliente
        assert {"categoria", "modo_esperado", "ofertas_esperadas", "escalamiento_esperado", "comportamiento_esperado"} <= expected.keys()

    early_payment = next(
        scenario
        for scenario in scenarios
        if scenario.resultado_esperado["categoria"] == ScenarioCategoria.MORA_TEMPRANA_ACUERDO_PAGO.value
    )
    assert early_payment.resultado_esperado["ofertas_esperadas"] == {
        "debe_incluir": ["acuerdo_pago_estandar"],
        "max_dias_compromiso": 5,
        "solo_ids_preaprobados": True,
    }
