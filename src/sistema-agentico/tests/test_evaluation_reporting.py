"""Focused coverage for programmatic golden-evaluation reporting."""
from __future__ import annotations

import json
import math

import pytest

from sistema_agentico.evaluation import (
    EvaluationReportBuilder,
    ScenarioEvaluationReportRow,
    build_evaluation_report,
)
from sistema_agentico.types import EvaluationOutcome


def _outcome(
    scenario_id: str,
    *,
    approved: bool = True,
    improvement: str | None = None,
) -> EvaluationOutcome:
    return EvaluationOutcome(
        scenario_id=scenario_id,
        metrica="ofertas_invalidas",
        umbral_aceptacion=0.0,
        resultado=0.0 if approved else 1.0,
        aprobado=approved,
        oportunidad_mejora=improvement,
    )


def test_build_evaluation_report_preserves_order_and_all_required_fields() -> None:
    outcomes = [
        _outcome("SYN-pass"),
        _outcome("SYN-fail", approved=False, improvement="Revisar whitelist."),
    ]

    report = build_evaluation_report(outcomes)

    assert report == [
        ScenarioEvaluationReportRow("SYN-pass", "ofertas_invalidas", 0.0, 0.0, True, None),
        ScenarioEvaluationReportRow(
            "SYN-fail", "ofertas_invalidas", 0.0, 1.0, False, "Revisar whitelist."
        ),
    ]
    assert outcomes[1].oportunidad_mejora == "Revisar whitelist."


def test_report_rows_are_json_ready() -> None:
    row = EvaluationReportBuilder.build([_outcome("SYN-json")])[0]

    assert row.as_dict() == {
        "scenario_id": "SYN-json",
        "metrica": "ofertas_invalidas",
        "umbral_aceptacion": 0.0,
        "resultado": 0.0,
        "aprobado": True,
        "oportunidad_mejora": None,
    }
    assert json.loads(json.dumps(row.as_dict())) == row.as_dict()


@pytest.mark.parametrize(
    ("outcomes", "error"),
    [
        ("not-a-list", TypeError),
        ([object()], TypeError),
    ],
)
def test_report_rejects_invalid_container_or_entries(outcomes: object, error: type[Exception]) -> None:
    with pytest.raises(error):
        build_evaluation_report(outcomes)  # type: ignore[arg-type]


@pytest.mark.parametrize("field_name", ["umbral_aceptacion", "resultado"])
def test_report_rejects_non_finite_metric_values(field_name: str) -> None:
    outcome = _outcome("SYN-invalid")
    object.__setattr__(outcome, field_name, math.nan)

    with pytest.raises(ValueError, match=field_name):
        build_evaluation_report([outcome])


def test_report_rejects_non_boolean_approval_status() -> None:
    outcome = _outcome("SYN-invalid")
    object.__setattr__(outcome, "aprobado", 1)

    with pytest.raises(TypeError, match="aprobado"):
        build_evaluation_report([outcome])
