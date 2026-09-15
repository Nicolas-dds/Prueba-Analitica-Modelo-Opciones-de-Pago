"""Pure, JSON-ready reporting for golden-dataset evaluation outcomes."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Any

from sistema_agentico.types import EvaluationOutcome

__all__ = ["EvaluationReportBuilder", "ScenarioEvaluationReportRow", "build_evaluation_report"]


@dataclass(frozen=True)
class ScenarioEvaluationReportRow:
    """One deterministic report row for a single evaluated golden scenario."""

    scenario_id: str
    metrica: str
    umbral_aceptacion: float
    resultado: float
    aprobado: bool
    oportunidad_mejora: str | None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation with the public outcome field names."""
        return {
            "scenario_id": self.scenario_id,
            "metrica": self.metrica,
            "umbral_aceptacion": self.umbral_aceptacion,
            "resultado": self.resultado,
            "aprobado": self.aprobado,
            "oportunidad_mejora": self.oportunidad_mejora,
        }


class EvaluationReportBuilder:
    """Build a deterministic report without invoking evaluation pipelines."""

    @staticmethod
    def build(outcomes: list[EvaluationOutcome]) -> list[ScenarioEvaluationReportRow]:
        if not isinstance(outcomes, list):
            raise TypeError("outcomes debe ser una lista de EvaluationOutcome")

        rows: list[ScenarioEvaluationReportRow] = []
        for index, outcome in enumerate(outcomes):
            _validate_outcome(outcome, index)
            rows.append(
                ScenarioEvaluationReportRow(
                    scenario_id=outcome.scenario_id,
                    metrica=outcome.metrica,
                    umbral_aceptacion=float(outcome.umbral_aceptacion),
                    resultado=float(outcome.resultado),
                    aprobado=outcome.aprobado,
                    oportunidad_mejora=outcome.oportunidad_mejora,
                )
            )
        return rows


def build_evaluation_report(outcomes: list[EvaluationOutcome]) -> list[ScenarioEvaluationReportRow]:
    """Build ordered scenario rows from ``GoldenDatasetEvaluator.run`` output."""
    return EvaluationReportBuilder.build(outcomes)


def _validate_outcome(outcome: object, index: int) -> None:
    if not isinstance(outcome, EvaluationOutcome):
        raise TypeError(f"outcomes[{index}] debe ser EvaluationOutcome")
    if not isinstance(outcome.scenario_id, str) or not outcome.scenario_id.strip():
        raise ValueError(f"outcomes[{index}].scenario_id debe ser texto no vacío")
    if not isinstance(outcome.metrica, str) or not outcome.metrica.strip():
        raise ValueError(f"outcomes[{index}].metrica debe ser texto no vacío")
    for field_name in ("umbral_aceptacion", "resultado"):
        value = getattr(outcome, field_name)
        if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
            raise ValueError(f"outcomes[{index}].{field_name} debe ser un número finito")
    if not isinstance(outcome.aprobado, bool):
        raise TypeError(f"outcomes[{index}].aprobado debe ser bool")
    if outcome.oportunidad_mejora is not None and not isinstance(outcome.oportunidad_mejora, str):
        raise TypeError(f"outcomes[{index}].oportunidad_mejora debe ser str o None")
