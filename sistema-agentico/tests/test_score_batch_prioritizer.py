"""Tests unitarios de `ScoreBatchPrioritizer` (tarea 20.4).

Ver `sistema_agentico.orchestration.score_batch_prioritizer` para el contexto completo
del gap que cierra este componente (Requirement 1, AC1 / RF-13).
"""
from __future__ import annotations

from sistema_agentico.context.fake_propension_client import (
    PropensionServiceUnavailableError,
    ScoreNotFoundError,
)
from sistema_agentico.context.fallback_score_strategy import FallbackScoreStrategy
from sistema_agentico.orchestration.score_batch_prioritizer import ScoreBatchPrioritizer
from sistema_agentico.types import ScoreInfo


class _StubPropensionClient:
    """Cliente de propensión determinista: responde por score o lanza el error configurado."""

    def __init__(
        self,
        scores: dict[str, float] | None = None,
        failing: dict[str, Exception] | None = None,
    ) -> None:
        self._scores = scores or {}
        self._failing = failing or {}
        self.calls: list[str] = []

    def get_score(self, id_obligacion: str, timeout_seconds: float) -> ScoreInfo:
        self.calls.append(id_obligacion)
        if id_obligacion in self._failing:
            raise self._failing[id_obligacion]
        score = self._scores[id_obligacion]
        return ScoreInfo(
            score=score,
            decil=5,
            version_modelo="stub-v1",
            fecha_calificacion="2026-01-01",
            degradado=False,
        )


def test_orders_candidates_by_descending_score() -> None:
    """(a) Orden descendente correcto para una mezcla de scores."""
    client = _StubPropensionClient({"A": 0.2, "B": 0.9, "C": 0.5})
    prioritizer = ScoreBatchPrioritizer(client)

    result = prioritizer.prioritize(["A", "B", "C"])

    assert result == ["B", "C", "A"]


def test_id_with_unresolvable_score_and_no_fallback_ends_up_last_but_present() -> None:
    """(b) Sin fallback inyectado: un id cuyo score falla se trata como prioridad mínima,
    nunca se descarta del resultado."""
    client = _StubPropensionClient(
        scores={"A": 0.5, "C": 0.1},
        failing={"B": ScoreNotFoundError("B")},
    )
    prioritizer = ScoreBatchPrioritizer(client)  # sin fallback_strategy

    result = prioritizer.prioritize(["A", "B", "C"])

    assert set(result) == {"A", "B", "C"}
    assert result[-1] == "B"
    assert result == ["A", "C", "B"]


def test_id_with_unresolvable_score_uses_injected_fallback_score() -> None:
    """(c) Con fallback inyectado: se usa el score degradado en vez de prioridad mínima."""
    fallback_strategy = FallbackScoreStrategy()
    # Score neutro conservador por defecto = 0.3 (ver fallback_score_strategy.py),
    # que queda entre los scores reales de A (0.1) y C (0.9).
    client = _StubPropensionClient(
        scores={"A": 0.1, "C": 0.9},
        failing={"B": PropensionServiceUnavailableError("B")},
    )
    prioritizer = ScoreBatchPrioritizer(client, fallback_strategy)

    result = prioritizer.prioritize(["A", "B", "C"])

    assert result == ["C", "B", "A"]


def test_empty_input_returns_empty_output() -> None:
    """(d) Lote vacío -> lista vacía, sin error."""
    client = _StubPropensionClient()
    prioritizer = ScoreBatchPrioritizer(client)

    assert prioritizer.prioritize([]) == []


def test_ties_preserve_original_relative_order() -> None:
    """(e) Empate exacto de score: se preserva el orden relativo original de entrada."""
    client = _StubPropensionClient({"A": 0.5, "B": 0.5, "C": 0.5, "D": 0.9})
    prioritizer = ScoreBatchPrioritizer(client)

    result = prioritizer.prioritize(["A", "B", "C", "D"])

    # D tiene el score más alto y va primero; A, B, C están empatados y deben
    # conservar su orden relativo original (A antes de B antes de C).
    assert result == ["D", "A", "B", "C"]


def test_a_single_failure_does_not_abort_prioritizing_the_rest() -> None:
    """Un fallo individual de score no debe abortar la priorización del resto del lote."""
    client = _StubPropensionClient(
        scores={"A": 0.4, "C": 0.6},
        failing={"B": RuntimeError("fallo inesperado del cliente")},
    )
    prioritizer = ScoreBatchPrioritizer(client)

    result = prioritizer.prioritize(["A", "B", "C"])

    assert set(result) == {"A", "B", "C"}
    assert client.calls == ["A", "B", "C"]
