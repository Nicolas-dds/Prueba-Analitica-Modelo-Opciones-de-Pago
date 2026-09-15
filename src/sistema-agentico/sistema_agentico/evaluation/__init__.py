"""Dataset dorado y evaluación (Component 9)."""

from sistema_agentico.evaluation.evaluator import (
    EvaluationExecution,
    GoldenDatasetEvaluator,
    GoldenScenarioPipeline,
    RunnerPipelineAdapter,
)
from sistema_agentico.evaluation.golden_scenarios import (
    MINIMUM_GOLDEN_CATEGORIES,
    build_minimum_golden_scenarios,
)
from sistema_agentico.evaluation.reporting import (
    EvaluationReportBuilder,
    ScenarioEvaluationReportRow,
    build_evaluation_report,
)

__all__ = [
    "EvaluationExecution",
    "EvaluationReportBuilder",
    "GoldenDatasetEvaluator",
    "GoldenScenarioPipeline",
    "MINIMUM_GOLDEN_CATEGORIES",
    "RunnerPipelineAdapter",
    "ScenarioEvaluationReportRow",
    "build_evaluation_report",
    "build_minimum_golden_scenarios",
]
