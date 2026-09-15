"""Ensamblaje único del pipeline completo (Components 1-8 de `design.md`).

Extraído de `run_demo.py` (tarea 17.3) para que tanto el script de demostración como
la futura API HTTP (`sistema_agentico.api`) construyan exactamente el mismo pipeline,
con el mismo comportamiento por defecto (componentes fake/en memoria, sin red, sin
LLM real) y las mismas banderas de entorno (`USE_REAL_LLM`, `USE_REAL_PROPENSION`).

Nota (tarea 20.2) — doble consulta al servicio de propensión: `build_pipeline` expone
`propension_client`/`fallback_strategy` (vía `Pipeline`) precisamente para que un
`ScoreBatchPrioritizer` (tarea 20.1) pueda priorizar un lote proactivo ANTES de invocar
`ProactiveBatchRunner.run` reutilizando la misma instancia de cliente que el
`Orchestrator` ya ensamblado aquí usará después, dentro de `ContextBuilder`, para cada
obligación individual. Esto significa que el score de una obligación priorizada se
consulta al servicio de propensión DOS VECES: una en la priorización (para decidir orden
de contacto) y otra dentro de `ContextBuilder.build` (para construir el contexto real de
esa interacción) — `ContextBuilder` no tiene ningún caché que vincule ambas consultas.
Ineficiencia conocida y aceptada para este prototipo (RNF-13), no oculta; ver también el
docstring de `sistema_agentico.orchestration.score_batch_prioritizer`.
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import NamedTuple

from sistema_agentico.context import (
    ContextBuilder,
    CsvObligacionProfileProvider,
    FakePropensionClient,
    FallbackScoreStrategy,
    HttpPropensionClient,
    InMemoryObligacionProfileProvider,
)
from sistema_agentico.conversational import (
    ConversationalAgent,
    DeterministicFakeChatModel,
    build_chat_openai,
)
from sistema_agentico.eligibility import EligibilityEngine, ReglasElegibilidad
from sistema_agentico.escalation import EscalationManager, InMemoryEscalationCaseSink
from sistema_agentico.guardrails import InputGuardrail, OutputGuardrail
from sistema_agentico.nba import NBARanker
from sistema_agentico.orchestration import Orchestrator, ProactiveBatchRunner, ReactiveRunner
from sistema_agentico.settings import PROPENSION_SERVICE_URL, RAW_DATA_DIR
from sistema_agentico.trace import JsonlTraceLogger

__all__ = [
    "USE_REAL_LLM",
    "USE_REAL_PROFILE",
    "USE_REAL_PROPENSION",
    "Pipeline",
    "build_pipeline",
]

# Raíz del servicio: `src/sistema-agentico/` (carpeta padre del paquete `sistema_agentico/`).
_SERVICE_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = _SERVICE_ROOT / "config"
DATA_OUTPUT_DIR = _SERVICE_ROOT / "data_output"

# --- Banderas de modo (todas por defecto en modo local/fake) ---------------
# Mismo nombre y convención (`os.environ.get(...) == "1"`) usados originalmente en
# `run_demo.py`, para que el comportamiento por defecto de la API HTTP sea idéntico.
USE_REAL_LLM = os.environ.get("USE_REAL_LLM") == "1"
USE_REAL_PROPENSION = os.environ.get("USE_REAL_PROPENSION") == "1"
# Tarea 19.3: perfil/historial real (`CsvObligacionProfileProvider`) en vez del
# sintético en memoria. Por defecto (sin definir o "0") el comportamiento es EXACTAMENTE
# el de antes de esta tarea (provider sintético) — cambio de comportamiento opt-in only.
USE_REAL_PROFILE = os.environ.get("USE_REAL_PROFILE") == "1"


class Pipeline(NamedTuple):
    """Componentes ensamblados por `build_pipeline` (tarea 20.3).

    Se extendió de una tupla `(reactive_runner, proactive_runner)` a este `NamedTuple`
    para además exponer `propension_client`/`fallback_strategy`: `run_demo.py` (y
    cualquier otro llamador) los necesita para construir un `ScoreBatchPrioritizer`
    (tarea 20.1) que consulte exactamente la misma instancia de cliente de propensión
    que usará después `ContextBuilder` dentro del `Orchestrator` ya ensamblado — en vez
    de crear una segunda instancia de cliente separada solo para priorizar.

    Sigue soportando desempaquetado posicional por índice (`pipeline[0]`,
    `pipeline[1]`, ...) como cualquier tupla; los llamadores existentes que
    desempaquetaban `reactive_runner, proactive_runner = build_pipeline()` deben
    actualizarse para incluir los dos campos nuevos (o acceder por nombre).
    """

    reactive_runner: ReactiveRunner
    proactive_runner: ProactiveBatchRunner
    propension_client: HttpPropensionClient | FakePropensionClient
    fallback_strategy: FallbackScoreStrategy


def build_pipeline() -> Pipeline:
    """Construye el pipeline completo (idéntico al de `run_demo.py`)."""
    # 1. Cliente de propensión (Component 3)
    if USE_REAL_PROPENSION:
        propension_client = HttpPropensionClient(base_url=PROPENSION_SERVICE_URL)
    else:
        propension_client = FakePropensionClient.with_default_synthetic_scores(n=20, seed=0)

    fallback_strategy = FallbackScoreStrategy()
    if USE_REAL_PROFILE:
        # Fail-fast intencional: si el CSV no está disponible en `RAW_DATA_DIR`,
        # `CsvObligacionProfileProvider` lanza `RawDataUnavailableError` aquí mismo, en
        # tiempo de construcción del pipeline. No se atrapa esa excepción ni se cae de
        # vuelta al provider sintético: activar `USE_REAL_PROFILE=1` sin datos reales
        # disponibles debe fallar de forma ruidosa, no degradar silenciosamente a datos
        # falsos (violaría la intención de "real significa real" de esta bandera).
        profile_provider = CsvObligacionProfileProvider(RAW_DATA_DIR)
    else:
        profile_provider = InMemoryObligacionProfileProvider.with_default_synthetic_profiles(n=20, seed=0)
    context_builder = ContextBuilder(propension_client, fallback_strategy, profile_provider)

    # 2. Motor de elegibilidad (Component 4) — reglas externas + fecha fija (determinismo)
    reglas = ReglasElegibilidad.from_yaml(CONFIG_DIR / "reglas_elegibilidad.yaml")
    eligibility_engine = EligibilityEngine(reglas, clock=lambda: date.today())

    # 3. Ranking NBA (Component 5)
    nba_ranker = NBARanker()

    # 4. Agente conversacional (Component 6, LangChain)
    if USE_REAL_LLM:
        chat_model = build_chat_openai(
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
        )
    else:
        chat_model = DeterministicFakeChatModel(
            responder=lambda prompt: (
                "Gracias por contactarnos. Según su situación, la alternativa "
                "disponible es ampliacion_plazo, que le permite extender el plazo "
                "de pago. ¿Desea continuar con esta opción?"
                if "ampliacion_plazo" in prompt
                else "En este momento no tenemos una alternativa autorizada para ofrecerle."
            )
        )
    conversational_agent = ConversationalAgent(chat_model=chat_model, prompt_version="demo-v1")

    # 5. Guardrail de salida + escalamiento (Component 7)
    output_guardrail = OutputGuardrail()
    case_sink = InMemoryEscalationCaseSink()

    # 6. Trazabilidad (Component 8) — un JSONL en data_output/
    DATA_OUTPUT_DIR.mkdir(exist_ok=True)
    trace_logger = JsonlTraceLogger(DATA_OUTPUT_DIR / "trace_log_demo.jsonl")

    escalation_manager = EscalationManager(trace_logger=trace_logger, case_sink=case_sink)

    # 7. Orquestador (Component 2)
    orchestrator = Orchestrator(
        context_builder,
        eligibility_engine,
        nba_ranker,
        conversational_agent,
        output_guardrail,
        escalation_manager,
        trace_logger,
    )

    # 8. Guardrail de entrada (Component 1) — único punto de entrada
    input_guardrail = InputGuardrail()

    reactive_runner = ReactiveRunner(input_guardrail, orchestrator)
    proactive_runner = ProactiveBatchRunner(input_guardrail, orchestrator)
    return Pipeline(
        reactive_runner=reactive_runner,
        proactive_runner=proactive_runner,
        propension_client=propension_client,
        fallback_strategy=fallback_strategy,
    )
