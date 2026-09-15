"""Script de demostración: ejecuta el pipeline completo localmente.

Uso:
    cd src/sistema-agentico
    python3 run_demo.py

Por defecto usa únicamente componentes fake/en memoria (sin red, sin LLM real,
sin el servicio `modelo-propension` corriendo) — ejecución autocontenida (RNF-13).

Para usar un LLM real (OpenAI) o el servicio `modelo-propension` real, ver las
banderas de entorno al final de este archivo (`USE_REAL_LLM`, `USE_REAL_PROPENSION`).
"""
from __future__ import annotations

import os
import sys

# Permite importar `sistema_agentico` sin instalar el paquete (carpeta con guion).
sys.path.insert(0, os.path.dirname(__file__))

from sistema_agentico.orchestration import ScoreBatchPrioritizer  # noqa: E402
from sistema_agentico.pipeline_factory import build_pipeline  # noqa: E402

# Re-exportadas por compatibilidad con quien importe estas banderas desde este script.
USE_REAL_LLM = os.environ.get("USE_REAL_LLM") == "1"
USE_REAL_PROPENSION = os.environ.get("USE_REAL_PROPENSION") == "1"


def main() -> None:
    pipeline = build_pipeline()
    reactive_runner = pipeline.reactive_runner
    proactive_runner = pipeline.proactive_runner

    # Los perfiles/scores sintéticos por defecto usan ids SYN-OBL-00000 .. SYN-OBL-00019.
    id_obligacion = "SYN-OBL-00000"

    print(f"=== Modo reactivo: {id_obligacion} ===")
    outcome = reactive_runner.run(id_obligacion, "Hola, quisiera saber qué opciones tengo para pagar mi deuda")
    print("status:", outcome.status)
    if outcome.response is not None:
        print("respuesta:", outcome.response.texto)
        print("ofertas_mencionadas:", outcome.response.ofertas_mencionadas)
    else:
        print("escalation_reason:", outcome.escalation_reason)
    print("trace_id:", outcome.trace_id)

    print(f"\n=== Modo proactivo: lote de 3 obligaciones ===")
    batch_ids = ["SYN-OBL-00001", "SYN-OBL-00002", "SYN-OBL-00003"]

    # Tarea 20.3: priorizar el lote por score de propensión (RF-13) ANTES de pasarlo a
    # `ProactiveBatchRunner.run`, en vez de asumir implícitamente que `batch_ids` ya
    # viene ordenado. Reutiliza la MISMA instancia de `propension_client`/
    # `fallback_strategy` que `build_pipeline()` ya construyó para el `Orchestrator`
    # (ver `Pipeline` en `pipeline_factory.py`), en vez de crear un segundo cliente
    # separado solo para esta priorización.
    #
    # Nota (tarea 20.2, ver también el docstring de `ScoreBatchPrioritizer`): esta
    # priorización consulta el score de propensión UNA VEZ por obligación, solo para
    # decidir el orden de contacto. `ContextBuilder` volverá a consultarlo DESPUÉS, al
    # procesar cada interacción real dentro de `proactive_runner.run(...)` — una
    # llamada HTTP adicional por obligación, ineficiencia conocida y aceptada para este
    # prototipo, no oculta.
    prioritizer = ScoreBatchPrioritizer(pipeline.propension_client, pipeline.fallback_strategy)
    prioritized_ids = prioritizer.prioritize(batch_ids)

    print("Orden original:   ", batch_ids)
    print("Orden priorizado por score (descendente):")
    for id_obligacion in prioritized_ids:
        try:
            score_info = pipeline.propension_client.get_score(id_obligacion, timeout_seconds=2.0)
            print(f"  - {id_obligacion}: score={score_info.score:.4f} (degradado={score_info.degradado})")
        except Exception as exc:  # noqa: BLE001 - solo impresión demostrativa, no afecta la priorización ya calculada
            print(f"  - {id_obligacion}: score no disponible ({type(exc).__name__})")

    results = proactive_runner.run(prioritized_ids)
    for result in results:
        print(f"- {result.id_obligacion}: completado={result.completed}, "
              f"status={result.outcome.status if result.outcome else result.error}")


if __name__ == "__main__":
    main()
