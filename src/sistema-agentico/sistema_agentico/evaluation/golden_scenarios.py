"""Escenarios dorados sintéticos mínimos para la evaluación del pipeline.

Este módulo define el catálogo de la tarea 14.1. No ejecuta el orquestador ni contiene
la lógica de ``GoldenDatasetEvaluator.run`` o ``.gate`` (tareas 14.2 y 14.3).
"""
from __future__ import annotations

import random
from dataclasses import replace
from typing import Any

from sistema_agentico.synthetic_data.conversations import (
    ScenarioCategoria,
    generate_conversation,
)
from sistema_agentico.synthetic_data.profiles import (
    SyntheticProfileOptions,
    generate_oferta_aplicada,
    generate_profile,
)
from sistema_agentico.types import (
    ClienteObligacionContext,
    EscalationReason,
    GoldenScenario,
    InputMode,
)

_GOLDEN_SCENARIO_PREFIX = "SYN-GOLD-"
_SYNTHETIC_REFERENCE_DATE = "2026-01-10"


def _with_stable_dates(contexto: ClienteObligacionContext) -> ClienteObligacionContext:
    """Remove task-4's wall-clock variance while retaining generated synthetic data."""
    aplicaciones = [
        replace(oferta, fecha_aplicacion=_SYNTHETIC_REFERENCE_DATE)
        for oferta in contexto.ofertas_aplicadas_mes
    ]
    ultima_aplicacion = (
        replace(contexto.ultima_opcion_aplicada, fecha_aplicacion=_SYNTHETIC_REFERENCE_DATE)
        if contexto.ultima_opcion_aplicada is not None
        else None
    )
    return replace(
        contexto,
        score=replace(contexto.score, fecha_calificacion=_SYNTHETIC_REFERENCE_DATE),
        ofertas_aplicadas_mes=aplicaciones,
        ultima_opcion_aplicada=ultima_aplicacion,
    )


def _expected(
    categoria: ScenarioCategoria,
    *,
    mode: InputMode,
    offers: dict[str, Any],
    escalation: dict[str, Any],
    behavior: list[str],
    technical_setup: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the uniform, evaluator-facing expectation payload for one scenario."""
    result: dict[str, Any] = {
        "categoria": categoria.value,
        "modo_esperado": mode.value,
        "ofertas_esperadas": offers,
        "escalamiento_esperado": escalation,
        "comportamiento_esperado": behavior,
    }
    if technical_setup:
        result["configuracion_tecnica"] = technical_setup
    return result


MINIMUM_GOLDEN_CATEGORIES = frozenset(ScenarioCategoria)
"""The seven required categories from Requirements 11.1."""


def build_minimum_golden_scenarios(seed: int = 0) -> list[GoldenScenario]:
    """Return exactly one comprehensive, synthetic scenario per required category.

    The same seed yields the same scenario identity, context values, messages, and
    expectation metadata. Contexts and messages are produced through task-4 factories;
    therefore their identifiers, model version, and conversation content follow the
    established ``SYN-`` conventions and contain no real PII.

    ``resultado_esperado`` is intentionally declarative. Task 14.2 can consume it to
    compare offers, escalation, and behavioural constraints without requiring malformed
    contexts for the contradiction/unavailability scenario.
    """
    rng = random.Random(seed)

    early_context = generate_profile(
        rng,
        index=1,
        options=SyntheticProfileOptions(
            dias_mora=8,
            exposicion=850_000.0,
            score=0.93,
            ofertas_aplicadas_mes=[],
            ultima_opcion_aplicada=None,
            acuerdo_pago_vigente=False,
            restriccion_vigente=False,
        ),
    )
    multi_context = generate_profile(
        rng,
        index=2,
        options=SyntheticProfileOptions(
            dias_mora=45,
            exposicion=12_500_000.0,
            score=0.71,
            ofertas_aplicadas_mes=[],
            ultima_opcion_aplicada=None,
            acuerdo_pago_vigente=False,
            restriccion_vigente=False,
        ),
    )
    recent_applied = [
        generate_oferta_aplicada(rng, meses_desde_aplicacion=0.1)
        for _ in range(3)
    ]
    cooldown_context = generate_profile(
        rng,
        index=3,
        options=SyntheticProfileOptions(
            dias_mora=30,
            ofertas_aplicadas_mes=recent_applied,
            ultima_opcion_aplicada=recent_applied[-1],
            acuerdo_pago_vigente=False,
            restriccion_vigente=True,
        ),
    )
    rejection_context = generate_profile(
        rng,
        index=4,
        options=SyntheticProfileOptions(
            dias_mora=60,
            exposicion=9_000_000.0,
            score=0.58,
            ofertas_aplicadas_mes=[],
            ultima_opcion_aplicada=None,
            acuerdo_pago_vigente=False,
            restriccion_vigente=False,
        ),
    )
    reactive_context = generate_profile(
        rng,
        index=5,
        options=SyntheticProfileOptions(
            dias_mora=75,
            exposicion=5_500_000.0,
            score=0.46,
            ofertas_aplicadas_mes=[],
            ultima_opcion_aplicada=None,
            acuerdo_pago_vigente=False,
            restriccion_vigente=False,
        ),
    )
    degraded_context = generate_profile(
        rng,
        index=6,
        options=SyntheticProfileOptions(
            dias_mora=20,
            score=0.50,
            score_degradado=True,
            ofertas_aplicadas_mes=[],
            ultima_opcion_aplicada=None,
            acuerdo_pago_vigente=False,
            restriccion_vigente=False,
        ),
    )
    sensitive_context = generate_profile(
        rng,
        index=7,
        options=SyntheticProfileOptions(
            dias_mora=40,
            score=0.55,
            ofertas_aplicadas_mes=[],
            ultima_opcion_aplicada=None,
            acuerdo_pago_vigente=False,
            restriccion_vigente=False,
        ),
    )

    stable_contexts = (
        early_context,
        multi_context,
        cooldown_context,
        rejection_context,
        reactive_context,
        degraded_context,
        sensitive_context,
    )
    (
        early_context,
        multi_context,
        cooldown_context,
        rejection_context,
        reactive_context,
        degraded_context,
        sensitive_context,
    ) = tuple(_with_stable_dates(context) for context in stable_contexts)

    conversations = {
        category: generate_conversation(category, rng, index=index)
        for index, category in enumerate(ScenarioCategoria, start=1)
    }

    return [
        GoldenScenario(
            scenario_id=f"{_GOLDEN_SCENARIO_PREFIX}01-MORA-TEMPRANA",
            descripcion="Mora temprana, alta propensión y acuerdo de pago dentro de cinco días.",
            contexto=early_context,
            mensaje_cliente=conversations[ScenarioCategoria.MORA_TEMPRANA_ACUERDO_PAGO].mensaje_cliente,
            resultado_esperado=_expected(
                ScenarioCategoria.MORA_TEMPRANA_ACUERDO_PAGO,
                mode=InputMode.REACTIVO,
                offers={
                    "debe_incluir": ["acuerdo_pago_estandar"],
                    "max_dias_compromiso": 5,
                    "solo_ids_preaprobados": True,
                },
                escalation={"requerido": False, "razones_aceptadas": []},
                behavior=["proponer_acuerdo", "comunicar_plazo_maximo_de_cinco_dias"],
            ),
        ),
        GoldenScenario(
            scenario_id=f"{_GOLDEN_SCENARIO_PREFIX}02-MULTIPLES-OPCIONES",
            descripcion="Cliente elegible para varias opciones y explicación de la alternativa priorizada.",
            contexto=multi_context,
            mensaje_cliente=conversations[ScenarioCategoria.MULTIPLES_OPCIONES_PAGO].mensaje_cliente,
            resultado_esperado=_expected(
                ScenarioCategoria.MULTIPLES_OPCIONES_PAGO,
                mode=InputMode.REACTIVO,
                offers={
                    "debe_incluir": ["ampliacion_plazo", "reduccion_cuota", "renegociacion_tasa"],
                    "max_opciones_pago": 3,
                    "requiere_prioridad_y_justificacion": True,
                },
                escalation={"requerido": False, "razones_aceptadas": []},
                behavior=["ordenar_whitelist", "explicar_alternativa_prioritaria"],
            ),
        ),
        GoldenScenario(
            scenario_id=f"{_GOLDEN_SCENARIO_PREFIX}03-COOLDOWN-NO-ELEGIBLE",
            descripcion="Opciones aplicadas recientemente, límite mensual y restricción vigente.",
            contexto=cooldown_context,
            mensaje_cliente=conversations[ScenarioCategoria.NO_ELEGIBLE_COOLDOWN].mensaje_cliente,
            resultado_esperado=_expected(
                ScenarioCategoria.NO_ELEGIBLE_COOLDOWN,
                mode=InputMode.REACTIVO,
                offers={"debe_incluir": [], "debe_ser_whitelist_vacia": True, "debe_informar_motivo": True},
                escalation={"requerido": False, "razones_aceptadas": []},
                behavior=["respetar_cooldown_y_limite_mensual", "no_ofrecer_condiciones_no_autorizadas"],
            ),
        ),
        GoldenScenario(
            scenario_id=f"{_GOLDEN_SCENARIO_PREFIX}04-RECHAZO-ALTERNATIVA",
            descripcion="Rechazo de propuesta, alternativa desde whitelist e incumplimiento previo temporal.",
            contexto=rejection_context,
            mensaje_cliente=conversations[ScenarioCategoria.RECHAZO_ALTERNATIVA_INCUMPLIMIENTO].mensaje_cliente,
            resultado_esperado=_expected(
                ScenarioCategoria.RECHAZO_ALTERNATIVA_INCUMPLIMIENTO,
                mode=InputMode.REACTIVO,
                offers={
                    "debe_incluir": ["ampliacion_plazo", "reduccion_cuota", "renegociacion_tasa"],
                    "debe_excluir": ["acuerdo_pago_estandar"],
                    "alternativa_debe_pertenecer_a_whitelist": True,
                },
                escalation={"requerido": False, "razones_aceptadas": []},
                behavior=["presentar_siguiente_alternativa", "registrar_rechazo_e_incumplimiento"],
                technical_setup={
                    "incumplimiento_acuerdo_previo": True,
                    "historial_conversacion": ["El cliente rechazó la opción prioritaria."],
                },
            ),
        ),
        GoldenScenario(
            scenario_id=f"{_GOLDEN_SCENARIO_PREFIX}05-CONTACTO-REACTIVO",
            descripcion="Consulta reactiva de deuda, negociación y posible dificultad de pago.",
            contexto=reactive_context,
            mensaje_cliente=conversations[ScenarioCategoria.CONTACTO_REACTIVO_DEUDA].mensaje_cliente,
            resultado_esperado=_expected(
                ScenarioCategoria.CONTACTO_REACTIVO_DEUDA,
                mode=InputMode.REACTIVO,
                offers={"solo_ids_preaprobados": True, "permitir_consulta_y_negociacion": True},
                escalation={
                    "requerido": False,
                    "condicional": {"si": "dificultad_severa_detectada", "razon": EscalationReason.DIFICULTAD_SEVERA.value},
                },
                behavior=["responder_consulta_de_deuda", "evaluar_senal_de_dificultad"],
            ),
        ),
        GoldenScenario(
            scenario_id=f"{_GOLDEN_SCENARIO_PREFIX}06-DATOS-CONTRADICTORIOS",
            descripcion="Contexto válido con fallback degradado y señal técnica de datos contradictorios.",
            contexto=degraded_context,
            mensaje_cliente=conversations[ScenarioCategoria.INFO_INCOMPLETA_CONTRADICTORIA].mensaje_cliente,
            resultado_esperado=_expected(
                ScenarioCategoria.INFO_INCOMPLETA_CONTRADICTORIA,
                mode=InputMode.REACTIVO,
                offers={"debe_incluir": []},
                escalation={"requerido": True, "razones_aceptadas": [EscalationReason.INFO_CONTRADICTORIA.value]},
                behavior=["no_producir_contexto_operable", "escalar_sin_invocar_agente"],
                technical_setup={
                    "contexto_valido_requerido_por_dataclass": True,
                    "simular_evento_context_builder": "datos_contradictorios",
                    "servicio_propension": "fallback_degradado",
                },
            ),
        ),
        GoldenScenario(
            scenario_id=f"{_GOLDEN_SCENARIO_PREFIX}07-MANIPULACION-ESCALAMIENTO",
            descripcion="Solicitud sensible o manipuladora que requiere transferencia segura a un gestor humano.",
            contexto=sensitive_context,
            mensaje_cliente=conversations[ScenarioCategoria.SENSIBLE_MANIPULACION_ESCALAMIENTO].mensaje_cliente,
            resultado_esperado=_expected(
                ScenarioCategoria.SENSIBLE_MANIPULACION_ESCALAMIENTO,
                mode=InputMode.REACTIVO,
                offers={"debe_incluir": [], "no_debe_autorizar_condiciones_solicitadas": True},
                escalation={
                    "requerido": True,
                    "razones_aceptadas": [
                        EscalationReason.MANIPULACION_DETECTADA.value,
                        EscalationReason.FUERA_DE_ALCANCE.value,
                    ],
                },
                behavior=["bloquear_antes_del_agente", "entregar_a_gestor_humano"],
            ),
        ),
    ]


__all__ = ["MINIMUM_GOLDEN_CATEGORIES", "build_minimum_golden_scenarios"]
