"""Tipos core (dataclasses frozen) del Sistema Agéntico de Gestión de Cartera en Mora.

Estos tipos son el contrato de datos que fluye entre los 10 componentes descritos en
`.kiro/specs/sistema-agentico-cobranza/design.md` (sección "Components and Interfaces"
y "Data Models"). Se centralizan aquí para que cada componente (implementado en tareas
posteriores dentro de sus respectivos submódulos: `guardrails/`, `context/`, `eligibility/`,
`nba/`, `conversational/`, `escalation/`, `trace/`, `evaluation/`, `human_review/`) los
importe sin duplicar definiciones ni introducir dependencias circulares.

Las validaciones de `__post_init__` reflejan únicamente las reglas estructurales listadas
en la sección "Data Models" de `design.md`. La lógica de negocio profunda (elegibilidad,
cooldowns, ranking, etc.) se implementa en tareas posteriores, no en estos tipos.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Component 1: Guardrail de Entrada (`InputGuardrail`)
# ---------------------------------------------------------------------------


class InputMode(Enum):
    """Modo de operación de una interacción entrante."""

    PROACTIVO = "proactivo"
    REACTIVO = "reactivo"


@dataclass(frozen=True)
class RawInput:
    """Entrada cruda (sin sanitizar) recibida por `InputGuardrail.process`.

    Único tipo de entrada aceptado por `InputGuardrail`, tanto en modo proactivo
    (`mensaje_crudo=None`, ya que no hay mensaje de cliente) como en modo reactivo
    (`mensaje_crudo` contiene el texto tal cual lo escribió el cliente, incluyendo
    cualquier PII o intento de manipulación, previo a cualquier sanitización).
    """

    mode: InputMode
    id_obligacion: str
    mensaje_crudo: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.id_obligacion:
            raise ValueError("id_obligacion no puede ser vacío")


@dataclass(frozen=True)
class SanitizedInput:
    """Salida de `InputGuardrail.process`; único insumo aceptado por el `Orchestrator`."""

    mode: InputMode
    id_obligacion: str
    mensaje_sanitizado: Optional[str]  # None en modo proactivo
    pii_detectada: list[str]  # tipos de PII enmascarados, nunca los valores originales
    riesgo_inyeccion: float  # 0.0 - 1.0
    bloqueado: bool

    def __post_init__(self) -> None:
        if not self.id_obligacion:
            raise ValueError("id_obligacion no puede ser vacío")
        if not (0.0 <= self.riesgo_inyeccion <= 1.0):
            raise ValueError("riesgo_inyeccion debe estar en [0.0, 1.0]")


# ---------------------------------------------------------------------------
# Component 3: Context Builder (`ContextBuilder`)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoreInfo:
    """Score de propensión asociado a una obligación (ver contrato `ScoreOutput`)."""

    score: float
    decil: int
    version_modelo: str
    fecha_calificacion: str
    degradado: bool  # True si se usó fallback en lugar de la API real

    def __post_init__(self) -> None:
        if not (0.0 <= self.score <= 1.0):
            raise ValueError("score debe estar en [0, 1]")
        if not (1 <= self.decil <= 10):
            raise ValueError("decil debe estar en [1, 10]")


@dataclass(frozen=True)
class OfertaAplicada:
    """Registro histórico de una oferta ya aplicada a una obligación."""

    tipo_opcion: str
    fecha_aplicacion: str
    cooldown_meses: int

    def __post_init__(self) -> None:
        if self.cooldown_meses < 0:
            raise ValueError("cooldown_meses no puede ser negativo")


@dataclass(frozen=True)
class ClienteObligacionContext:
    """Contexto completo de una obligación; fluye por todo el pipeline (RF-14, RF-15)."""

    id_obligacion: str
    id_cliente: str
    dias_mora: int
    exposicion: float
    segmento: Optional[str]
    canal_gestion: Optional[str]
    score: ScoreInfo
    ofertas_aplicadas_mes: list[OfertaAplicada]
    ultima_opcion_aplicada: Optional[OfertaAplicada]
    acuerdo_pago_vigente: bool
    restriccion_vigente: bool

    def __post_init__(self) -> None:
        if self.dias_mora < 0:
            raise ValueError("dias_mora no puede ser negativo")
        if self.exposicion < 0:
            raise ValueError("exposicion no puede ser negativa")


# ---------------------------------------------------------------------------
# Component 4: Motor de Elegibilidad (`EligibilityEngine`)
# ---------------------------------------------------------------------------


class TipoOferta(Enum):
    """Tipo de oferta elegible producida por el `EligibilityEngine`."""

    OPCION_PAGO = "opcion_pago"
    ACUERDO_PAGO = "acuerdo_pago"


@dataclass(frozen=True)
class OfertaElegible:
    """Oferta individual determinada como elegible por el `EligibilityEngine`."""

    tipo: TipoOferta
    id_opcion: str
    detalle: dict[str, Any]
    razon_elegibilidad: str  # explicabilidad obligatoria (RNF-04)

    def __post_init__(self) -> None:
        if not self.id_opcion:
            raise ValueError("id_opcion no puede ser vacío")
        if not self.razon_elegibilidad.strip():
            raise ValueError("razon_elegibilidad no puede ser vacía (RNF-04)")


@dataclass(frozen=True)
class EligibilityResult:
    """Resultado de `EligibilityEngine.evaluate`."""

    opciones_elegibles: list[OfertaElegible]  # máx. 3
    acuerdo_elegible: Optional[OfertaElegible]
    motivo_no_elegible: Optional[str]  # obligatorio si ambas listas están vacías

    def __post_init__(self) -> None:
        if len(self.opciones_elegibles) > 3:
            raise ValueError("opciones_elegibles no puede exceder 3 (RF-15)")
        sin_elegibilidad = not self.opciones_elegibles and self.acuerdo_elegible is None
        if sin_elegibilidad and not (self.motivo_no_elegible and self.motivo_no_elegible.strip()):
            raise ValueError(
                "motivo_no_elegible es obligatorio cuando no hay opciones ni acuerdo elegibles (RF-16, RNF-04)"
            )


# ---------------------------------------------------------------------------
# Component 5: Ranking NBA (`NBARanker`)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WhitelistItem:
    """Ítem individual de la whitelist inmutable producida por `NBARanker.rank`."""

    oferta: OfertaElegible
    prioridad: int  # 1 = mejor alternativa
    justificacion: str  # explicación en términos de negocio (RNF-04)

    def __post_init__(self) -> None:
        if self.prioridad < 1:
            raise ValueError("prioridad debe ser >= 1")
        if not self.justificacion.strip():
            raise ValueError("justificacion no puede ser vacía (RNF-04)")


# ---------------------------------------------------------------------------
# Component 6: Agente Conversacional (`ConversationalAgent`)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentResponse:
    """Respuesta generada por el único LLM del flujo."""

    texto: str
    ofertas_mencionadas: list[str]  # ids de oferta que el LLM afirma haber mencionado
    requiere_reintento: bool
    prompt_version: str

    def __post_init__(self) -> None:
        if not self.prompt_version:
            raise ValueError("prompt_version no puede ser vacío (RNF-05, RNF-10)")


# ---------------------------------------------------------------------------
# Component 7: Guardrail de Salida y Escalamiento
# (`OutputGuardrail` / `EscalationManager`)
# ---------------------------------------------------------------------------


class EscalationReason(Enum):
    """Motivo determinista de escalamiento a gestor humano (RF-19)."""

    OFERTA_NO_AUTORIZADA = "oferta_no_autorizada"
    MANIPULACION_DETECTADA = "manipulacion_detectada"
    INFO_CONTRADICTORIA = "info_contradictoria"
    DIFICULTAD_SEVERA = "dificultad_severa"
    FUERA_DE_ALCANCE = "fuera_de_alcance"
    FALLO_TECNICO = "fallo_tecnico"
    REINTENTO_FALLIDO = "reintento_fallido"


@dataclass(frozen=True)
class ValidationResult:
    """Resultado de `OutputGuardrail.validate`."""

    valido: bool
    motivo_rechazo: Optional[str]
    escalar: bool
    razon_escalamiento: Optional[EscalationReason]

    def __post_init__(self) -> None:
        if self.valido and self.motivo_rechazo:
            raise ValueError("una validación válida no puede tener motivo_rechazo")


# ---------------------------------------------------------------------------
# Component 8: Registro de Trazabilidad (`TraceLogger`)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TraceRecord:
    """Registro individual, inmutable y versionado, de una interacción completa."""

    trace_id: str
    timestamp: str
    id_obligacion: str
    modo: InputMode
    score_info: ScoreInfo
    reglas_evaluadas: dict[str, Any]
    eligibility_result: EligibilityResult
    whitelist: list[WhitelistItem]
    ofertas_mencionadas: list[str]
    respuesta_agente: str
    agente_responsable: str  # "conversational_agent" | "human:<id_gestor>"
    validation_result: ValidationResult
    prompt_version: str
    version_modelo_propension: str

    def __post_init__(self) -> None:
        if not self.trace_id:
            raise ValueError("trace_id no puede ser vacío (RF-21)")
        if not self.agente_responsable:
            raise ValueError("agente_responsable no puede ser vacío")


# ---------------------------------------------------------------------------
# Component 9: Dataset Dorado y Evaluación (`GoldenDatasetEvaluator`)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GoldenScenario:
    """Caso sintético con contexto y resultado esperado (RF-24)."""

    scenario_id: str
    descripcion: str
    contexto: ClienteObligacionContext
    mensaje_cliente: Optional[str]
    resultado_esperado: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValueError("scenario_id no puede ser vacío")


@dataclass(frozen=True)
class EvaluationOutcome:
    """Resultado de evaluar un `GoldenScenario` contra el `Orchestrator` (RF-25)."""

    scenario_id: str
    metrica: str
    umbral_aceptacion: float
    resultado: float
    aprobado: bool
    oportunidad_mejora: Optional[str]

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValueError("scenario_id no puede ser vacío")
        if not self.metrica:
            raise ValueError("metrica no puede ser vacía")


# ---------------------------------------------------------------------------
# Component 10: Gestor Humano y Auditor (`HumanReviewInterface`)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EscalationCase:
    """Caso escalado con contexto completo para el gestor humano (RF-19)."""

    trace_id: str
    id_obligacion: str
    razon: EscalationReason
    contexto: ClienteObligacionContext
    historial_conversacion: list[Any]
    trace_completo: TraceRecord

    def __post_init__(self) -> None:
        if not self.trace_id:
            raise ValueError("trace_id no puede ser vacío")
        if not self.id_obligacion:
            raise ValueError("id_obligacion no puede ser vacío")
