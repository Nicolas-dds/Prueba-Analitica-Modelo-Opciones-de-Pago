"""Pruebas unitarias basicas de estructura para los tipos core (`sistema_agentico.types`).

Cubre la tarea 1 del plan de implementacion: verifica que los dataclasses core son
importables, frozen, y que sus validaciones estructurales minimas (segun la seccion
"Data Models" de design.md) funcionan como se espera. La logica de negocio profunda
(elegibilidad, ranking, etc.) se prueba en las tareas correspondientes a cada componente.
"""
import dataclasses

import pytest

from sistema_agentico.types import (
    AgentResponse,
    ClienteObligacionContext,
    EligibilityResult,
    EscalationCase,
    EscalationReason,
    EvaluationOutcome,
    GoldenScenario,
    InputMode,
    OfertaAplicada,
    OfertaElegible,
    SanitizedInput,
    ScoreInfo,
    TipoOferta,
    TraceRecord,
    ValidationResult,
    WhitelistItem,
)


def _score_info(degradado: bool = False) -> ScoreInfo:
    return ScoreInfo(
        score=0.5,
        decil=5,
        version_modelo="v1.0.0",
        fecha_calificacion="2025-01-01",
        degradado=degradado,
    )


def _contexto() -> ClienteObligacionContext:
    return ClienteObligacionContext(
        id_obligacion="OBL-1",
        id_cliente="CLI-1",
        dias_mora=10,
        exposicion=1000.0,
        segmento="masivo",
        canal_gestion="directo",
        score=_score_info(),
        ofertas_aplicadas_mes=[],
        ultima_opcion_aplicada=None,
        acuerdo_pago_vigente=False,
        restriccion_vigente=False,
    )


def _oferta_elegible() -> OfertaElegible:
    return OfertaElegible(
        tipo=TipoOferta.OPCION_PAGO,
        id_opcion="ampliacion_plazo",
        detalle={},
        razon_elegibilidad="Preaprobada, sin cooldown activo",
    )


class TestDataclassesFrozen:
    def test_all_core_types_are_frozen(self) -> None:
        for cls in (
            SanitizedInput,
            ScoreInfo,
            ClienteObligacionContext,
            OfertaAplicada,
            OfertaElegible,
            EligibilityResult,
            WhitelistItem,
            AgentResponse,
            ValidationResult,
            TraceRecord,
            GoldenScenario,
            EvaluationOutcome,
            EscalationCase,
        ):
            assert dataclasses.is_dataclass(cls)
            assert cls.__dataclass_params__.frozen is True

    def test_frozen_instance_raises_on_mutation(self) -> None:
        score = _score_info()
        with pytest.raises(dataclasses.FrozenInstanceError):
            score.score = 0.9  # type: ignore[misc]


class TestSanitizedInput:
    def test_valid_construction(self) -> None:
        sanitized = SanitizedInput(
            mode=InputMode.REACTIVO,
            id_obligacion="OBL-1",
            mensaje_sanitizado="hola",
            pii_detectada=[],
            riesgo_inyeccion=0.1,
            bloqueado=False,
        )
        assert sanitized.mode is InputMode.REACTIVO

    def test_riesgo_inyeccion_fuera_de_rango_lanza_error(self) -> None:
        with pytest.raises(ValueError):
            SanitizedInput(
                mode=InputMode.PROACTIVO,
                id_obligacion="OBL-1",
                mensaje_sanitizado=None,
                pii_detectada=[],
                riesgo_inyeccion=1.5,
                bloqueado=False,
            )

    def test_id_obligacion_vacio_lanza_error(self) -> None:
        with pytest.raises(ValueError):
            SanitizedInput(
                mode=InputMode.PROACTIVO,
                id_obligacion="",
                mensaje_sanitizado=None,
                pii_detectada=[],
                riesgo_inyeccion=0.0,
                bloqueado=False,
            )


class TestScoreInfo:
    def test_score_fuera_de_rango_lanza_error(self) -> None:
        with pytest.raises(ValueError):
            ScoreInfo(score=1.5, decil=5, version_modelo="v1", fecha_calificacion="2025-01-01", degradado=False)

    def test_decil_fuera_de_rango_lanza_error(self) -> None:
        with pytest.raises(ValueError):
            ScoreInfo(score=0.5, decil=11, version_modelo="v1", fecha_calificacion="2025-01-01", degradado=False)


class TestClienteObligacionContext:
    def test_dias_mora_negativo_lanza_error(self) -> None:
        with pytest.raises(ValueError):
            dataclasses.replace(_contexto(), dias_mora=-1)

    def test_exposicion_negativa_lanza_error(self) -> None:
        with pytest.raises(ValueError):
            dataclasses.replace(_contexto(), exposicion=-1.0)


class TestEligibilityResult:
    def test_mas_de_tres_opciones_lanza_error(self) -> None:
        opciones = [_oferta_elegible() for _ in range(4)]
        with pytest.raises(ValueError):
            EligibilityResult(opciones_elegibles=opciones, acuerdo_elegible=None, motivo_no_elegible=None)

    def test_sin_elegibilidad_requiere_motivo(self) -> None:
        with pytest.raises(ValueError):
            EligibilityResult(opciones_elegibles=[], acuerdo_elegible=None, motivo_no_elegible=None)

    def test_sin_elegibilidad_con_motivo_es_valido(self) -> None:
        resultado = EligibilityResult(
            opciones_elegibles=[], acuerdo_elegible=None, motivo_no_elegible="cooldown activo"
        )
        assert resultado.motivo_no_elegible == "cooldown activo"


class TestOfertaElegible:
    def test_razon_elegibilidad_vacia_lanza_error(self) -> None:
        with pytest.raises(ValueError):
            OfertaElegible(tipo=TipoOferta.OPCION_PAGO, id_opcion="x", detalle={}, razon_elegibilidad="  ")


class TestWhitelistItem:
    def test_prioridad_invalida_lanza_error(self) -> None:
        with pytest.raises(ValueError):
            WhitelistItem(oferta=_oferta_elegible(), prioridad=0, justificacion="mejor opcion")

    def test_justificacion_vacia_lanza_error(self) -> None:
        with pytest.raises(ValueError):
            WhitelistItem(oferta=_oferta_elegible(), prioridad=1, justificacion="")


class TestAgentResponse:
    def test_prompt_version_vacio_lanza_error(self) -> None:
        with pytest.raises(ValueError):
            AgentResponse(texto="hola", ofertas_mencionadas=[], requiere_reintento=False, prompt_version="")


class TestValidationResult:
    def test_valido_con_motivo_rechazo_lanza_error(self) -> None:
        with pytest.raises(ValueError):
            ValidationResult(valido=True, motivo_rechazo="algo", escalar=False, razon_escalamiento=None)

    def test_invalido_con_motivo_es_valido(self) -> None:
        resultado = ValidationResult(
            valido=False,
            motivo_rechazo="oferta fuera de whitelist",
            escalar=False,
            razon_escalamiento=None,
        )
        assert resultado.valido is False


class TestEscalationReason:
    def test_todas_las_razones_definidas(self) -> None:
        esperadas = {
            "oferta_no_autorizada",
            "manipulacion_detectada",
            "info_contradictoria",
            "dificultad_severa",
            "fuera_de_alcance",
            "fallo_tecnico",
            "reintento_fallido",
        }
        assert {r.value for r in EscalationReason} == esperadas
