"""Pruebas estructurales de los contratos pydantic (RF-07, RF-08, RF-09)."""
from datetime import date

import pytest
from pydantic import ValidationError

from schemas.contracts import ModelMetadata, ObligacionInput, ScoreOutput


def test_obligacion_input_valido() -> None:
    obligacion = ObligacionInput(
        id_obligacion="OB-0001",
        id_cliente="CL-0001",
        dias_mora=15,
        exposicion=1_500_000.0,
        fecha_corte=date(2026, 8, 31),
    )
    assert obligacion.dias_mora == 15


def test_obligacion_input_rechaza_dias_mora_negativos() -> None:
    with pytest.raises(ValidationError):
        ObligacionInput(
            id_obligacion="OB-0002",
            id_cliente="CL-0002",
            dias_mora=-1,
            exposicion=1000.0,
            fecha_corte=date(2026, 8, 31),
        )


def test_score_output_rechaza_score_fuera_de_rango() -> None:
    with pytest.raises(ValidationError):
        ScoreOutput(
            id_obligacion="OB-0001",
            score=1.5,
            decil=1,
            version_modelo="v1",
            fecha_calificacion=date(2026, 9, 1),
        )


def test_model_metadata_default_no_es_campeon() -> None:
    metadata = ModelMetadata(version="v1")
    assert metadata.es_campeon is False
