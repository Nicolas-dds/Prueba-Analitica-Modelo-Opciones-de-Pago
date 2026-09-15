"""Pruebas de pipeline.ingest con datos sinteticos pequenos (RNF-14)."""
import numpy as np
import pandas as pd
import pytest

from pipeline.ingest import _build_id_obligacion, clean_raw_data, validate_schema


def test_build_id_obligacion_concatena_con_numeral() -> None:
    df = pd.DataFrame(
        {
            "nit_enmascarado": [1, 2],
            "num_oblig_orig_enmascarado": [10, 20],
            "num_oblig_enmascarado": [100, 200],
        }
    )
    result = _build_id_obligacion(df)
    assert list(result) == ["1#10#100", "2#20#200"]


def test_validate_schema_pasa_con_columnas_completas() -> None:
    datasets = {
        "trtest": pd.DataFrame(
            columns=["nit_enmascarado", "num_oblig_orig_enmascarado", "num_oblig_enmascarado",
                     "fecha_var_rpta_alt", "var_rpta_alt"]
        ),
        "oot": pd.DataFrame(
            columns=["nit_enmascarado", "num_oblig_orig_enmascarado", "num_oblig_enmascarado",
                     "fecha_var_rpta_alt"]
        ),
        "probabilidad": pd.DataFrame(
            columns=["nit_enmascarado", "num_oblig_enmascarado", "fecha_corte",
                     "prob_propension", "prob_alrt_temprana", "prob_auto_cura", "lote"]
        ),
        "pagos": pd.DataFrame(
            columns=["nit_enmascarado", "num_oblig_enmascarado", "fecha_corte", "porc_pago", "marca_pago"]
        ),
        "customer": pd.DataFrame(columns=["nit_enmascarado", "year", "month"]),
    }
    validate_schema(datasets)  # no debe lanzar


def test_validate_schema_falla_si_falta_columna() -> None:
    datasets = {
        "trtest": pd.DataFrame(columns=["nit_enmascarado"]),  # le faltan casi todas
        "oot": pd.DataFrame(
            columns=["nit_enmascarado", "num_oblig_orig_enmascarado", "num_oblig_enmascarado",
                     "fecha_var_rpta_alt"]
        ),
        "probabilidad": pd.DataFrame(
            columns=["nit_enmascarado", "num_oblig_enmascarado", "fecha_corte",
                     "prob_propension", "prob_alrt_temprana", "prob_auto_cura", "lote"]
        ),
        "pagos": pd.DataFrame(
            columns=["nit_enmascarado", "num_oblig_enmascarado", "fecha_corte", "porc_pago", "marca_pago"]
        ),
        "customer": pd.DataFrame(columns=["nit_enmascarado", "year", "month"]),
    }
    with pytest.raises(ValueError, match="trtest"):
        validate_schema(datasets)


def test_validate_schema_falla_si_falta_fuente_completa() -> None:
    with pytest.raises(ValueError, match="falta la fuente"):
        validate_schema({})


def test_clean_raw_data_reemplaza_inf_por_nan() -> None:
    datasets = {
        "pagos": pd.DataFrame({"porc_pago": [0.5, np.inf, -np.inf, 1.2], "marca_pago": ["A", "B", "A", "B"]})
    }
    cleaned = clean_raw_data(datasets)
    assert cleaned["pagos"]["porc_pago"].isna().sum() == 2
    assert list(cleaned["pagos"]["marca_pago"]) == ["A", "B", "A", "B"]  # columnas no numericas intactas
