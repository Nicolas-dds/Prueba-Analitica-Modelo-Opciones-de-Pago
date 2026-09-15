"""Tests de `CsvObligacionProfileProvider` (tarea 19.1) contra el CSV real `trtest`.

Usa directamente el archivo real `data/prueba_op_base_pivot_var_rpta_alt_enmascarado_trtest.csv`
(fixture real de este proyecto, ya presente en el repo) en vez de mockearlo.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sistema_agentico.context.context_builder import ObligacionProfileNotFoundError
from sistema_agentico.context.csv_profile_provider import (
    CsvObligacionProfileProvider,
    RawDataUnavailableError,
    TRTEST_FILENAME,
    _build_id_obligacion,
)

_map_row_to_profile = CsvObligacionProfileProvider._map_row_to_profile

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATA_DIR = _REPO_ROOT / "data"

# id_obligacion real, con una sola fila en trtest, verificado manualmente contra el CSV
# real (ver reporte de la tarea): nit_enmascarado=630611, dias_mora_fin=71,
# vlr_vencido=1000259.0.
_KNOWN_ID = "630611#219718#863073"


def _provider() -> CsvObligacionProfileProvider:
    return CsvObligacionProfileProvider(_DATA_DIR)


@pytest.mark.skipif(not _DATA_DIR.is_dir(), reason="carpeta data/ del repo no disponible")
def test_perfil_real_conocido_resuelve_campos_clave() -> None:
    provider = _provider()
    perfil = provider.get_profile(_KNOWN_ID)

    assert perfil.id_cliente == "630611"
    assert perfil.dias_mora == 71
    assert perfil.exposicion == pytest.approx(1000259.0)


@pytest.mark.skipif(not _DATA_DIR.is_dir(), reason="carpeta data/ del repo no disponible")
def test_id_inexistente_lanza_not_found() -> None:
    provider = _provider()
    with pytest.raises(ObligacionProfileNotFoundError):
        provider.get_profile("0#0#0")


@pytest.mark.skipif(not _DATA_DIR.is_dir(), reason="carpeta data/ del repo no disponible")
def test_cobertura_real_de_ids_duplicados(capsys: pytest.CaptureFixture[str]) -> None:
    """No es una aserción de comportamiento: documenta, para el reporte de la tarea, qué
    fracción de ids reales de `trtest` tiene más de una fila (multi-mes)."""
    df = pd.read_csv(_DATA_DIR / TRTEST_FILENAME, usecols=[
        "nit_enmascarado", "num_oblig_orig_enmascarado", "num_oblig_enmascarado",
    ])
    df = df.assign(id_obligacion=_build_id_obligacion(df))
    conteos = df["id_obligacion"].value_counts()
    con_multiples_filas = int((conteos > 1).sum())
    print(f"ids con >1 fila en trtest: {con_multiples_filas} / {conteos.size}")
    assert con_multiples_filas > 0  # el CSV real SÍ exhibe ids con múltiples fecha_var_rpta_alt


def test_seleccion_de_fila_mas_reciente_con_dataframe_sintetico() -> None:
    """Test double sintético, claramente etiquetado como tal: aunque el CSV real SÍ
    contiene ids con múltiples filas (ver test anterior), se construye aquí un
    DataFrame in-memory mínimo y controlado para verificar sin ambigüedad la regla de
    selección "fila más reciente por fecha_var_rpta_alt"."""
    filas = [
        {
            "nit_enmascarado": 1, "num_oblig_orig_enmascarado": 2, "num_oblig_enmascarado": 3,
            "fecha_var_rpta_alt": 202301, "dias_mora_fin": 10, "vlr_vencido": 100.0,
            "vlr_obligacion": 500.0, "segmento": "Personal", "aplicativo": "V",
            "marca_alt_apli": "NO", "alternativa_aplicada_agr": None, "cant_acuerdo_binario": 0,
        },
        {
            "nit_enmascarado": 1, "num_oblig_orig_enmascarado": 2, "num_oblig_enmascarado": 3,
            "fecha_var_rpta_alt": 202312, "dias_mora_fin": 99, "vlr_vencido": 999.0,
            "vlr_obligacion": 500.0, "segmento": "Personal", "aplicativo": "L",
            "marca_alt_apli": "NO", "alternativa_aplicada_agr": None, "cant_acuerdo_binario": 0,
        },
    ]
    df = pd.DataFrame(filas)
    df = df.assign(id_obligacion=_build_id_obligacion(df))
    mas_reciente = (
        df.sort_values("fecha_var_rpta_alt")
        .drop_duplicates(subset="id_obligacion", keep="last")
        .set_index("id_obligacion")
        .to_dict(orient="index")
    )
    perfil = _map_row_to_profile(mas_reciente["1#2#3"])

    assert perfil.dias_mora == 99  # la fila de 202312 (más reciente), no la de 202301
    assert perfil.exposicion == pytest.approx(999.0)
    assert perfil.canal_gestion == "L"


def test_dias_mora_nan_se_deja_como_none_no_como_cero() -> None:
    fila = {
        "nit_enmascarado": 1, "dias_mora_fin": float("nan"), "vlr_vencido": float("nan"),
        "vlr_obligacion": float("nan"), "segmento": None, "aplicativo": None,
        "marca_alt_apli": "NO", "alternativa_aplicada_agr": None, "cant_acuerdo_binario": 0,
        "fecha_var_rpta_alt": 202301,
    }
    perfil = _map_row_to_profile(fila)

    assert perfil.dias_mora is None
    assert perfil.exposicion is None
    assert perfil.ofertas_aplicadas_mes == []
    assert perfil.ultima_opcion_aplicada is None
    assert perfil.acuerdo_pago_vigente is False
    assert perfil.restriccion_vigente is False


def test_exposicion_usa_vlr_obligacion_como_respaldo_si_vlr_vencido_es_nan() -> None:
    fila = {
        "nit_enmascarado": 1, "dias_mora_fin": 5, "vlr_vencido": float("nan"),
        "vlr_obligacion": 4321.0, "segmento": None, "aplicativo": None,
        "marca_alt_apli": "NO", "alternativa_aplicada_agr": None, "cant_acuerdo_binario": 0,
        "fecha_var_rpta_alt": 202301,
    }
    perfil = _map_row_to_profile(fila)

    assert perfil.exposicion == pytest.approx(4321.0)


def test_alternativa_mapeable_produce_oferta_aplicada() -> None:
    fila = {
        "nit_enmascarado": 1, "dias_mora_fin": 5, "vlr_vencido": 10.0,
        "vlr_obligacion": 20.0, "segmento": "Personal", "aplicativo": "V",
        "marca_alt_apli": "SI", "alternativa_aplicada_agr": "AMPLIACION",
        "cant_acuerdo_binario": 1, "fecha_var_rpta_alt": 202305,
    }
    perfil = _map_row_to_profile(fila)

    assert len(perfil.ofertas_aplicadas_mes) == 1
    oferta = perfil.ofertas_aplicadas_mes[0]
    assert oferta.tipo_opcion == "ampliacion_plazo"
    assert oferta.fecha_aplicacion == "2023-05-01"
    assert oferta.cooldown_meses == 3
    assert perfil.ultima_opcion_aplicada == oferta
    assert perfil.acuerdo_pago_vigente is True


def test_alternativa_no_mapeable_deja_lista_vacia_no_none() -> None:
    fila = {
        "nit_enmascarado": 1, "dias_mora_fin": 5, "vlr_vencido": 10.0,
        "vlr_obligacion": 20.0, "segmento": "Personal", "aplicativo": "V",
        "marca_alt_apli": "SI", "alternativa_aplicada_agr": "CONDONACION",
        "cant_acuerdo_binario": 1, "fecha_var_rpta_alt": 202305,
    }
    perfil = _map_row_to_profile(fila)

    assert perfil.ofertas_aplicadas_mes == []  # nunca None
    assert perfil.ultima_opcion_aplicada is None
    assert perfil.acuerdo_pago_vigente is False  # sin ultima_opcion_aplicada que lo sustente


def test_archivo_csv_faltante_lanza_error_al_construir(tmp_path: Path) -> None:
    with pytest.raises(RawDataUnavailableError):
        CsvObligacionProfileProvider(tmp_path)
