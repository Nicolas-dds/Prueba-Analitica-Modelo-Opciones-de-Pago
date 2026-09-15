"""Pruebas de pipeline.features con datos sinteticos pequenos (RNF-14).

El caso mas importante es la prueba anti-fuga: verifica que build_features
NUNCA traiga informacion posterior a t = fecha_var_rpta_alt - 1 mes (RF-03).
"""
import pandas as pd

from pipeline.features import NON_FEATURE_COLUMNS, build_features, get_feature_columns

CONFIG = {"data": {"label_column": "var_rpta_alt"}}


def _make_datasets() -> dict[str, pd.DataFrame]:
    trtest = pd.DataFrame(
        {
            "id_obligacion": ["1#10#100"],
            "nit_enmascarado": [1],
            "num_oblig_enmascarado": [100],
            "fecha_var_rpta_alt": [202309],  # t = 202308 (agosto)
            "var_rpta_alt": [1],
            "banca": ["Banca Personas"],
            "vlr_obligacion": [1000.0],
            "min_mora": [5],  # fuga: NO debe aparecer en el resultado
            "marca_alt_apli": ["SI"],  # fuga: NO debe aparecer en el resultado
        }
    )
    probabilidad = pd.DataFrame(
        {
            "nit_enmascarado": [1, 1, 1],
            "num_oblig_enmascarado": [100, 100, 100],
            "fecha_corte": [202307, 202308, 202309],  # el ultimo es POSTERIOR a t
            "prob_propension": [0.5, 0.7, 0.9],
            "prob_alrt_temprana": [0.1, 0.2, 0.3],
            "prob_auto_cura": [0.1, 0.2, 0.3],
            "lote": [3, 2, 1],
        }
    )
    pagos = pd.DataFrame(
        {
            "nit_enmascarado": [1, 1, 1],
            "num_oblig_enmascarado": [100, 100, 100],
            "fecha_corte": [20230731, 20230831, 20230930],  # el ultimo es POSTERIOR a t
            "porc_pago": [0.4, 0.6, 0.99],
            "marca_pago": ["PAGO_MENOS", "PAGO_TOTAL", "PAGO_TOTAL"],
            "pago_total": [100.0, 200.0, 999.0],
            "valor_cuota_mes": [100.0, 100.0, 100.0],
        }
    )
    customer = pd.DataFrame(
        {
            "nit_enmascarado": [1, 1, 1],
            "year": [2023, 2023, 2023],
            "month": [7, 8, 9],  # septiembre es POSTERIOR a t
            "edad_cli": [30, 31, 99],
            "genero_cli": ["F", "F", "F"],
            "estado_civil": ["SOLTERO", "SOLTERO", "SOLTERO"],
            "nivel_academico": ["UNIVERSITARIO", "UNIVERSITARIO", "UNIVERSITARIO"],
            "total_ing": [1000.0, 2000.0, 9999.0],
            "tot_activos": [0.0, 0.0, 0.0],
            "tot_pasivos": [0.0, 0.0, 0.0],
            "segm": ["PERSONAL", "PERSONAL", "PERSONAL"],
            "subsegm": ["MEDIO", "MEDIO", "MEDIO"],
            "region_of": ["CENTRO", "CENTRO", "CENTRO"],
        }
    )
    return {"trtest": trtest, "probabilidad": probabilidad, "pagos": pagos, "customer": customer}


def test_build_features_calcula_fecha_corte_como_mes_anterior() -> None:
    result = build_features(_make_datasets(), reference_month="2023-09", config=CONFIG)
    assert result.loc[0, "fecha_corte"] == pd.Timestamp("2023-08-01")


def test_build_features_excluye_columnas_con_fuga() -> None:
    result = build_features(_make_datasets(), reference_month="2023-09", config=CONFIG)
    assert "min_mora" not in result.columns
    assert "marca_alt_apli" not in result.columns
    assert "tipo_var_rpta_alt" not in result.columns


def test_build_features_no_trae_informacion_posterior_a_t() -> None:
    """Caso critico anti-fuga (RF-03): debe usar la fila de agosto (t), no la de septiembre."""
    result = build_features(_make_datasets(), reference_month="2023-09", config=CONFIG)
    assert result.loc[0, "prob_propension"] == 0.7  # NO 0.9 (septiembre)
    assert result.loc[0, "hist_porc_pago"] == 0.6  # NO 0.99 (septiembre)
    assert result.loc[0, "edad_cli"] == 31  # NO 99 (septiembre)


def test_get_feature_columns_excluye_ids_fechas_y_label() -> None:
    result = build_features(_make_datasets(), reference_month="2023-09", config=CONFIG)
    feature_columns = get_feature_columns(result)
    for excluded in NON_FEATURE_COLUMNS:
        assert excluded not in feature_columns
    assert "prob_propension" in feature_columns
    assert "vlr_obligacion" in feature_columns


def test_build_features_trae_historial_propio_sin_fuga() -> None:
    """La obligacion ya aparecio en agosto (fecha_var_rpta_alt=202308, var_rpta_alt=0);
    la fila de septiembre debe ver ese resultado previo como feature, la de agosto no debe ver nada (es su primera vez)."""
    datasets = _make_datasets()
    fila_previa = datasets["trtest"].iloc[[0]].copy()
    fila_previa["fecha_var_rpta_alt"] = 202308
    fila_previa["var_rpta_alt"] = 0
    fila_previa["id_obligacion"] = "1#10#100-agosto"
    datasets["trtest"] = pd.concat([datasets["trtest"], fila_previa], ignore_index=True)

    result = build_features(datasets, reference_month="2023-09", config=CONFIG)
    result = result.set_index("id_obligacion")

    fila_septiembre = result.loc["1#10#100"]
    assert fila_septiembre["prev_var_rpta_alt"] == 0  # resultado de agosto, ya conocido
    assert fila_septiembre["prev_tenencia_meses"] == 1  # una aparicion previa (agosto)
    assert fila_septiembre["prev_cant_aceptaciones"] == 0  # en agosto no acepto (var_rpta_alt=0)

    fila_agosto = result.loc["1#10#100-agosto"]
    assert pd.isna(fila_agosto["prev_var_rpta_alt"])  # es su primera aparicion, sin historial previo
