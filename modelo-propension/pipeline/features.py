"""Feature engineering compartido entre entrenamiento e inferencia (RF-05, RF-03).

El mismo `build_features` debe usarse en train.py e infer_batch.py para
evitar 'training-serving skew' (ver componente Data Platform & Feature Store
en la arquitectura del modelo de propension).

HALLAZGO CRITICO (RF-03, sin fuga de informacion): segun Metadata.xlsx, casi
todas las columnas de `trtest` describen el MISMO mes que la variable
respuesta (`fecha_var_rpta_alt`) -- mora, gestiones, pagos y si se aplico la
alternativa son eventos de ESE mes, no del anterior. Usarlas tal cual como
predictoras seria fuga directa. Por eso, para cada obligacion se define
t = fecha_var_rpta_alt - 1 mes, y las variables predictoras solo se
construyen con: (a) atributos estaticos/de catalogo de trtest (producto,
banca, alternativas preaprobadas) que no dependen del mes evaluado, y (b) la
informacion historica de `probabilidad`, `pagos` y `customer` con fecha <= t
(join "as of", nunca posterior a t).
"""
from __future__ import annotations

import pandas as pd

# Atributos de trtest que SI son seguros como predictores: describen el
# producto/catalogo de alternativas preaprobadas, no un resultado del mes de
# la variable respuesta.
STATIC_SAFE_COLUMNS = [
    "banca",
    "segmento",
    "producto",
    "producto_cons",
    "aplicativo",
    "vlr_obligacion",  # valor desembolsado/cupo: atributo fijo de la obligacion, no cambia mes a mes
    "cant_alter_posibles",
    "alter_posible1_2",
    "alter_posible2_2",
    "alter_posible3_2",
    "desc_alternativa1",
    "desc_alternativa2",
    "desc_alternativa3",
]

# Columnas de trtest EXCLUIDAS a proposito (fuga de informacion, RF-03): son
# outcomes del mes de la variable respuesta segun Metadata.xlsx, no inputs
# disponibles de antemano. `tipo_var_rpta_alt` tambien se excluye por ser un
# desglose del propio label.
EXCLUDED_LEAKY_COLUMNS = [
    "tipo_var_rpta_alt",
    "min_mora",
    "max_mora",
    "dias_mora_fin",
    "rango_mora",
    "vlr_vencido",
    "saldo_capital",
    "endeudamiento",
    "cant_gestiones",
    "cant_gestiones_binario",
    "rpc",
    "promesas_cumplidas",
    "cant_promesas_cumplidas_binario",
    "cant_acuerdo",
    "cant_acuerdo_binario",
    "descripcion_ranking_mejor_ult",
    "descripcion_ranking_post_ult",
    "marca_alt_rank",
    "marca_alt_apli",
    "valor_cuota_mes",
    "pago_cuota",
    "porc_pago_cuota",
    "pago_mes",
    "porc_pago_mes",
    "pagos_tanque",
    "marca_debito_mora",
    "alternativa_aplicada_agr",
    "marca_agrupada_rgo",
    "marca_pago",
    "marca_alternativa",
    "marca_alternativa_orig",
]

# Columnas que nunca deben entrar como predictoras (identificadores, fechas,
# label) -- usar junto con el DataFrame resultante de build_features.
NON_FEATURE_COLUMNS = [
    "id_obligacion",
    "nit_enmascarado",
    "num_oblig_enmascarado",
    "fecha_var_rpta_alt",
    "fecha_corte",
    "var_rpta_alt",
]


def _asof_join(
    population: pd.DataFrame,
    source: pd.DataFrame,
    by: list[str],
    left_on: str,
    source_date_col: str,
    prefix: str,
    value_columns: list[str],
) -> pd.DataFrame:
    """Trae, para cada fila de `population`, los valores de `source` con `source_date_col` <= population[left_on] mas recientes (join as-of, agrupado por `by`).

    Implementa la regla "sin fuga": nunca se trae informacion posterior a t.
    El renombrado con `prefix` se hace ANTES del merge (no despues) para que
    nunca colisione con una columna homonima ya presente en `population`
    (ej. un join de respaldo que reusa un nombre de columna del join anterior).
    """
    left = population.sort_values(left_on).reset_index(drop=True)
    renamed = {c: f"{prefix}{c}" for c in value_columns}
    right = (
        source[[*by, source_date_col, *value_columns]]
        .rename(columns=renamed)
        .sort_values(source_date_col)
        .reset_index(drop=True)
    )
    merged = pd.merge_asof(
        left,
        right,
        left_on=left_on,
        right_on=source_date_col,
        by=by,
        direction="backward",
    )
    if source_date_col in merged.columns and source_date_col != left_on:
        merged = merged.drop(columns=[source_date_col])
    return merged


def build_features(
    datasets: dict[str, pd.DataFrame],
    reference_month: str,
    config: dict,
    population: str = "trtest",
) -> pd.DataFrame:
    """Construye las variables predictoras para las obligaciones de `population` hasta `reference_month`.

    Para cada fila de `datasets[population]` (trtest para entrenar, oot para
    calificar) con `fecha_var_rpta_alt` <= `reference_month`, calcula
    t = fecha_var_rpta_alt - 1 mes y arma las features con informacion
    disponible hasta t (ver docstring del modulo, RF-03).

    Args:
        datasets: salida de pipeline.ingest.load_raw_data (trtest, oot,
            probabilidad, pagos, customer).
        reference_month: mes de corte "YYYY-MM"; filtra `population` a filas
            con fecha_var_rpta_alt <= este mes.
        config: configuracion parametrizable (ver config/pipeline_config.yaml).
        population: "trtest" (con label, entrenamiento) u "oot" (sin label,
            inferencia batch).

    Returns:
        DataFrame con id_obligacion, fecha_var_rpta_alt, fecha_corte (= t),
        var_rpta_alt (si `population` la trae) y las columnas de features
        (STATIC_SAFE_COLUMNS + prob_*/hist_*/cli_* de las fuentes historicas).
    """
    label_col = config["data"]["label_column"]
    resp_col = "fecha_var_rpta_alt"

    pop_df = datasets[population].copy()
    pop_df["_resp_period"] = pd.PeriodIndex(pop_df[resp_col].astype(int).astype(str), freq="M")
    ref_period = pd.Period(reference_month, freq="M")
    pop_df = pop_df.loc[pop_df["_resp_period"] <= ref_period].copy()

    # t = mes anterior al de la variable respuesta: unico corte seguro (RF-03).
    # `_t_ts` (inicio de mes t) es lo que se expone como "fecha_corte"; el
    # join as-of usa `_asof_ts` (FIN de mes t) porque `fecha_corte` en
    # pagos/customer suele venir como fecha de fin de mes (ej. 20230831).
    t_period = pop_df["_resp_period"] - 1
    pop_df["_t_ts"] = t_period.dt.to_timestamp()
    pop_df["_asof_ts"] = t_period.dt.to_timestamp(how="end")

    keep_cols = ["id_obligacion", "nit_enmascarado", "num_oblig_enmascarado", resp_col, "_t_ts", "_asof_ts"]
    if label_col in pop_df.columns:
        keep_cols.append(label_col)

    if population == "trtest":
        # trtest trae las columnas estaticas en la propia fila (RF-03: son
        # atributos de producto/catalogo, no dependen del mes evaluado).
        static_cols = [c for c in STATIC_SAFE_COLUMNS if c in pop_df.columns]
        result = pop_df[keep_cols + static_cols].copy()
    else:
        # oot (y cualquier poblacion sin esas columnas en su propia fila,
        # ver Metadata.xlsx: oot solo trae los identificadores + fecha):
        # se obtienen por historial (asof, backward) contra trtest. Solo
        # ~49% de las obligaciones de la OOT real tienen historial en
        # trtest -- para producto/aplicativo/segmento se completa con
        # `pagos` (~99.9% de cobertura) como respaldo. vlr_obligacion y el
        # catalogo de alternativas preaprobadas SOLO existen en trtest: para
        # el ~51% sin historial quedan NaN (XGBoost los trata como
        # faltantes; es una limitacion real de los datos, no del diseño).
        result = pop_df[keep_cols].copy()

        trtest_hist = datasets["trtest"].copy()
        trtest_hist["_fecha_ts"] = pd.PeriodIndex(
            trtest_hist[resp_col].astype(int).astype(str), freq="M"
        ).to_timestamp()
        static_cols_en_trtest = [c for c in STATIC_SAFE_COLUMNS if c in trtest_hist.columns]
        result = _asof_join(
            result,
            trtest_hist,
            by=["nit_enmascarado", "num_oblig_enmascarado"],
            left_on="_asof_ts",
            source_date_col="_fecha_ts",
            prefix="",
            value_columns=static_cols_en_trtest,
        )

        pagos_respaldo = datasets["pagos"].copy()
        pagos_respaldo["_fecha_ts"] = pd.to_datetime(
            pagos_respaldo["fecha_corte"].astype(int).astype(str), format="%Y%m%d"
        )
        result = _asof_join(
            result,
            pagos_respaldo,
            by=["nit_enmascarado", "num_oblig_enmascarado"],
            left_on="_asof_ts",
            source_date_col="_fecha_ts",
            prefix="pagos_",
            value_columns=["producto", "aplicativo", "segmento"],
        )
        for col in ["producto", "aplicativo", "segmento"]:
            result[col] = result[col].fillna(result.pop(f"pagos_{col}"))

    # Historial propio de la obligacion (RF-03: solo filas con
    # fecha_var_rpta_alt <= t, es decir ya observadas -> nunca la fila que
    # se esta evaluando). Senal de comportamiento repetido: si acepto una
    # alternativa antes y hace cuantos meses viene en el panel de gestion.
    own_hist = datasets["trtest"][["nit_enmascarado", "num_oblig_enmascarado", resp_col, label_col]].copy()
    own_hist["_fecha_ts"] = pd.PeriodIndex(own_hist[resp_col].astype(int).astype(str), freq="M").to_timestamp()
    own_hist = own_hist.sort_values("_fecha_ts")
    own_hist["tenencia_meses"] = own_hist.groupby(["nit_enmascarado", "num_oblig_enmascarado"]).cumcount() + 1
    # Suma acumulada (incluyendo la propia fila historica emparejada) de
    # aceptaciones previas -- cuantas veces ya acepto una alternativa antes.
    own_hist["cant_aceptaciones"] = own_hist.groupby(["nit_enmascarado", "num_oblig_enmascarado"])[
        label_col
    ].cumsum()
    result = _asof_join(
        result,
        own_hist,
        by=["nit_enmascarado", "num_oblig_enmascarado"],
        left_on="_asof_ts",
        source_date_col="_fecha_ts",
        prefix="prev_",
        value_columns=[label_col, "tenencia_meses", "cant_aceptaciones"],
    )

    prob = datasets["probabilidad"].copy()
    prob["_fecha_ts"] = pd.PeriodIndex(prob["fecha_corte"].astype(int).astype(str), freq="M").to_timestamp()
    prob = prob.sort_values(["nit_enmascarado", "num_oblig_enmascarado", "_fecha_ts"])
    # Tendencia (no solo el ultimo dato puntual): promedio movil de los
    # ultimos 3 registros disponibles por obligacion (backward-looking, sin
    # fuga -- ver docstring de _asof_join).
    grp = prob.groupby(["nit_enmascarado", "num_oblig_enmascarado"])
    prob["prob_propension_roll3"] = grp["prob_propension"].transform(lambda s: s.rolling(window=3, min_periods=1).mean())
    prob["prob_alrt_temprana_roll3"] = grp["prob_alrt_temprana"].transform(
        lambda s: s.rolling(window=3, min_periods=1).mean()
    )
    prob["prob_auto_cura_roll3"] = grp["prob_auto_cura"].transform(lambda s: s.rolling(window=3, min_periods=1).mean())
    result = _asof_join(
        result,
        prob,
        by=["nit_enmascarado", "num_oblig_enmascarado"],
        left_on="_asof_ts",
        source_date_col="_fecha_ts",
        prefix="",  # nombres ya autodescriptivos (prob_propension, lote, etc.), sin colision
        value_columns=[
            "prob_propension",
            "prob_alrt_temprana",
            "prob_auto_cura",
            "lote",
            "prob_propension_roll3",
            "prob_alrt_temprana_roll3",
            "prob_auto_cura_roll3",
        ],
    )

    pagos = datasets["pagos"].copy()
    pagos["_fecha_ts"] = pd.to_datetime(pagos["fecha_corte"].astype(int).astype(str), format="%Y%m%d")
    pagos = pagos.sort_values(["nit_enmascarado", "num_oblig_enmascarado", "_fecha_ts"])
    pagos_grp = pagos.groupby(["nit_enmascarado", "num_oblig_enmascarado"])
    pagos["porc_pago_roll3"] = pagos_grp["porc_pago"].transform(lambda s: s.rolling(window=3, min_periods=1).mean())
    pagos["pago_total_roll3"] = pagos_grp["pago_total"].transform(lambda s: s.rolling(window=3, min_periods=1).mean())
    result = _asof_join(
        result,
        pagos,
        by=["nit_enmascarado", "num_oblig_enmascarado"],
        left_on="_asof_ts",
        source_date_col="_fecha_ts",
        prefix="hist_",
        value_columns=["porc_pago", "marca_pago", "pago_total", "valor_cuota_mes", "porc_pago_roll3", "pago_total_roll3"],
    )

    # Nivel CLIENTE (no obligacion): cuantas obligaciones distintas tiene el
    # cliente hasta t -- señal de carga de deuda total, no solo de esta
    # obligacion. Se usa la fecha de PRIMERA aparicion de cada obligacion en
    # `pagos` (mayor cobertura que trtest) para contar solo las que ya
    # existian hasta t, sin fuga (nunca cuenta obligaciones abiertas despues).
    primeras_apariciones = (
        pagos.groupby(["nit_enmascarado", "num_oblig_enmascarado"])["_fecha_ts"].min().reset_index()
    )
    primeras_apariciones = primeras_apariciones.sort_values("_fecha_ts")
    primeras_apariciones["cliente_num_obligaciones"] = (
        primeras_apariciones.groupby("nit_enmascarado").cumcount() + 1
    )
    result = _asof_join(
        result,
        primeras_apariciones,
        by=["nit_enmascarado"],
        left_on="_asof_ts",
        source_date_col="_fecha_ts",
        prefix="",
        value_columns=["cliente_num_obligaciones"],
    )

    cust = datasets["customer"].copy()
    cust["_fecha_ts"] = pd.to_datetime(
        cust["year"].astype(int).astype(str) + cust["month"].astype(int).astype(str).str.zfill(2), format="%Y%m"
    )
    result = _asof_join(
        result,
        cust,
        by=["nit_enmascarado"],
        left_on="_asof_ts",
        source_date_col="_fecha_ts",
        prefix="",  # nombres ya autodescriptivos (edad_cli, segm, etc.), sin colision
        value_columns=[
            "edad_cli",
            "genero_cli",
            "estado_civil",
            "nivel_academico",
            "total_ing",
            "tot_activos",
            "tot_pasivos",
            "segm",
            "subsegm",
            "region_of",
        ],
    )

    return result.drop(columns=["_asof_ts"]).rename(columns={"_t_ts": "fecha_corte"})


def get_feature_columns(features_df: pd.DataFrame) -> list[str]:
    """Retorna las columnas de `features_df` que son predictoras validas (excluye ids/fechas/label)."""
    return [c for c in features_df.columns if c not in NON_FEATURE_COLUMNS]
