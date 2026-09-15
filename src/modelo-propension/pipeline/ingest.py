"""Data Platform: ingesta y validacion de datos crudos (RF-04, RF-05).

Bancolombia entrega 5 fuentes (ver Metadata.xlsx, revisado y documentado en
la memoria del repo):
    trtest       -> label var_rpta_alt + atributos del mes de la variable
                    respuesta (usar con cuidado: ver pipeline/features.py,
                    la mayoria de sus columnas NO son seguras como predictoras)
    oot          -> obligaciones a calificar (sin label), enero 2024
    probabilidad -> scores de 3 modelos existentes del banco (propension,
                    alerta temprana, auto cura) por obligacion-mes
    pagos        -> comportamiento de pago historico por obligacion-mes
    customer     -> demograficos por cliente-mes (solo cubre jul-dic 2023)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Columnas que identifican una obligacion en trtest/oot; se concatenan con
# "#" para formar `id_obligacion`, igual formato que el archivo de sumision.
ID_COMPONENT_COLUMNS = ["nit_enmascarado", "num_oblig_orig_enmascarado", "num_oblig_enmascarado"]

# Columnas minimas esperadas por fuente (quality gate previo al feature
# engineering, RF-05). No es el esquema completo, solo lo indispensable.
REQUIRED_COLUMNS: dict[str, set[str]] = {
    "trtest": {*ID_COMPONENT_COLUMNS, "fecha_var_rpta_alt", "var_rpta_alt"},
    "oot": {*ID_COMPONENT_COLUMNS, "fecha_var_rpta_alt"},
    "probabilidad": {
        "nit_enmascarado",
        "num_oblig_enmascarado",
        "fecha_corte",
        "prob_propension",
        "prob_alrt_temprana",
        "prob_auto_cura",
        "lote",
    },
    "pagos": {"nit_enmascarado", "num_oblig_enmascarado", "fecha_corte", "porc_pago", "marca_pago"},
    "customer": {"nit_enmascarado", "year", "month"},
}


def _build_id_obligacion(df: pd.DataFrame) -> pd.Series:
    """Concatena nit#num_oblig_orig#num_oblig, igual formato que sample_submission.csv."""
    return (
        df["nit_enmascarado"].astype(str)
        + "#"
        + df["num_oblig_orig_enmascarado"].astype(str)
        + "#"
        + df["num_oblig_enmascarado"].astype(str)
    )


def load_raw_data(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Carga las 5 fuentes crudas entregadas por Bancolombia (RF-04).

    Usa exclusivamente los archivos definidos en `config["data"]` (sin
    fuentes externas), y construye la columna `id_obligacion` en trtest/oot.

    Args:
        config: configuracion cargada (ver config/pipeline_config.yaml,
            seccion `data`: raw_dir + *_file por cada fuente).

    Returns:
        Diccionario {"trtest", "oot", "probabilidad", "pagos", "customer"} -> DataFrame.
    """
    raw_dir = Path(config["data"]["raw_dir"])
    files = {
        "trtest": config["data"]["trtest_file"],
        "oot": config["data"]["oot_file"],
        "probabilidad": config["data"]["probabilidad_file"],
        "pagos": config["data"]["pagos_file"],
        "customer": config["data"]["customer_file"],
    }
    datasets = {name: pd.read_csv(raw_dir / filename) for name, filename in files.items()}

    for name in ("trtest", "oot"):
        datasets[name] = datasets[name].assign(id_obligacion=_build_id_obligacion(datasets[name]))

    return datasets


def validate_schema(datasets: dict[str, pd.DataFrame]) -> None:
    """Valida que cada fuente tenga, como minimo, las columnas de REQUIRED_COLUMNS (RF-05).

    Actua como quality gate previo al feature engineering.

    Raises:
        ValueError: descriptivo (fuente + columnas faltantes) si algo no cuadra.
    """
    errores: list[str] = []
    for name, required in REQUIRED_COLUMNS.items():
        if name not in datasets:
            errores.append(f"falta la fuente '{name}'")
            continue
        faltantes = required - set(datasets[name].columns)
        if faltantes:
            errores.append(f"'{name}' no tiene las columnas: {sorted(faltantes)}")
    if errores:
        raise ValueError("Esquema invalido en los datos crudos: " + "; ".join(errores))


def clean_raw_data(datasets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Limpia problemas de calidad conocidos en los datos crudos (RF-05).

    Hoy corrige: valores `inf`/`-inf` en columnas numericas (encontrado en
    pagos.porc_pago, por divisiones cerca de cero aguas arriba) -- XGBoost y
    la mayoria de librerias de ML no aceptan inf, solo NaN, asi que se trata
    como faltante en vez de inventar un tope/clipping arbitrario.

    Se aplica antes de features.build_features (entre validate_schema y el
    feature engineering) para que train, infer_batch y monitor compartan la
    misma limpieza, en vez de que cada consumidor la reimplemente por su cuenta.

    Returns:
        Copia de `datasets` con las columnas numericas saneadas.
    """
    cleaned = {}
    for name, df in datasets.items():
        df = df.copy()
        numeric_cols = df.select_dtypes(include="number").columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        cleaned[name] = df
    return cleaned

