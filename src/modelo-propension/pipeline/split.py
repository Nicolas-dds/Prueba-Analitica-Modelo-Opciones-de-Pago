"""Particion temporal train / validacion / out-of-time (RF-06)."""
from __future__ import annotations

import pandas as pd


def temporal_split(
    df: pd.DataFrame,
    date_column: str,
    train_end_month: str,
    val_end_month: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Particiona `df` en train / validacion / out-of-time (OOT) por fecha.

    La particion es estrictamente cronologica (nunca aleatoria) para
    respetar RF-03/RF-06: train = `date_column` <= train_end_month;
    validacion = train_end_month < `date_column` <= val_end_month; OOT =
    `date_column` > val_end_month (simula el mes que se calificara en
    produccion; no debe usarse para entrenar ni tunear hiperparametros).

    Nota sobre fuga por cliente: `nit_enmascarado`/`num_oblig_enmascarado`
    nunca son features (ver pipeline.features.NON_FEATURE_COLUMNS), asi que
    un mismo cliente puede tener obligaciones-mes en mas de un conjunto sin
    que eso sea fuga -- el corte cronologico ya garantiza que ninguna fila de
    validacion/OOT existia (con su fecha_corte real) durante el entrenamiento.

    Args:
        df: dataset con features ya construidas (ver features.build_features).
        date_column: columna de fecha usada para particionar (ej. "fecha_corte").
        train_end_month: ultimo mes incluido en entrenamiento ("YYYY-MM").
        val_end_month: ultimo mes incluido en validacion ("YYYY-MM", posterior a train_end_month).

    Returns:
        Tupla (train_df, val_df, oot_df).

    Raises:
        ValueError: si alguna de las 3 particiones queda vacia (revisar los
            meses de corte contra el rango real de `date_column`).
    """
    periods = df[date_column].dt.to_period("M")
    train_end = pd.Period(train_end_month, freq="M")
    val_end = pd.Period(val_end_month, freq="M")

    train_df = df.loc[periods <= train_end].copy()
    val_df = df.loc[(periods > train_end) & (periods <= val_end)].copy()
    oot_df = df.loc[periods > val_end].copy()

    for name, split_df in (("train", train_df), ("validacion", val_df), ("OOT", oot_df)):
        if split_df.empty:
            raise ValueError(
                f"La particion '{name}' quedo vacia. Revisa train_end_month="
                f"'{train_end_month}' / val_end_month='{val_end_month}' contra el "
                f"rango real de '{date_column}' ({periods.min()} a {periods.max()})."
            )

    return train_df, val_df, oot_df
