"""Pruebas de pipeline.split con datos sinteticos pequenos (RNF-14)."""
import pandas as pd
import pytest

from pipeline.split import temporal_split


def _make_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id_obligacion": [f"id{i}" for i in range(5)],
            "fecha_corte": pd.to_datetime(
                ["2023-07-01", "2023-08-01", "2023-09-01", "2023-10-01", "2023-11-01"]
            ),
        }
    )


def test_temporal_split_particiona_cronologicamente() -> None:
    train_df, val_df, oot_df = temporal_split(
        _make_df(), date_column="fecha_corte", train_end_month="2023-09", val_end_month="2023-10"
    )
    assert list(train_df["id_obligacion"]) == ["id0", "id1", "id2"]  # jul, ago, sep
    assert list(val_df["id_obligacion"]) == ["id3"]  # oct
    assert list(oot_df["id_obligacion"]) == ["id4"]  # nov


def test_temporal_split_falla_si_oot_queda_vacia() -> None:
    with pytest.raises(ValueError, match="OOT"):
        temporal_split(_make_df(), date_column="fecha_corte", train_end_month="2023-10", val_end_month="2023-11")


def test_temporal_split_falla_si_train_queda_vacia() -> None:
    with pytest.raises(ValueError, match="train"):
        temporal_split(_make_df(), date_column="fecha_corte", train_end_month="2023-01", val_end_month="2023-08")
