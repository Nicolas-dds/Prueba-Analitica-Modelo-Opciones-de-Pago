"""Genera el archivo de sumision real para la OOT (RF-07, formato ID/var_rpta_alt).

Reentrena el modelo campeon vigente sobre TODOS los meses disponibles de
`trtest` (sin separar train/val/OOT), usando los mismos hiperparametros y
umbral ya validados (pipeline.registry.get_champion) -- una vez validada la
generalizacion con el split, se aprovecha el 100% de la informacion
etiquetada antes de calificar la muestra fuera de tiempo real (RNF-06:
trazable, se sabe exactamente que version/hiperparametros la generaron).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config_loader import load_config
from pipeline import features, ingest, registry, train
from pipeline.train import prepare_features_for_model


def generate_submission(
    config_path: str | Path,
    train_reference_month: str,
    oot_month: str,
    output_path: str | Path,
) -> pd.DataFrame:
    """Reentrena con todo `trtest` y califica la OOT real, en formato de sumision.

    Args:
        config_path: ruta a pipeline_config.yaml.
        train_reference_month: mes de corte que incluye TODOS los meses
            disponibles de trtest (ej. "2023-12" incluye jul-nov como t).
        oot_month: mes de la OOT real a calificar (ej. "2024-01").
        output_path: ruta del CSV de sumision a generar.

    Returns:
        DataFrame con columnas ID (nit#num_oblig_orig#num_oblig) y
        var_rpta_alt (0/1), una fila por obligacion de la OOT -- mismo
        formato que sample_submission.csv.

    Raises:
        registry.NoChampionRegisteredError: si no hay ningun modelo
            campeon (correr primero orchestration.run_pipeline).
    """
    config = load_config(config_path)

    champion_metadata = registry.get_champion(config["registry"]["path"])
    hiperparametros = champion_metadata["hiperparametros"]
    threshold = champion_metadata["metricas"]["f1_threshold"]

    datasets = ingest.load_raw_data(config)
    ingest.validate_schema(datasets)
    datasets = ingest.clean_raw_data(datasets)

    # Refit final: TODO trtest disponible, sin held-out -- la generalizacion
    # ya se valido en el pipeline de desarrollo (run_pipeline.py); aqui se
    # maximiza el uso de datos etiquetados para la calificacion real.
    train_final_df = features.build_features(
        datasets, reference_month=train_reference_month, config=config, population="trtest"
    )
    feature_columns = features.get_feature_columns(train_final_df)
    model_final = train.train_model(
        train_final_df,
        feature_columns,
        config["data"]["label_column"],
        hiperparametros,
        config["model"]["random_seed"],
    )

    oot_features = features.build_features(datasets, reference_month=oot_month, config=config, population="oot")
    X_oot, _ = prepare_features_for_model(oot_features, feature_columns, category_maps=model_final.category_maps_)
    scores = model_final.predict_proba(X_oot)[:, 1]
    predicciones = (scores >= threshold).astype(int)

    submission = pd.DataFrame({"ID": oot_features["id_obligacion"], "var_rpta_alt": predicciones})
    submission.to_csv(output_path, index=False)
    return submission


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Genera el archivo de sumision real (ID/var_rpta_alt).")
    parser.add_argument("--config", default=None, help="Ruta al archivo pipeline_config.yaml")
    parser.add_argument(
        "--train-reference-month",
        required=True,
        help="Mes que incluye TODO trtest disponible para el refit final, formato YYYY-MM",
    )
    parser.add_argument("--oot-month", required=True, help="Mes de la OOT real a calificar, formato YYYY-MM")
    parser.add_argument("--output", required=True, help="Ruta del CSV de sumision a generar")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    config_path = args.config or (Path(__file__).parent.parent / "config" / "pipeline_config.yaml")
    submission = generate_submission(config_path, args.train_reference_month, args.oot_month, args.output)
    print(f"Sumision generada: {args.output} ({len(submission)} filas)")
    print(submission["var_rpta_alt"].value_counts(normalize=True))
