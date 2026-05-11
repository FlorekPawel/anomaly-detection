from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data_loader import DataLoader
from src.datasets import CLUSTBENCH_DATASETS, ODDS_DATASETS
from src.ensemble import EnsembleDetector
from src.experiment import ExperimentRunner
from src.models import AnomalyDetectorFactory


def _normalize_mlflow_labels(preds: np.ndarray) -> np.ndarray:
    return np.where(preds == -1, 1, preds).astype(int)


def main() -> None:
    data_loader = DataLoader()
    experiment = ExperimentRunner()

    model_names = [
        "isolation_forest",
        "local_outlier_factor",
        "one_class_svm",
        "dbscan",
        "pca",
        "elliptic_envelope",
    ]

    for dataset in tqdm(ODDS_DATASETS, desc="ODDS datasets"):
        loaded = data_loader.load_odds_mat(dataset.data_path)
        standardized = data_loader.standardize(loaded.features, loaded.labels)
        if standardized.labels is None:
            continue

        for name in tqdm(model_names, desc=f"{dataset.name} models", leave=False):
            spec = AnomalyDetectorFactory.create(name)
            experiment.evaluate_grid(
                dataset=dataset.name,
                model_spec=spec,
                features=standardized.features,
                labels=standardized.labels,
            )

    for dataset in tqdm(CLUSTBENCH_DATASETS, desc="Clustbench datasets"):
        loaded = data_loader.load_clustbench_files(
            dataset.data_path,
            dataset.labels_path,
            outlier_labels=dataset.outlier_labels,
        )
        standardized = data_loader.standardize(loaded.features, loaded.labels)
        if standardized.labels is None:
            continue

        for name in tqdm(model_names, desc=f"{dataset.name} models", leave=False):
            spec = AnomalyDetectorFactory.create(name)
            experiment.evaluate_grid(
                dataset=dataset.name,
                model_spec=spec,
                features=standardized.features,
                labels=standardized.labels,
            )

    dataset_path = Path("data/test_data.csv")
    verification = data_loader.load_verification_csv(str(dataset_path))
    standardized = data_loader.standardize(verification.features)

    winners = experiment.get_best_models_per_model_name(metric="f1_score")
    models = []
    for summary in winners:
        model_factory_key = summary.model_name
        spec = AnomalyDetectorFactory.create(model_factory_key)
        model = spec.model
        params = {k.replace("params.", ""): v for k, v in summary.params.items()}
        model.set_params(**params)
        model.fit(standardized.features)
        models.append(model)

    predictions = []
    for _, model in enumerate(models):
        if hasattr(model, "predict"):
            preds = model.predict(standardized.features)
        else:
            preds = model.fit_predict(standardized.features)
        predictions.append(_normalize_mlflow_labels(preds))

    ensemble = EnsembleDetector()
    final_labels = ensemble.majority_vote(predictions)

    output = pd.DataFrame({"class": final_labels})
    output.to_csv("test_labels.csv", index=False)


if __name__ == "__main__":
    main()
