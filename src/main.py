from __future__ import annotations

from tqdm import tqdm

from src.data_loader import DataLoader
from src.datasets import CLUSTBENCH_DATASETS, ODDS_DATASETS
from src.experiment import ExperimentRunner
from src.models import AnomalyDetectorFactory


def main() -> None:
    data_loader = DataLoader()
    experiment = ExperimentRunner()

    model_names = [
        "isolation_forest",
        "local_outlier_factor",
        "one_class_svm",
        "dbscan",
        "pca",
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


if __name__ == "__main__":
    main()
