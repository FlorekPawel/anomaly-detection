from __future__ import annotations

import pytest
from pyod.models.pca import PCA
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from src.models import AnomalyDetectorFactory


@pytest.mark.parametrize(
    ("name", "expected_type", "expected_grid"),
    [
        (
            "isolation_forest",
            IsolationForest,
            {"contamination": [0.01, 0.05, 0.1], "n_estimators": [50, 100, 200]},
        ),
        (
            "local_outlier_factor",
            LocalOutlierFactor,
            {
                "n_neighbors": [10, 20, 50, 100],
                "contamination": [0.05, 0.1, 0.2],
            },
        ),
        (
            "one_class_svm",
            OneClassSVM,
            {"nu": [0.01, 0.05, 0.1, 0.2], "gamma": ["scale", "auto", 0.1, 0.01]},
        ),
        (
            "dbscan",
            DBSCAN,
            {
                "eps": [0.5, 1.0, 2.0, 4.0, 6.0],
                "min_samples": [5, 10, 20],
            },
        ),
        (
            "pca",
            PCA,
            {"n_components": [0.5, 0.7, 0.9, 0.95]},
        ),
    ],
)
def test_factory_builds_model(name: str, expected_type: type, expected_grid: dict) -> None:
    spec = AnomalyDetectorFactory.create(name)

    assert isinstance(spec.model, expected_type)
    assert spec.param_grid == expected_grid


def test_factory_unknown_model() -> None:
    with pytest.raises(ValueError, match="Unknown model name"):
        AnomalyDetectorFactory.create("unknown")
