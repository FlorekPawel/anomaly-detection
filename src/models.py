from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from pyod.models.pca import PCA
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


@dataclass(frozen=True)
class ModelSpec:
    """Model instance plus its hyperparameter grid."""

    name: str
    family: str
    model: object
    param_grid: dict[str, list[object]]
    factory_key: str = ""


class AnomalyDetectorFactory:
    """Create anomaly detector models with exact parameter grids."""

    _SPECS: ClassVar[dict[str, ModelSpec]] = {
        "isolation_forest": ModelSpec(
            name="IsolationForest",
            family="tree",
            model=IsolationForest(random_state=42),
            param_grid={
                "contamination": [0.01, 0.05, 0.1],
                "n_estimators": [50, 100, 200],
            },
        ),
        "local_outlier_factor": ModelSpec(
            name="LocalOutlierFactor",
            family="density",
            model=LocalOutlierFactor(novelty=True),
            param_grid={"n_neighbors": [10, 20, 50, 100]},
        ),
        "one_class_svm": ModelSpec(
            name="OneClassSVM",
            family="svm",
            model=OneClassSVM(),
            param_grid={
                "nu": [0.01, 0.05, 0.1, 0.2],
                "gamma": ["scale", "auto", 0.1, 0.01],
            },
        ),
        "dbscan": ModelSpec(
            name="DBSCAN",
            family="cluster",
            model=DBSCAN(),
            param_grid={"eps": [0.1, 0.5, 1.0, 2.0], "min_samples": [5, 10, 20]},
        ),
        "pca": ModelSpec(
            name="PCA",
            family="projection",
            model=PCA(),
            param_grid={"n_components": [0.5, 0.7, 0.9, 0.95]},
        ),
        "elliptic_envelope": ModelSpec(
            name="EllipticEnvelope",
            family="statistical",
            model=EllipticEnvelope(random_state=42),
            param_grid={"contamination": [0.01, 0.05, 0.1, 0.2]},
        ),
    }

    @classmethod
    def create(cls, model_name: str) -> ModelSpec:
        """Return model spec for a known model name."""
        key = model_name.strip().lower()
        if key not in cls._SPECS:
            raise ValueError(f"Unknown model name: {model_name}")

        spec = cls._SPECS[key]
        return ModelSpec(
            name=spec.name,
            family=spec.family,
            model=spec.model.__class__(**spec.model.get_params()),
            param_grid={**spec.param_grid},
            factory_key=key,
        )
