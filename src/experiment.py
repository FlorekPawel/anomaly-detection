from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from src.models import ModelSpec


@dataclass(frozen=True)
class RunSummary:
    """Best run summary for a model family."""

    family: str
    model_name: str
    run_id: str
    metrics: dict[str, float]
    params: dict[str, Any]


def _params_hash(params: dict[str, Any]) -> str:
    payload = json.dumps(params, sort_keys=True, default=str)
    return str(abs(hash(payload)))


def _predict_outliers(model: Any, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "fit_predict"):
        labels = model.fit_predict(features)
        return np.where(labels == -1, 1, 0)

    labels = model.predict(features)
    if labels.dtype.kind in {"i", "b"} and set(np.unique(labels)) <= {0, 1}:
        return labels.astype(int)

    return np.where(labels == -1, 1, 0)


def _decision_scores(model: Any, features: np.ndarray) -> np.ndarray | None:
    if hasattr(model, "decision_function"):
        return model.decision_function(features)
    if hasattr(model, "score_samples"):
        return model.score_samples(features)
    if hasattr(model, "decision_scores_"):
        return model.decision_scores_
    return None


def _compute_metrics(
    labels_true: np.ndarray, labels_pred: np.ndarray, scores: np.ndarray | None
) -> dict[str, float]:
    metrics: dict[str, float] = {
        "f1_score": f1_score(labels_true, labels_pred, zero_division=0),
        "precision": precision_score(labels_true, labels_pred, zero_division=0),
        "recall": recall_score(labels_true, labels_pred, zero_division=0),
        "accuracy": accuracy_score(labels_true, labels_pred),
    }

    if scores is not None:
        try:
            metrics["auc"] = roc_auc_score(labels_true, scores)
        except ValueError:
            metrics["auc"] = math.nan

    return metrics


class ExperimentRunner:
    """Run MLflow experiments with checkpointing."""

    def __init__(
        self,
        experiment_name: str = "anomaly-detection",
        tracking_uri: str | None = None,
        seeds: list[int] | None = None,
    ):
        self.experiment_name = experiment_name
        self.seeds = seeds if seeds is not None else [42, 43, 44]
        os.environ.setdefault("MLFLOW_DISABLE_ENVIRONMENT_INFERENCE", "1")
        for logger_name in ["mlflow", "mlflow.utils", "mlflow.sklearn", "urllib3.connectionpool"]:
            logging.getLogger(logger_name).setLevel(logging.ERROR)
        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(self.experiment_name)

    def already_run(self, dataset: str, model_factory_key: str, params: dict[str, Any]) -> bool:
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            return False

        try:
            params_hash = _params_hash(params)
            # Search by params_hash tag only (safest approach)
            filter_string = f"tags.params_hash = '{params_hash}'"
            runs = mlflow.search_runs([experiment.experiment_id], filter_string=filter_string)
            # Further filter in Python for dataset and model_factory_key
            for _, row in runs.iterrows():
                if (
                    row.get("tags.dataset") == dataset
                    and row.get("tags.model_factory_key") == model_factory_key
                ):
                    return True
            return False
        except Exception:
            # If search fails, assume not run (safe fallback)
            return False

    def evaluate_grid(
        self,
        dataset: str,
        model_spec: ModelSpec,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> list[str]:
        run_ids: list[str] = []
        grid = list(ParameterGrid(model_spec.param_grid))
        supports_seed = "random_state" in model_spec.model.get_params()
        seed_values = self.seeds if supports_seed else [None]

        for params in tqdm(grid, desc=f"{dataset}-{model_spec.name}", leave=False):
            for seed in seed_values:
                run_params = {**params}
                if seed is not None:
                    run_params["random_state"] = seed

                if self.already_run(dataset, model_spec.factory_key, run_params):
                    continue

                model = clone(model_spec.model)
                model.set_params(**run_params)
                model.fit(features)

                predictions = _predict_outliers(model, features)
                scores = _decision_scores(model, features)
                metrics = _compute_metrics(labels, predictions, scores)

                with mlflow.start_run(run_name=f"{dataset}-{model_spec.name}") as run:
                    mlflow.log_params(run_params)
                    mlflow.log_metrics(metrics)
                    mlflow.set_tag("dataset", dataset)
                    mlflow.set_tag("model_factory_key", model_spec.factory_key)
                    mlflow.set_tag("family", model_spec.family)
                    mlflow.set_tag("params_hash", _params_hash(run_params))
                    mlflow.sklearn.log_model(model, name="model")
                    run_ids.append(run.info.run_id)

        return run_ids

    def get_best_models_per_family(self, metric: str = "f1_score") -> list[RunSummary]:
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            return []

        runs = mlflow.search_runs([experiment.experiment_id])
        if runs.empty or f"metrics.{metric}" not in runs.columns:
            return []

        best_rows = (
            runs.sort_values(by=f"metrics.{metric}", ascending=False).groupby("tags.family").head(1)
        )

        summaries: list[RunSummary] = []
        for _, row in best_rows.iterrows():
            params_filtered = {
                k: v for k, v in row.filter(like="params.").to_dict().items() if pd.notna(v)
            }
            params_typed = self._convert_param_types(params_filtered)
            summaries.append(
                RunSummary(
                    family=row["tags.family"],
                    model_name=row["tags.model_factory_key"],
                    run_id=row["run_id"],
                    metrics={metric: row[f"metrics.{metric}"]},
                    params=params_typed,
                )
            )

        return summaries

    def get_best_models_per_model_name(self, metric: str = "f1_score") -> list[RunSummary]:
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            return []

        runs = mlflow.search_runs([experiment.experiment_id])
        if runs.empty or f"metrics.{metric}" not in runs.columns:
            return []

        best_rows = (
            runs.sort_values(by=f"metrics.{metric}", ascending=False)
            .groupby("tags.model_factory_key")
            .head(1)
        )

        summaries: list[RunSummary] = []
        for _, row in best_rows.iterrows():
            params_filtered = {
                k: v for k, v in row.filter(like="params.").to_dict().items() if pd.notna(v)
            }
            params_typed = self._convert_param_types(params_filtered)
            summaries.append(
                RunSummary(
                    family=row["tags.family"],
                    model_name=row["tags.model_factory_key"],
                    run_id=row["run_id"],
                    metrics={metric: row[f"metrics.{metric}"]},
                    params=params_typed,
                )
            )

        return summaries

    @staticmethod
    def _convert_param_types(params: dict[str, Any]) -> dict[str, Any]:
        """Convert string params from MLflow to correct types."""
        converted = {}
        for key, value in params.items():
            if value is None:
                continue
            if isinstance(value, str):
                try:
                    converted[key] = int(value)
                except ValueError:
                    try:
                        converted[key] = float(value)
                    except ValueError:
                        converted[key] = value
            else:
                converted[key] = value
        return converted
