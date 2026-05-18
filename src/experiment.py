from __future__ import annotations

import hashlib
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
from sklearn.exceptions import NotFittedError
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
    """Return a stable hash of a params dict (survives process restarts)."""
    payload = json.dumps(params, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalize_labels(labels: np.ndarray) -> np.ndarray:
    """Map outlier-detector outputs to the unified {0=inlier, 1=outlier} convention.

    - sklearn detectors (IsolationForest, OneClassSVM, LocalOutlierFactor): -1 = outlier
    - DBSCAN: -1 = noise (outlier), >=0 = cluster member (inlier)
    - pyod / already-binary: leave as-is
    """
    arr = np.asarray(labels)
    if arr.dtype.kind in {"i", "b"} and set(np.unique(arr).tolist()) <= {0, 1}:
        return arr.astype(int)
    return np.where(arr == -1, 1, 0).astype(int)


def _fit_and_predict_raw(model: Any, features: np.ndarray) -> np.ndarray:
    """Fit ``model`` on ``features`` and return the raw label array.

    Avoids ``fit_predict`` on pyod detectors (it inherits the deprecated sklearn
    ``OutlierMixin.fit_predict``); pyod exposes ``predict`` after ``fit`` directly.
    For sklearn detectors and DBSCAN, ``fit_predict`` is the canonical API.
    """
    cls_module = getattr(type(model), "__module__", "") or ""
    if cls_module.startswith("pyod"):
        model.fit(features)
        return np.asarray(model.predict(features))
    if hasattr(model, "fit_predict"):
        return np.asarray(model.fit_predict(features))
    model.fit(features)
    return np.asarray(model.predict(features))


def _predict_outliers(model: Any, features: np.ndarray) -> np.ndarray:
    """Get binary outlier predictions from a fitted model."""
    if hasattr(model, "labels_"):
        return _normalize_labels(model.labels_)
    if hasattr(model, "predict"):
        try:
            return _normalize_labels(model.predict(features))
        except (AttributeError, NotFittedError, ValueError):
            pass
    return _normalize_labels(model.fit_predict(features))


def _outlier_scores(model: Any, features: np.ndarray) -> np.ndarray | None:
    """Return scores oriented so that *higher = more anomalous*.

    sklearn outlier detectors return higher = more *normal*; we negate them.
    pyod detectors already return higher = more *anomalous*; we pass them through.
    """
    cls_module = getattr(type(model), "__module__", "") or ""
    is_pyod = cls_module.startswith("pyod")

    if hasattr(model, "decision_function"):
        try:
            scores = np.asarray(model.decision_function(features), dtype=float)
            return scores if is_pyod else -scores
        except (AttributeError, NotFittedError, ValueError):
            pass
    if hasattr(model, "decision_scores_"):
        return np.asarray(model.decision_scores_, dtype=float)
    if hasattr(model, "score_samples"):
        try:
            return -np.asarray(model.score_samples(features), dtype=float)
        except (AttributeError, NotFittedError, ValueError):
            pass
    if hasattr(model, "negative_outlier_factor_"):
        nof = np.asarray(model.negative_outlier_factor_, dtype=float)
        if len(nof) == len(features):
            return -nof
    return None


def _decision_scores(model: Any, features: np.ndarray) -> np.ndarray | None:
    """Backward-compatible alias for ``_outlier_scores``."""
    return _outlier_scores(model, features)


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
        self.seeds = seeds if seeds is not None else [0, 42, 3407]
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

                raw_predictions = _fit_and_predict_raw(model, features)
                predictions = _normalize_labels(raw_predictions)
                scores = _outlier_scores(model, features)
                metrics = _compute_metrics(labels, predictions, scores)

                with mlflow.start_run(run_name=f"{dataset}-{model_spec.name}") as run:
                    mlflow.log_params(run_params)
                    mlflow.log_metrics(metrics)
                    mlflow.set_tag("dataset", dataset)
                    mlflow.set_tag("model_factory_key", model_spec.factory_key)
                    mlflow.set_tag("family", model_spec.family)
                    mlflow.set_tag("params_hash", _params_hash(run_params))
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

    def get_aggregate_best_per_model_on_subset(
        self,
        metric: str = "f1_score",
        dataset_prefix: str | None = "odds_",
        seed_param: str = "params.random_state",
    ) -> list[RunSummary]:
        """Pick, per model family, the hyperparameter configuration with the best *mean* metric.

        Selection averages the metric across all matching datasets and seeds (the seed
        parameter is excluded from the params key, so it does not split groups).

        Parameters
        ----------
        metric:
            Metric name without the ``metrics.`` prefix (default ``"f1_score"``).
        dataset_prefix:
            Restrict aggregation to runs whose ``tags.dataset`` starts with this prefix
            (default ``"odds_"`` — real-world tabular benchmarks closest in shape to a
            typical verification set). Pass ``None`` to aggregate over every dataset.
            If the filter would discard all runs, the full set is used as a fallback.
        seed_param:
            Name of the seed parameter column to exclude from the params identity.
        """
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            return []

        runs = mlflow.search_runs([experiment.experiment_id])
        metric_col = f"metrics.{metric}"
        if runs.empty or metric_col not in runs.columns:
            return []

        df = runs.copy()
        if dataset_prefix is not None and "tags.dataset" in df.columns:
            filtered = df[df["tags.dataset"].astype(str).str.startswith(dataset_prefix)]
            if not filtered.empty:
                df = filtered

        df = df.dropna(subset=[metric_col])
        if df.empty:
            return []

        param_cols = [c for c in df.columns if c.startswith("params.") and c != seed_param]
        df["_params_key"] = (
            df[param_cols].astype(str).fillna("").agg("|".join, axis=1) if param_cols else ""
        )

        agg = df.groupby(["tags.model_factory_key", "_params_key"], as_index=False).agg(
            mean_metric=(metric_col, "mean"),
            n_runs=(metric_col, "size"),
            n_datasets=("tags.dataset", pd.Series.nunique),
        )
        best = (
            agg.sort_values("mean_metric", ascending=False)
            .groupby("tags.model_factory_key", as_index=False)
            .head(1)
        )

        summaries: list[RunSummary] = []
        for _, row in best.iterrows():
            sample = df[
                (df["tags.model_factory_key"] == row["tags.model_factory_key"])
                & (df["_params_key"] == row["_params_key"])
            ].iloc[0]
            params_filtered = {
                k: v
                for k, v in sample.filter(like="params.").to_dict().items()
                if pd.notna(v) and k != seed_param
            }
            params_typed = self._convert_param_types(params_filtered)
            summaries.append(
                RunSummary(
                    family=sample.get("tags.family", ""),
                    model_name=sample["tags.model_factory_key"],
                    run_id=sample["run_id"],
                    metrics={metric: float(row["mean_metric"])},
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
