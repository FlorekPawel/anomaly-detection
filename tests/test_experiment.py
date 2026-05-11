from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.experiment import ExperimentRunner, RunSummary
from src.models import ModelSpec


@dataclass
class FakeRun:
    run_id: str = "run-1"


@dataclass
class FakeRunInfo:
    run_id: str = "run-1"


@contextmanager
def fake_start_run(**_: Any) -> Any:
    class FakeContext:
        info = FakeRunInfo()

    yield FakeContext()


def test_already_run_false_when_no_experiment() -> None:
    with patch("src.experiment.mlflow.get_experiment_by_name", return_value=None):
        runner = ExperimentRunner(experiment_name="exp")
        assert not runner.already_run("ds", "model", {"a": 1})


def test_already_run_true_with_match() -> None:
    fake_exp = MagicMock(experiment_id="1")
    fake_df = pd.DataFrame(
        {
            "run_id": ["r1"],
            "tags.dataset": ["ds"],
            "tags.model_factory_key": ["model"],
        }
    )

    with (
        patch("src.experiment.mlflow.get_experiment_by_name", return_value=fake_exp),
        patch("src.experiment.mlflow.search_runs", return_value=fake_df),
    ):
        runner = ExperimentRunner(experiment_name="exp")
        assert runner.already_run("ds", "model", {"a": 1})


def test_evaluate_grid_logs_runs() -> None:
    features = np.array([[0.0], [1.0], [2.0]])
    labels = np.array([0, 1, 0])

    class FakeModel:
        def __init__(self) -> None:
            self.params: dict[str, Any] = {}

        def get_params(self, deep: bool = True) -> dict[str, Any]:  # noqa: ARG002
            return {}

        def set_params(self, **params: Any) -> FakeModel:
            self.params.update(params)
            return self

        def fit(self, x: np.ndarray) -> FakeModel:  # noqa: ARG002
            return self

        def predict(self, x: np.ndarray) -> np.ndarray:  # noqa: ARG002
            return np.array([1, -1, 1])

        def decision_function(self, x: np.ndarray) -> np.ndarray:  # noqa: ARG002
            return np.array([0.1, 0.9, 0.2])

    spec = ModelSpec(
        name="FakeModel",
        family="fake",
        model=FakeModel(),
        param_grid={"alpha": [1, 2]},
    )

    with (
        patch("src.experiment.mlflow.start_run", side_effect=fake_start_run),
        patch(
            "src.experiment.mlflow.get_experiment_by_name",
            return_value=MagicMock(experiment_id="1"),
        ),
        patch("src.experiment.mlflow.search_runs", return_value=pd.DataFrame()),
        patch("src.experiment.mlflow.log_params") as log_params,
        patch("src.experiment.mlflow.log_metrics") as log_metrics,
        patch("src.experiment.mlflow.set_tag"),
        patch("src.experiment.mlflow.sklearn.log_model"),
    ):
        runner = ExperimentRunner(experiment_name="exp")
        run_ids = runner.evaluate_grid("ds", spec, features, labels)

    assert len(run_ids) == 2
    assert log_params.call_count == 2
    assert log_metrics.call_count == 2


def test_get_best_models_per_family() -> None:
    fake_exp = MagicMock(experiment_id="1")
    runs = pd.DataFrame(
        {
            "run_id": ["r1", "r2"],
            "tags.family": ["svm", "svm"],
            "tags.model_factory_key": ["one_class_svm", "one_class_svm"],
            "metrics.f1_score": [0.2, 0.8],
            "params.nu": [0.1, 0.2],
        }
    )

    with (
        patch("src.experiment.mlflow.get_experiment_by_name", return_value=fake_exp),
        patch("src.experiment.mlflow.search_runs", return_value=runs),
    ):
        runner = ExperimentRunner(experiment_name="exp")
        best = runner.get_best_models_per_family()

    assert best == [
        RunSummary(
            family="svm",
            model_name="one_class_svm",
            run_id="r2",
            metrics={"f1_score": 0.8},
            params={"params.nu": 0.2},
        )
    ]


def test_evaluate_grid_expands_seeds_for_random_state() -> None:
    features = np.array([[0.0], [1.0], [2.0]])
    labels = np.array([0, 1, 0])

    class SeededModel:
        def __init__(self, random_state: int = 0) -> None:
            self.params: dict[str, Any] = {"random_state": random_state}

        def get_params(self, deep: bool = True) -> dict[str, Any]:  # noqa: ARG002
            return {"random_state": self.params["random_state"]}

        def set_params(self, **params: Any) -> SeededModel:
            self.params.update(params)
            return self

        def fit(self, x: np.ndarray) -> SeededModel:  # noqa: ARG002
            return self

        def predict(self, x: np.ndarray) -> np.ndarray:  # noqa: ARG002
            return np.array([1, -1, 1])

    spec = ModelSpec(
        name="Seeded",
        family="seed",
        model=SeededModel(),
        param_grid={"alpha": [1]},
    )

    with (
        patch("src.experiment.mlflow.start_run", side_effect=fake_start_run),
        patch(
            "src.experiment.mlflow.get_experiment_by_name",
            return_value=MagicMock(experiment_id="1"),
        ),
        patch("src.experiment.mlflow.search_runs", return_value=pd.DataFrame()),
        patch("src.experiment.mlflow.log_params") as log_params,
        patch("src.experiment.mlflow.log_metrics") as log_metrics,
        patch("src.experiment.mlflow.set_tag"),
        patch("src.experiment.mlflow.sklearn.log_model"),
    ):
        runner = ExperimentRunner(experiment_name="exp", seeds=[1, 2, 3])
        run_ids = runner.evaluate_grid("ds", spec, features, labels)

    assert len(run_ids) == 3
    assert log_params.call_count == 3
    assert log_metrics.call_count == 3
