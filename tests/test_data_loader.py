from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.data_loader import DataLoader, LoadedData


def test_load_odds_mat_happy_path() -> None:
    loader = DataLoader()
    fake_mat = {"X": np.array([[1.0, 2.0], [3.0, 4.0]]), "y": np.array([[0], [1]])}

    with patch("src.data_loader.loadmat", return_value=fake_mat) as mocked:
        loaded = loader.load_odds_mat("fake.mat")

    mocked.assert_called_once_with("fake.mat")
    assert isinstance(loaded, LoadedData)
    assert loaded.features.shape == (2, 2)
    assert loaded.labels is not None
    assert loaded.labels.tolist() == [0, 1]


def test_load_odds_mat_missing_keys() -> None:
    loader = DataLoader()
    fake_mat = {"X": np.array([[1.0, 2.0]])}

    with (
        patch("src.data_loader.loadmat", return_value=fake_mat),
        pytest.raises(KeyError, match="'X' and 'y'"),
    ):
        loader.load_odds_mat("fake.mat")


def test_load_clustbench_csv_with_labels() -> None:
    loader = DataLoader()
    data_df = pd.DataFrame([[1, 2], [3, 4]])
    labels_df = pd.DataFrame([0, 1])

    with patch("src.data_loader.pd.read_csv", side_effect=[data_df, labels_df]) as mocked:
        loaded = loader.load_clustbench_files("data.gz", "labels.gz")

    assert mocked.call_count == 2
    assert loaded.features.shape == (2, 2)
    assert loaded.labels is not None
    assert loaded.labels.tolist() == [0, 1]


def test_load_clustbench_without_labels() -> None:
    loader = DataLoader()
    df = pd.DataFrame([[1, 2], [3, 4]])

    with patch("src.data_loader.pd.read_csv", return_value=df):
        loaded = loader.load_clustbench_files("data.gz")

    assert loaded.features.shape == (2, 2)
    assert loaded.labels is None


def test_load_clustbench_maps_outliers() -> None:
    loader = DataLoader()
    data_df = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
    labels_df = pd.DataFrame([0, 1, 0])

    with patch("src.data_loader.pd.read_csv", side_effect=[data_df, labels_df]):
        loaded = loader.load_clustbench_files(
            "data.gz",
            "labels.gz",
            outlier_labels=[0],
        )

    assert loaded.labels is not None
    assert loaded.labels.tolist() == [1, 0, 1]


def test_load_verification_csv() -> None:
    loader = DataLoader()
    df = pd.DataFrame({"f1": [1, 2], "f2": [3, 4]})

    with patch("src.data_loader.pd.read_csv", return_value=df) as mocked:
        loaded = loader.load_verification_csv("verify.csv")

    mocked.assert_called_once_with("verify.csv")
    assert loaded.features.shape == (2, 2)
    assert loaded.labels is None


def test_standardize_does_not_touch_labels() -> None:
    loader = DataLoader()
    features = np.array([[1.0, 2.0], [3.0, 4.0]])
    labels = np.array([0, 1])

    loaded = loader.standardize(features, labels)

    assert loaded.features.shape == (2, 2)
    assert loaded.labels is not None
    assert np.array_equal(loaded.labels, labels)
    assert not np.array_equal(loaded.features, features)
