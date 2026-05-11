from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class LoadedData:
    """Container for feature matrix and optional labels."""

    features: np.ndarray
    labels: np.ndarray | None


class DataLoader:
    """Load and standardize data for anomaly detection."""

    def load_odds_mat(self, file_path: str) -> LoadedData:
        """Load ODDS .mat file with 'X' features and 'y' labels."""
        data = loadmat(file_path)
        if "X" not in data or "y" not in data:
            raise KeyError("ODDS file must contain 'X' and 'y' keys")

        features = np.asarray(data["X"])
        labels = np.asarray(data["y"]).ravel()
        return LoadedData(features=features, labels=labels)

    def load_clustbench_files(
        self,
        data_path: str,
        labels_path: str | None = None,
        *,
        outlier_labels: Iterable[int] | None = None,
    ) -> LoadedData:
        """Load clustbench data/labels files (supports .gz)."""
        data = pd.read_csv(data_path, sep=r"\s+", header=None, compression="infer")
        features = data.to_numpy()
        labels: np.ndarray | None = None

        if labels_path is not None:
            raw_labels = pd.read_csv(labels_path, header=None, compression="infer").iloc[:, 0]
            labels = raw_labels.to_numpy()
            if outlier_labels is not None:
                outlier_set = set(outlier_labels)
                labels = np.array([1 if value in outlier_set else 0 for value in labels])

        return LoadedData(features=features, labels=labels)

    def load_verification_csv(self, file_path: str) -> LoadedData:
        """Load verification dataset with features only."""
        df = pd.read_csv(file_path)
        features = df.to_numpy()
        return LoadedData(features=features, labels=None)

    def standardize(self, features: np.ndarray, labels: np.ndarray | None = None) -> LoadedData:
        """Standardize features without fitting on labels."""
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)
        return LoadedData(features=scaled, labels=labels)
