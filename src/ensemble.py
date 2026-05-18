from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EnsembleDetector:
    """Majority voting ensemble for binary outlier labels."""

    def majority_vote(self, predictions: list[np.ndarray]) -> np.ndarray:
        if not predictions:
            raise ValueError("Predictions list cannot be empty")

        stacked = np.vstack(predictions)
        votes = stacked.sum(axis=0)
        threshold = len(predictions) / 2.0
        return (votes >= threshold).astype(int)
