from __future__ import annotations

import numpy as np
import pytest

from src.ensemble import EnsembleDetector


def test_majority_vote() -> None:
    ensemble = EnsembleDetector()
    preds = [
        np.array([0, 1, 0, 1]),
        np.array([0, 1, 1, 1]),
        np.array([1, 0, 0, 1]),
    ]

    result = ensemble.majority_vote(preds)

    assert result.tolist() == [0, 1, 0, 1]


def test_majority_vote_empty() -> None:
    ensemble = EnsembleDetector()

    with pytest.raises(ValueError, match="Predictions list cannot be empty"):
        ensemble.majority_vote([])
