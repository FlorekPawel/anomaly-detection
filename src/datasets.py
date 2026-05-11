from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    kind: str
    data_path: str
    labels_path: str | None = None
    outlier_labels: list[int] | None = None


ODDS_DATASETS: list[DatasetConfig] = [
    DatasetConfig(
        name="odds_annthyroid",
        kind="odds",
        data_path="data/odds/annthyroid.mat",
    ),
    DatasetConfig(
        name="odds_ionosphere",
        kind="odds",
        data_path="data/odds/ionosphere.mat",
    ),
    DatasetConfig(
        name="odds_breastw",
        kind="odds",
        data_path="data/odds/breastw.mat",
    ),
]

CLUSTBENCH_DATASETS: list[DatasetConfig] = [
    DatasetConfig(
        name="clustbench_ring_noisy",
        kind="clustbench",
        data_path="data/clusterin_benchmarks/ring_noisy.data.gz",
        labels_path="data/clusterin_benchmarks/ring_noisy.labels0.gz",
        outlier_labels=[0],
    ),
    DatasetConfig(
        name="clustbench_zigzag_noisy",
        kind="clustbench",
        data_path="data/clusterin_benchmarks/zigzag_noisy.data.gz",
        labels_path="data/clusterin_benchmarks/zigzag_noisy.labels0.gz",
        outlier_labels=[0],
    ),
    DatasetConfig(
        name="clustbench_hdbscan",
        kind="clustbench",
        data_path="data/clusterin_benchmarks/hdbscan.data.gz",
        labels_path="data/clusterin_benchmarks/hdbscan.labels0.gz",
        outlier_labels=[0],
    ),
]
