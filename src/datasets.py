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
        name="odds_breastw",
        kind="odds",
        data_path="data/odds/breastw.mat",
    ),
    DatasetConfig(
        name="odds_cardio",
        kind="odds",
        data_path="data/odds/cardio.mat",
    ),
    DatasetConfig(
        name="odds_glass",
        kind="odds",
        data_path="data/odds/glass.mat",
    ),
    DatasetConfig(
        name="odds_letter",
        kind="odds",
        data_path="data/odds/letter.mat",
    ),
    DatasetConfig(
        name="odds_lympho",
        kind="odds",
        data_path="data/odds/lympho.mat",
    ),
    DatasetConfig(
        name="odds_satimage",
        kind="odds",
        data_path="data/odds/satimage-2.mat",
    ),
    DatasetConfig(
        name="odds_vowels",
        kind="odds",
        data_path="data/odds/vowels.mat",
    ),
    DatasetConfig(
        name="odds_shuttle",
        kind="odds",
        data_path="data/odds/shuttle.mat",
    ),
    DatasetConfig(
        name="odds_arrhythmia",
        kind="odds",
        data_path="data/odds/arrhythmia.mat",
    ),
    DatasetConfig(
        name="odds_mammography",
        kind="odds",
        data_path="data/odds/mammography.mat",
    ),
    DatasetConfig(
        name="odds_musk",
        kind="odds",
        data_path="data/odds/musk.mat",
    ),
    DatasetConfig(
        name="odds_pendigits",
        kind="odds",
        data_path="data/odds/pendigits.mat",
    ),
    DatasetConfig(
        name="odds_satellite",
        kind="odds",
        data_path="data/odds/satellite.mat",
    ),
    DatasetConfig(
        name="odds_speech",
        kind="odds",
        data_path="data/odds/speech.mat",
    ),
    DatasetConfig(
        name="odds_thyroid",
        kind="odds",
        data_path="data/odds/thyroid.mat",
    ),
    DatasetConfig(
        name="odds_vertebral",
        kind="odds",
        data_path="data/odds/vertebral.mat",
    ),
    DatasetConfig(
        name="odds_wine",
        kind="odds",
        data_path="data/odds/wine.mat",
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
    DatasetConfig(
        name="clustbench_compound",
        kind="clustbench",
        data_path="data/clusterin_benchmarks/compound.data.gz",
        labels_path="data/clusterin_benchmarks/compound.labels3.gz",
        outlier_labels=[0],
    ),
    DatasetConfig(
        name="clustbench_fuzzyx",
        kind="clustbench",
        data_path="data/clusterin_benchmarks/fuzzyx.data.gz",
        labels_path="data/clusterin_benchmarks/fuzzyx.labels4.gz",
        outlier_labels=[0],
    ),
    DatasetConfig(
        name="clustbench_ring_outliers",
        kind="clustbench",
        data_path="data/clusterin_benchmarks/ring_outliers.data.gz",
        labels_path="data/clusterin_benchmarks/ring_outliers.labels1.gz",
        outlier_labels=[0],
    ),
]
