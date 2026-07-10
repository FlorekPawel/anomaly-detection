# Unsupervised Anomaly Detection

Comparative study of unsupervised anomaly detection methods on tabular and synthetic benchmarks. The pipeline runs hyperparameter grid search across multiple datasets, logs results to MLflow, and supports ensemble prediction on an unlabeled verification set.

## Models

| Model | Family | Library |
| --- | --- | --- |
| Isolation Forest | tree | scikit-learn |
| Local Outlier Factor | density | scikit-learn |
| One-Class SVM | svm | scikit-learn |
| DBSCAN | cluster | scikit-learn |
| PCA | decomposition | PyOD |

Each model is evaluated over a fixed hyperparameter grid (see `src/models.py`).

## Datasets

- **ODDS** — 18 real-world tabular datasets from the [Outlier Detection DataSets](https://odds.cs.stonybrook.edu/) repository (`.mat` files under `data/odds/`).
- **Clustbench** — 6 synthetic clustering benchmarks with injected outliers (`.data.gz` / `.labels*.gz` under `data/clusterin_benchmarks/`).
- **Verification** — unlabeled `data/test_data.csv` used for ensemble prediction in the analysis notebook.

Dataset paths are configured in `src/datasets.py`. The `data/` directory is gitignored; place files locally before running experiments.

## Project structure

```
src/
  main.py          # entry point — runs grid search over all datasets and models
  data_loader.py   # load and standardize ODDS, Clustbench, and verification data
  datasets.py      # dataset registry
  models.py        # model factory and hyperparameter grids
  experiment.py    # MLflow experiment runner with checkpointing
  ensemble.py      # majority-voting ensemble
notebooks/
  analysis.ipynb   # results analysis, plots, and verification ensemble
tests/             # pytest suite
report/            # LaTeX report sources
```

## Requirements

- Python ≥ 3.10
- [uv](https://docs.astral.sh/uv/) for dependency management

## Setup

```bash
make setup
```

This creates a virtual environment, installs dependencies, and sets up pre-commit hooks.

## Usage

```bash
make help          # list all targets
make run           # run the full experiment pipeline
make test          # run pytest
make mlflow        # start MLflow UI at http://localhost:5000
make pre-commit    # lint/format changed files
```

The main pipeline (`make run`) iterates over every ODDS and Clustbench dataset, fits each model over its hyperparameter grid with multiple random seeds, and logs metrics (F1, precision, recall, accuracy, AUC) to MLflow. Already-completed runs are skipped via parameter-hash checkpointing.

## MLflow

Experiments are stored under the `anomaly-detection` experiment name. By default, tracking uses a local SQLite database (`mlflow.db`) and artifact directory (`mlruns/`). Start the UI with:

```bash
make mlflow
```

## Analysis

Open `notebooks/analysis.ipynb` after running experiments. The notebook:

- aggregates MLflow runs into comparison tables and heatmaps
- visualizes hyperparameter sensitivity and robustness
- builds a majority-voting ensemble from the best ODDS configurations
- predicts labels for the verification set and writes `test_labels.csv`

## Data layout

```
data/
  odds/                      # ODDS .mat files (e.g. annthyroid.mat)
  clusterin_benchmarks/      # Clustbench .data.gz and .labels*.gz files
  test_data.csv              # unlabeled verification features
```

## License

GPL-3.0 — see [LICENSE](LICENSE).
