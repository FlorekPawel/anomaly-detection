help:
	@echo "Available targets:"
	@echo "  make help                   - Show this help message"
	@echo "  make setup                  - Create venv and install deps"
	@echo "  make install                - Install dependencies and hooks"
	@echo "  make clean                  - Clean virtual environment and lockfile"
	@echo "  make test                   - Run tests"
	@echo "  make run                    - Run main pipeline"
	@echo "  make mlflow                 - Start MLflow UI (http://localhost:5000)"
	@echo "  make pre-commit             - Run pre-commit checks on changed files"
	@echo "  make pre-commit-all         - Run pre-commit checks on all files"

# install dependencies and pre-commit hooks
install:
	uv sync --all-groups
	uv run pre-commit install

# setup venv and dependencies
setup: install

# clean up virtual environment and lockfile
clean:
	rm -rf .venv
	rm -rf uv.lock

# run tests with pytest
test::
	uv run pytest -v

# run main pipeline
run:
	uv run python -m src.main

# start MLflow UI
mlflow:
	uv run mlflow ui

# pre-commit checks on changed files only
pre-commit:
	uv run pre-commit run

# pre-commit checks (linting, formatting, type checking)
pre-commit-all:
	uv run pre-commit run --all-files
