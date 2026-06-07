.PHONY: install install-dev test test-cov lint lint-fix clean

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest -xvs

test-cov:
	pytest --cov=core --cov-report=term -xvs

lint:
	ruff check . && mypy core/

lint-fix:
	ruff check --fix .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
