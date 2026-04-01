.PHONY: format lint typecheck test test-unit test-integration test-e2e security install dev sandbox-deps download-data docker-sandbox clean

install:
	uv pip install -e .

dev:
	uv pip install -e ".[dev]"

sandbox-deps:
	uv pip install -r sandbox-requirements.txt

download-data:
	./download-datasets.sh

docker-sandbox:
	docker build -f Dockerfile.sandbox -t codebench-sandbox .

format:
	uv run ruff format src tests
	uv run ruff check --fix src tests

lint:
	uv run ruff check src tests

typecheck:
	uv run mypy src/codebench

test:
	uv run pytest -m "not e2e" --tb=short -q

test-unit:
	uv run pytest tests/unit --tb=short -q

test-integration:
	uv run pytest tests/integration --tb=short -q

test-e2e:
	uv run pytest tests/e2e --tb=short -q

security:
	uv run ruff check --select S src tests

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist build *.egg-info
