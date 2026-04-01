# Testing Guide

## Test Structure

```
tests/
├── unit/              # Fast, isolated tests
│   ├── test_core/     # Models, config, artifacts, pipeline
│   ├── test_providers/# Provider adapter tests (mocked HTTP)
│   ├── test_scenarios/# Scenario adapter tests
│   ├── test_datasets/ # Registry and manifest tests
│   ├── test_sandbox/  # Sandbox runner tests (mocked Docker)
│   └── test_scoring/  # Scorer tests
├── integration/       # Cross-module tests
├── e2e/               # Full pipeline tests
└── fixtures/          # Shared test data
```

## Running Tests

```bash
# All tests except e2e
make test

# Unit tests only
make test-unit

# Integration tests
make test-integration

# End-to-end tests (Docker optional)
make test-e2e
```

## Markers

- `@pytest.mark.unit` — Unit tests, no external dependencies
- `@pytest.mark.integration` — Cross-module, may use fixtures
- `@pytest.mark.e2e` — Full pipeline with fixture data (Docker optional)

## Writing Tests

### Unit Tests

- Mock external dependencies (HTTP, Docker)
- Use `tmp_path` for filesystem tests
- Use fixtures from `tests/fixtures/`
- Test both success and error paths

### Integration Tests

- Test cross-module interactions
- Use fixture-backed data
- Verify artifact persistence

### E2E Tests

- Run complete benchmark flows
- Full pipeline tests with fixture data (Docker optional)
- Verify reproducibility and artifact capture

## Determinism

- All tests must be deterministic
- Use explicit seeds where randomness is involved
- Mock time-dependent operations
- Avoid network calls in unit/integration tests
