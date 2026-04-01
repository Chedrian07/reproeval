# codebench

Benchmark and evaluation framework for coding LLMs and coding agents.

## Features

- **Provider-neutral** — Swap LLM providers without changing benchmark logic
- **Reproducible** — Every run captures model, prompt version, dataset snapshot, seed, and full results
- **Isolated** — Docker sandbox with network disabled by default
- **Auditable** — Full artifact capture for replay and analysis
- **Extensible** — Config-driven dataset registry and scenario adapters

## Supported Scenarios

| Scenario | Description | Status |
|---|---|---|
| `function_codegen` | Generate standalone functions (HumanEval+, MBPP+, BigCodeBench-Hard) | Implemented |
| `repo_patch` | Generate patches for repositories (SWE-bench) | Stub |
| `contest_codegen` | Competitive programming (LiveCodeBench) | Stub |
| `code_reasoning` | Predict code behavior (CRUXEval) | Stub |
| `terminal_agent` | Terminal-based tasks (Terminal-Bench) | Stub |

## Quick Start

```bash
# Install
uv pip install -e ".[dev]"

# Run a benchmark
export OPENAI_API_KEY="sk-..."
codebench run examples/humaneval_run.yaml

# View results
codebench list-runs
codebench replay <run-id>
```

## Development

```bash
make format      # Format code
make lint        # Lint
make typecheck   # Type check
make test        # Run tests
```

## Documentation

- [Architecture](docs/architecture.md)
- [Quickstart](docs/quickstart.md)
- [Dataset Portfolio](docs/dataset_portfolio.md)
- [Testing Guide](docs/testing.md)
- [Security Review](docs/security_review.md)

## License

MIT
