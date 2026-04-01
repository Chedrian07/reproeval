# Quickstart

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker (for sandboxed execution)

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd codebench

# Install with dev dependencies
uv pip install -e ".[dev]"
```

## Running Your First Benchmark

### 1. Create a run config

Create `example_run.yaml`:

```yaml
provider:
  name: openai_chat_completions
  model: gpt-4o
  api_key_env: OPENAI_API_KEY

dataset:
  name: humaneval_plus
  version: "1.0"
  max_instances: 5

scenario_type: function_codegen
track_type: lm_only

sandbox:
  backend: docker
  timeout_seconds: 30
  network_enabled: false

artifacts_dir: artifacts
seed: 42
```

### 2. Set your API key

```bash
export OPENAI_API_KEY="sk-..."
```

### 3. Dry run (validate config)

```bash
codebench run example_run.yaml --dry-run
```

### 4. Execute the benchmark

```bash
codebench run example_run.yaml
```

### 5. Review results

```bash
# List all runs
codebench list-runs

# Replay a specific run
codebench replay <run-id>
```

## Available Commands

```bash
codebench version        # Print version
codebench run            # Execute a benchmark run
codebench list-datasets  # Show registered datasets
codebench list-runs      # Show completed runs
codebench replay         # Inspect a previous run
```

## Development

```bash
make format      # Format code
make lint        # Run linter
make typecheck   # Run type checker
make test        # Run all tests (except e2e)
make test-unit   # Run unit tests only
```
