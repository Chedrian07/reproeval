# reproeval

Benchmark and evaluation framework for coding LLMs and coding agents.

Feed coding problems to a model, execute the generated code in a sandbox, and score by test pass rate.

## Key Features

- **Provider-neutral** — Works with OpenAI Responses API, Chat Completions API, or any compatible server (vLLM, Ollama, LM Studio, etc.)
- **7 benchmarks built-in** — HumanEval+, MBPP+, SWE-bench, LiveCodeBench, BigCodeBench-Hard, CRUXEval
- **Single `.env` config** — No config files needed, just set environment variables and run
- **Sandbox isolation** — Docker containers with network disabled, memory/CPU/PID limits, all capabilities dropped
- **Full artifact capture** — Every run saves inputs, outputs, and scoring results as JSON for replay and analysis

## Quick Start

### 1. Install

```bash
git clone https://github.com/Chedrian07/reproeval.git
cd reproeval

uv pip install -e ".[dev]"   # framework
make sandbox-deps            # benchmark test dependencies (numpy, pandas, etc.)
```

### 2. Download Datasets

```bash
./download-datasets.sh                          # all 7 datasets (3,337 instances)
./download-datasets.sh humaneval_plus mbpp_plus  # or specific ones
./download-datasets.sh --list                    # check download status
```

### 3. Configure API

```bash
cp .env.example .env
```

Edit `.env`:

```env
CODEBENCH_PROVIDER=openai_responses       # openai_responses or openai_chat_completions
CODEBENCH_MODEL=gpt-4o                    # model name
CODEBENCH_API_KEY=sk-...                  # API key
CODEBENCH_BASE_URL=https://api.openai.com # API endpoint (local servers work too)
CODEBENCH_SANDBOX=local                   # local (simple) or docker (isolated)
CODEBENCH_MAX_INSTANCES=                  # empty = all, number = limit per dataset
CODEBENCH_REASONING_EFFORT=               # low / medium / high (Responses API only)
```

### 4. Run Benchmarks

```bash
codebench run                      # benchmark all downloaded datasets
codebench run -d humaneval_plus    # specific dataset
codebench run -n 10                # max 10 instances per dataset
codebench run --dry-run            # validate config without executing
```

### 5. View Results

```bash
codebench list-runs                # list all runs
codebench replay <run-id>          # inspect a specific run
codebench list-datasets            # show registered datasets
```

Example output:

```
                      Benchmark Results
┏━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Dataset        ┃ Passed ┃ Total ┃ Pass Rate ┃ Run ID       ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ humaneval_plus │     10 │    10 │    100.0% │ b5cd9144ece6 │
│ mbpp_plus      │      8 │    10 │     80.0% │ a3f2e1d09c87 │
├────────────────┼────────┼───────┼───────────┼──────────────┤
│ TOTAL          │     18 │    20 │     90.0% │              │
└────────────────┴────────┴───────┴───────────┴──────────────┘
```

## How Scoring Works

```
Prompt → LLM → Generated Code → Combined with Dataset Tests → Sandbox Execution → exit 0 = PASS
```

The generated function code is concatenated with the dataset's test assertions and executed.
If all assertions pass (exit code 0) → **PASS**. Otherwise → **FAIL**.

## Benchmark Datasets

| Dataset | Instances | Scenario | Description |
|---|---|---|---|
| `humaneval_plus` | 164 | function_codegen | Function implementation with extended tests (EvalPlus) |
| `mbpp_plus` | 378 | function_codegen | Basic programming with extended tests (EvalPlus) |
| `bigcodebench_hard` | 140 | function_codegen | Library-heavy hard problems (BigCode) |
| `swe_bench_lite` | 300 | repo_patch | Real GitHub issue patch generation (Princeton NLP) |
| `swe_bench_verified` | 500 | repo_patch | Verified GitHub issue patch generation (Princeton NLP) |
| `livecodebench_lite` | 1,055 | contest_codegen | Competitive programming (Codeforces/LeetCode) |
| `cruxeval` | 800 | code_reasoning | Code input/output prediction |

## Scenario Types

| Scenario | Description | Status |
|---|---|---|
| `function_codegen` | Generate standalone functions, verify with tests | Implemented |
| `repo_patch` | Generate patches for existing repositories | Stub |
| `contest_codegen` | Competitive programming (stdin/stdout) | Stub |
| `code_reasoning` | Predict code execution results | Stub |
| `terminal_agent` | Terminal-based task completion | Stub |

## Provider Examples

### OpenAI Official API

```env
CODEBENCH_PROVIDER=openai_responses
CODEBENCH_MODEL=gpt-4o
CODEBENCH_API_KEY=sk-...
CODEBENCH_BASE_URL=https://api.openai.com
```

### Local / Compatible Servers (vLLM, Ollama, LM Studio, etc.)

```env
CODEBENCH_PROVIDER=openai_chat_completions
CODEBENCH_MODEL=Qwen3.5-35B-A3B-Fast
CODEBENCH_API_KEY=api
CODEBENCH_BASE_URL=http://localhost:8317
```

### Reasoning Models (o1, o3, etc.)

```env
CODEBENCH_PROVIDER=openai_responses
CODEBENCH_MODEL=o3-mini
CODEBENCH_API_KEY=sk-...
CODEBENCH_REASONING_EFFORT=medium
```

## Docker Sandbox

Run generated code in an isolated Docker container instead of a local process:

```bash
# 1. Build the sandbox image
make docker-sandbox

# 2. Update .env
CODEBENCH_SANDBOX=docker
```

Docker sandbox security:
- `--network=none` (network disabled)
- `--memory` / `--pids-limit` / `--cpus` (resource limits)
- `cap_drop=ALL` (all capabilities dropped)
- `read_only=True` (read-only root filesystem)
- `user=nobody` (non-root execution)

## YAML Config (Optional)

For fine-grained control, use a YAML config file instead of `.env`:

```yaml
# examples/humaneval_run.yaml
provider:
  name: openai_chat_completions
  model: gpt-4o
  api_key_env: OPENAI_API_KEY

dataset:
  name: humaneval_plus
  version: "1.0"
  max_instances: 10

scenario_type: function_codegen
sandbox:
  backend: local
  timeout_seconds: 30

artifacts_dir: artifacts
seed: 42
```

```bash
codebench run examples/humaneval_run.yaml
```

## Artifact Structure

All run results are saved to `artifacts/<run-id>/`:

```
artifacts/
  b5cd9144ece6/
    manifest.json          # run metadata (model, config, result summary)
    a1b2c3d4/
      instance_input.json  # input data
      result.json          # LLM response, execution result, scoring
    e5f6g7h8/
      ...
```

Replay any run anytime with `codebench replay <run-id>`.

## Development

```bash
make format      # code formatting (ruff)
make lint        # lint check (ruff)
make typecheck   # type check (mypy strict)
make test        # run tests (171 tests)
make test-unit   # unit tests only
make test-e2e    # e2e tests only
```

## Project Structure

```
src/codebench/
  core/              # framework foundation (interfaces, models, pipeline)
  providers/         # LLM adapters (OpenAI Responses, Chat Completions)
  scenarios/         # benchmark scenario adapters
  datasets/          # dataset registry and manifests
  sandbox/           # code execution engines (Docker, Local)
  scoring/           # scoring logic
  artifacts/         # artifact storage
  cli/               # CLI (Typer)
tests/               # 171 tests (unit + e2e)
docs/                # documentation
data/                # downloaded datasets (.gitignore'd)
```

## Documentation

- [Architecture](docs/architecture.md) — System design and data flow
- [Dataset Portfolio](docs/dataset_portfolio.md) — Dataset details and how to add new ones
- [Quickstart](docs/quickstart.md) — Detailed getting started guide
- [Testing Guide](docs/testing.md) — Test structure and execution
- [Security Review](docs/security_review.md) — Security audit results and mitigations

## License

MIT
