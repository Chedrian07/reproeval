# codebench Architecture

## Overview

codebench is a benchmark and evaluation framework for coding LLMs and coding agents. It provides a provider-neutral, reproducible, and auditable way to evaluate model performance across multiple benchmark scenarios.

## Core Design Principles

1. **Provider Neutrality** — Benchmark logic is independent of any specific LLM provider
2. **Reproducibility** — Every run captures model, adapter, prompt version, dataset snapshot, harness version, seed, and results
3. **Isolation** — Sandboxed execution with no implicit network access
4. **Auditability** — Full artifact capture (diffs, logs, commands, stdout/stderr, exit codes, judge decisions)
5. **Separation of Concerns** — Provider response shapes don't leak into benchmark core

## Module Structure

```
src/codebench/
├── core/              # Framework foundation
│   ├── config/        # RunConfig, ProviderConfig, SandboxConfig
│   ├── interfaces/    # ABCs: ProviderInterface, ScenarioAdapter, SandboxRunner, Scorer, ArtifactStore
│   ├── models/        # Pydantic models: ProviderRequest/Response, RunManifest, etc.
│   └── runner/        # BenchmarkPipeline orchestrator
├── providers/         # LLM provider adapters
│   ├── openai_responses/
│   └── openai_chat_completions/
├── scenarios/         # Benchmark scenario adapters
│   ├── function_codegen/
│   ├── repo_patch/
│   ├── contest_codegen/
│   ├── code_reasoning/
│   └── terminal_agent/
├── datasets/          # Dataset registry and manifests
│   ├── registry/
│   ├── manifests/
│   └── adapters/
├── sandbox/           # Code execution backends
│   └── docker/
├── scoring/           # Scoring implementations
├── artifacts/         # Artifact storage
├── cli/               # Typer CLI
└── utils/             # Shared utilities
```

## Data Flow

```
Config → Registry → Instances → Scenario Adapter → Provider → Sandbox → Scorer → Artifacts
```

1. **Config** defines the run parameters (provider, dataset, scenario, track)
2. **Registry** loads dataset instances from manifests and JSONL files
3. **Scenario Adapter** formats prompts and maps instances to the benchmark format
4. **Provider** sends the prompt to the LLM and returns a structured response
5. **Sandbox** executes the generated code in an isolated Docker container
6. **Scorer** evaluates the execution result against expected behavior
7. **Artifacts** persists all intermediate data for replay and audit

## Track Types

- `lm_only` — Model generates code from prompt alone
- `standard_tools` — Model can use predefined tool functions
- `open_agent` — Model can use arbitrary tools and iterate

Tracks are kept separate — fundamentally different tracks are never collapsed into a single score.

## Benchmark Scenarios

| Scenario | Description | Example Datasets |
|---|---|---|
| `function_codegen` | Generate standalone functions | HumanEval+, MBPP+, BigCodeBench-Hard |
| `repo_patch` | Generate patches for existing repos | SWE-bench Lite/Verified |
| `contest_codegen` | Competitive programming problems | LiveCodeBench |
| `code_reasoning` | Predict code behavior | CRUXEval |
| `terminal_agent` | Terminal-based task completion | Terminal-Bench 2.0 |
