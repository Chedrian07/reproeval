# Dataset Portfolio

## MVP Benchmark Datasets

| Dataset | Scenario | Track | Instances | Language | Status |
|---|---|---|---|---|---|
| HumanEval+ | function_codegen | lm_only | 164 | Python | Manifest ready |
| MBPP+ | function_codegen | lm_only | 399 | Python | Manifest ready |
| SWE-bench Lite | repo_patch | standard_tools | 300 | Python | Manifest ready |
| SWE-bench Verified | repo_patch | standard_tools | 500 | Python | Manifest ready |
| LiveCodeBench Lite | contest_codegen | lm_only | 400 | Multi | Manifest ready |
| BigCodeBench-Hard | function_codegen | lm_only | 148 | Python | Manifest ready |
| CRUXEval | code_reasoning | lm_only | 800 | Python | Manifest ready |

## Planned (Post-MVP)

| Dataset | Scenario | Track | Notes |
|---|---|---|---|
| SWE-bench Live | repo_patch | standard_tools | Requires live repo access |
| RepoBench-P | function_codegen | lm_only | Cross-file completion |
| Terminal-Bench 2.0 | terminal_agent | open_agent | Terminal interaction |

## Registry Schema

Each dataset is described by a YAML manifest in `src/codebench/datasets/manifests/`:

```yaml
name: humaneval_plus
version: "1.0"
description: "HumanEval+ with extended test cases"
scenario_type: function_codegen
track_types: [lm_only]
instance_count: 164
source_url: "https://github.com/evalplus/evalplus"
license: MIT
language: python
splits: [test]
format_type: jsonl
required_fields: [task_id, prompt, entry_point, test]
```

## Adding a New Dataset

1. Create a manifest YAML in `src/codebench/datasets/manifests/`
2. Implement a dataset adapter if the format differs from standard JSONL
3. Add test fixtures in `tests/fixtures/`
4. Register in the DatasetRegistry
