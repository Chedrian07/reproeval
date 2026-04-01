#!/usr/bin/env python3
"""Download benchmark datasets from HuggingFace and convert to JSONL.

Usage:
    python scripts/download_datasets.py              # download all
    python scripts/download_datasets.py humaneval_plus mbpp_plus  # specific ones
    python scripts/download_datasets.py --list       # show available datasets
    python scripts/download_datasets.py --force      # re-download even if exists

Requires: pip install datasets huggingface_hub
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ── Dataset definitions ────────────────────────────────────────────────

DATASETS: dict[str, dict] = {
    "humaneval_plus": {
        "hf_repo": "evalplus/humanevalplus",
        "hf_split": "test",
        "fields": ["task_id", "prompt", "canonical_solution", "entry_point", "test"],
        "description": "HumanEval+ (164 instances, function_codegen)",
    },
    "mbpp_plus": {
        "hf_repo": "evalplus/mbppplus",
        "hf_split": "test",
        "fields": ["task_id", "prompt", "canonical_solution", "entry_point", "test"],
        "description": "MBPP+ (378 instances, function_codegen)",
    },
    "swe_bench_lite": {
        "hf_repo": "princeton-nlp/SWE-bench_Lite",
        "hf_split": "test",
        "fields": [
            "instance_id", "repo", "base_commit", "problem_statement",
            "hints_text", "patch", "test_patch", "version",
        ],
        "description": "SWE-bench Lite (300 instances, repo_patch)",
    },
    "swe_bench_verified": {
        "hf_repo": "princeton-nlp/SWE-bench_Verified",
        "hf_split": "test",
        "fields": [
            "instance_id", "repo", "base_commit", "problem_statement",
            "hints_text", "patch", "test_patch", "version",
        ],
        "description": "SWE-bench Verified (500 instances, repo_patch)",
    },
    "livecodebench_lite": {
        "method": "hf_download",  # direct JSONL download
        "hf_repo": "livecodebench/code_generation_lite",
        "hf_files": ["test.jsonl", "test2.jsonl", "test3.jsonl",
                      "test4.jsonl", "test5.jsonl", "test6.jsonl"],
        "description": "LiveCodeBench code gen lite (880+ instances, contest_codegen)",
    },
    "bigcodebench_hard": {
        "hf_repo": "bigcode/bigcodebench",
        "hf_split": "v0.1.4",
        "post_filter": lambda row: int(row["task_id"].split("/")[-1]) >= 1000,
        "fields": [
            "task_id", "complete_prompt", "instruct_prompt",
            "canonical_solution", "code_prompt", "test", "entry_point",
        ],
        "description": "BigCodeBench-Hard (140 instances, function_codegen)",
    },
    "cruxeval": {
        "hf_repo": "cruxeval-org/cruxeval",
        "hf_split": "test",
        "fields": ["id", "code", "input", "output"],
        "description": "CRUXEval (800 instances, code_reasoning)",
    },
}


def _download_via_load_dataset(name: str, info: dict, force: bool) -> Path:
    """Download using HuggingFace `datasets` library."""
    from datasets import load_dataset

    out_path = DATA_DIR / f"{name}.jsonl"
    if out_path.exists() and not force:
        print(f"  ✓ {name}: already exists ({out_path.name})")
        return out_path

    print(f"  ↓ {name}: downloading from {info['hf_repo']}...", flush=True)

    ds = load_dataset(
        info["hf_repo"],
        name=info.get("hf_subset"),
        split=info["hf_split"],
    )
    rows = list(ds)

    # Post-filter
    pf = info.get("post_filter")
    if pf:
        rows = [r for r in rows if pf(r)]

    # Field selection
    keep = info.get("fields")
    if keep:
        rows = [{k: r.get(k) for k in keep if k in r} for r in rows]

    # Ensure id field
    for i, row in enumerate(rows):
        if "id" not in row:
            row["id"] = (
                row.get("task_id") or row.get("instance_id") or f"{name}/{i}"
            )

    _write_jsonl(out_path, rows)
    print(f"  ✓ {name}: {len(rows)} instances → {out_path.name}")
    return out_path


def _download_via_hf_hub(name: str, info: dict, force: bool) -> Path:
    """Download raw JSONL files directly from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download

    out_path = DATA_DIR / f"{name}.jsonl"
    if out_path.exists() and not force:
        print(f"  ✓ {name}: already exists ({out_path.name})")
        return out_path

    print(f"  ↓ {name}: downloading {len(info['hf_files'])} file(s) "
          f"from {info['hf_repo']}...", flush=True)

    all_rows: list[dict] = []
    for filename in info["hf_files"]:
        local = hf_hub_download(
            repo_id=info["hf_repo"],
            filename=filename,
            repo_type="dataset",
        )
        with open(local, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    if "id" not in row:
                        row["id"] = (
                            row.get("task_id")
                            or row.get("question_id")
                            or f"{name}/{len(all_rows)}"
                        )
                    all_rows.append(row)

    _write_jsonl(out_path, all_rows)
    print(f"  ✓ {name}: {len(all_rows)} instances → {out_path.name}")
    return out_path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def download_dataset(name: str, info: dict, force: bool = False) -> Path:
    """Download a single dataset."""
    out_path = DATA_DIR / f"{name}.jsonl"
    try:
        if info.get("method") == "hf_download":
            return _download_via_hf_hub(name, info, force)
        else:
            return _download_via_load_dataset(name, info, force)
    except Exception as exc:
        print(f"  ✗ {name}: failed — {exc}")
        return out_path


def main() -> None:
    args = sys.argv[1:]

    if "--list" in args:
        print("Available datasets:\n")
        for name, info in DATASETS.items():
            out = DATA_DIR / f"{name}.jsonl"
            status = "✓ downloaded" if out.exists() else "  not yet"
            print(f"  {name:25s} {info['description']:55s} [{status}]")
        return

    force = "--force" in args
    args = [a for a in args if not a.startswith("--")]
    targets = args if args else list(DATASETS.keys())

    unknown = [t for t in targets if t not in DATASETS]
    if unknown:
        print(f"Unknown datasets: {', '.join(unknown)}")
        print(f"Available: {', '.join(DATASETS.keys())}")
        sys.exit(1)

    print(f"Downloading {len(targets)} dataset(s) to {DATA_DIR}/\n")

    ok, fail = 0, 0
    for name in targets:
        path = download_dataset(name, DATASETS[name], force=force)
        if path.exists():
            ok += 1
        else:
            fail += 1

    print(f"\nDone: {ok} downloaded, {fail} failed.")
    print("Run 'codebench run' to benchmark all downloaded datasets.")


if __name__ == "__main__":
    main()
