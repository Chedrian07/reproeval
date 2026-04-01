"""Tests for the function_codegen scenario adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from codebench.core.models.common import (
    ExecutionResult,
    ProviderResponse,
    ScenarioType,
    TokenUsage,
)
from codebench.scenarios.function_codegen.adapter import FunctionCodegenAdapter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures"


def _sample_instances() -> list[dict[str, Any]]:
    """Load sample HumanEval+ instances from the fixture file."""
    path = FIXTURES_DIR / "humaneval_sample.jsonl"
    instances: list[dict[str, Any]] = []
    with open(path) as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                instances.append(json.loads(stripped))
    return instances


def _make_response(content: str) -> ProviderResponse:
    """Build a minimal ProviderResponse for testing."""
    return ProviderResponse(
        content=content,
        model="test-model",
        usage=TokenUsage(input_tokens=10, output_tokens=20),
        latency_ms=100.0,
    )


def _make_execution_result(
    *,
    exit_code: int = 0,
    stdout: str = "",
    stderr: str = "",
    timed_out: bool = False,
) -> ExecutionResult:
    return ExecutionResult(
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        duration_ms=50.0,
        timed_out=timed_out,
    )


# ---------------------------------------------------------------------------
# Adapter basics
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFunctionCodegenAdapterBasics:
    def test_scenario_type(self) -> None:
        adapter = FunctionCodegenAdapter()
        assert adapter.scenario_type == ScenarioType.FUNCTION_CODEGEN

    def test_is_scenario_adapter_subclass(self) -> None:
        from codebench.core.interfaces.scenario import ScenarioAdapter

        assert issubclass(FunctionCodegenAdapter, ScenarioAdapter)


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFormatPrompt:
    def test_basic_prompt(self) -> None:
        adapter = FunctionCodegenAdapter()
        instance = {
            "task_id": "HumanEval/0",
            "prompt": 'def foo(x):\n    """Return x + 1."""\n',
            "entry_point": "foo",
        }
        req = adapter.format_prompt(instance)
        assert "Complete the following Python function" in req.prompt
        assert "def foo(x)" in req.prompt
        assert req.system_prompt is not None
        assert req.temperature == 0.0
        assert req.metadata["task_id"] == "HumanEval/0"
        assert req.metadata["entry_point"] == "foo"

    def test_prompt_from_fixture(self) -> None:
        adapter = FunctionCodegenAdapter()
        instances = _sample_instances()
        for inst in instances:
            req = adapter.format_prompt(inst)
            assert inst["prompt"] in req.prompt
            assert req.metadata["task_id"] == inst["task_id"]


# ---------------------------------------------------------------------------
# Submission extraction
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExtractSubmission:
    def test_plain_code(self) -> None:
        adapter = FunctionCodegenAdapter()
        code = "def foo(x):\n    return x + 1\n"
        resp = _make_response(code)
        assert adapter.extract_submission(resp) == code.strip()

    def test_strips_markdown_fences(self) -> None:
        adapter = FunctionCodegenAdapter()
        raw = "```python\ndef foo(x):\n    return x + 1\n```"
        resp = _make_response(raw)
        result = adapter.extract_submission(resp)
        assert "```" not in result
        assert "def foo(x)" in result

    def test_strips_py_fence_variant(self) -> None:
        adapter = FunctionCodegenAdapter()
        raw = "```py\ndef bar(): pass\n```"
        resp = _make_response(raw)
        result = adapter.extract_submission(resp)
        assert "```" not in result
        assert "def bar" in result

    def test_whitespace_only_response(self) -> None:
        adapter = FunctionCodegenAdapter()
        resp = _make_response("   \n  \n  ")
        result = adapter.extract_submission(resp)
        assert result == ""

    def test_multiple_code_blocks_takes_last(self) -> None:
        adapter = FunctionCodegenAdapter()
        raw = (
            "```python\ndef helper(): pass\n```\n"
            "Now the real one:\n"
            "```python\ndef solve(x): return x * 2\n```"
        )
        resp = _make_response(raw)
        result = adapter.extract_submission(resp)
        assert "solve" in result
        assert "helper" not in result

    def test_preamble_text_stripped(self) -> None:
        adapter = FunctionCodegenAdapter()
        raw = "Here is the solution:\ndef foo(x):\n    return x + 1"
        resp = _make_response(raw)
        result = adapter.extract_submission(resp)
        assert result.startswith("def foo")
        assert "Here is" not in result

    def test_empty_response(self) -> None:
        adapter = FunctionCodegenAdapter()
        resp = _make_response("")
        result = adapter.extract_submission(resp)
        assert result == ""


# ---------------------------------------------------------------------------
# Build execution payload
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildExecutionPayload:
    def test_payload_structure(self) -> None:
        adapter = FunctionCodegenAdapter()
        instance = {
            "task_id": "HumanEval/0",
            "prompt": "def foo(x): ...",
            "entry_point": "foo",
            "test": "def check(candidate):\n    assert candidate(1) == 2\n",
        }
        submission = "def foo(x):\n    return x + 1\n"
        payload = adapter.build_execution_payload(instance, submission)

        assert payload["language"] == "python"
        assert "code" in payload
        assert "timeout_seconds" in payload
        assert payload["entry_point"] == "foo"

    def test_payload_contains_submission_and_test(self) -> None:
        adapter = FunctionCodegenAdapter()
        instance = {
            "task_id": "T/0",
            "entry_point": "solve",
            "test": "def check(candidate):\n    assert candidate() == 42\n",
        }
        submission = "def solve():\n    return 42\n"
        payload = adapter.build_execution_payload(instance, submission)
        code = payload["code"]
        assert "def solve():" in code
        assert "def check(candidate):" in code
        assert "check(solve)" in code

    def test_payload_from_fixture(self) -> None:
        adapter = FunctionCodegenAdapter()
        instances = _sample_instances()
        for inst in instances:
            submission = inst.get("canonical_solution", "pass")
            payload = adapter.build_execution_payload(inst, submission)
            code = payload["code"]
            assert inst["entry_point"] in code
            # HumanEval fixtures have check(candidate) wrapper
            if "def check(" in inst.get("test", ""):
                assert f"check({inst['entry_point']})" in code

    def test_mbpp_style_no_check_wrapper(self) -> None:
        """MBPP tests use inline assertions — no check() should be appended."""
        adapter = FunctionCodegenAdapter()
        instance = {
            "task_id": "Mbpp/1",
            "prompt": "Write a function to find similar elements.",
            "test": "assertion(similar_elements((1,2),(2,3)), (2,), 0)",
        }
        payload = adapter.build_execution_payload(instance, "def similar_elements(a,b): pass")
        assert "check(" not in payload["code"]
        assert "assertion(" in payload["code"]

    def test_mbpp_style_no_entry_point(self) -> None:
        """MBPP instances may lack entry_point — should not crash."""
        adapter = FunctionCodegenAdapter()
        instance = {
            "task_id": "Mbpp/2",
            "prompt": "Write a function.",
            "test": "assert func(1) == 2",
        }
        payload = adapter.build_execution_payload(instance, "def func(x): return x+1")
        assert "check(" not in payload["code"]
        assert "assert func(1) == 2" in payload["code"]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestScoring:
    def test_pass(self) -> None:
        adapter = FunctionCodegenAdapter()
        instance = {"task_id": "T/0"}
        result = adapter.score(instance, "code", _make_execution_result(exit_code=0))
        assert result.passed is True
        assert result.score == 1.0

    def test_fail_nonzero_exit(self) -> None:
        adapter = FunctionCodegenAdapter()
        instance = {"task_id": "T/0"}
        result = adapter.score(
            instance, "code", _make_execution_result(exit_code=1, stderr="AssertionError")
        )
        assert result.passed is False
        assert result.score == 0.0
        assert result.details["exit_code"] == 1

    def test_fail_timeout(self) -> None:
        adapter = FunctionCodegenAdapter()
        instance = {"task_id": "T/0"}
        result = adapter.score(
            instance, "code", _make_execution_result(timed_out=True, exit_code=-1)
        )
        assert result.passed is False
        assert result.details["reason"] == "timeout"

    def test_fail_no_execution(self) -> None:
        adapter = FunctionCodegenAdapter()
        instance = {"task_id": "T/0"}
        result = adapter.score(instance, "code", None)
        assert result.passed is False
        assert result.details["reason"] == "no_execution"

    def test_score_includes_task_id(self) -> None:
        adapter = FunctionCodegenAdapter()
        instance = {"task_id": "HumanEval/99"}
        result = adapter.score(instance, "code", _make_execution_result(exit_code=0))
        assert result.details["task_id"] == "HumanEval/99"


# ---------------------------------------------------------------------------
# End-to-end prompt-to-payload round-trip
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRoundTrip:
    """Verify the full prompt -> extract -> payload pipeline with fixtures."""

    def test_fixture_round_trip(self) -> None:
        adapter = FunctionCodegenAdapter()
        instances = _sample_instances()
        inst = instances[2]  # truncate_number — simplest

        # Step 1: format prompt
        req = adapter.format_prompt(inst)
        assert "truncate_number" in req.prompt

        # Step 2: simulate LLM returning canonical solution
        resp = _make_response(inst["canonical_solution"])
        submission = adapter.extract_submission(resp)
        assert "return" in submission

        # Step 3: build payload
        payload = adapter.build_execution_payload(inst, submission)
        assert "check(truncate_number)" in payload["code"]

        # Step 4: score (simulating successful execution)
        result = adapter.score(inst, submission, _make_execution_result(exit_code=0))
        assert result.passed is True
