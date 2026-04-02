"""Scenario adapter for competitive programming / contest code generation.

Handles LiveCodeBench and similar datasets where the task is to produce a
complete program that reads from stdin and writes to stdout.
"""

from __future__ import annotations

import json
import re
import textwrap
from typing import Any

from codebench.core.interfaces.scenario import ScenarioAdapter
from codebench.core.models.common import (
    ExecutionResult,
    ProviderRequest,
    ProviderResponse,
    ScenarioType,
    ScoringResult,
)

_SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert competitive programmer. Solve the given problem.
    Write a complete Python program that reads from stdin and writes to stdout.
    Return ONLY the Python code — no markdown fences, no explanations.
""").strip()

_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


class ContestCodegenAdapter(ScenarioAdapter):
    """Adapter for ``contest_codegen`` scenarios (LiveCodeBench, etc.)."""

    @property
    def scenario_type(self) -> ScenarioType:
        return ScenarioType.CONTEST_CODEGEN

    def format_prompt(self, instance: dict[str, Any]) -> ProviderRequest:
        content = instance.get("question_content", "")
        title = instance.get("question_title", "")
        starter = instance.get("starter_code", "")

        test_cases = instance.get("public_test_cases", [])
        if isinstance(test_cases, str):
            test_cases = json.loads(test_cases)

        examples = ""
        for i, tc in enumerate(test_cases[:3], 1):
            inp = tc.get("input", "").strip()
            out = tc.get("output", "").strip()
            examples += f"\n\nExample {i}:\nInput:\n{inp}\nOutput:\n{out}"

        prompt = f"# {title}\n\n{content}"
        if starter:
            prompt += f"\n\nStarter code:\n{starter}"
        if examples:
            prompt += examples

        return ProviderRequest(
            prompt=prompt,
            system_prompt=_SYSTEM_PROMPT,
            metadata={
                "task_id": instance.get("question_id", instance.get("id", "")),
                "platform": instance.get("platform", ""),
            },
        )

    def extract_submission(self, response: ProviderResponse) -> str:
        content = response.content.strip()
        if not content:
            return ""
        matches = _FENCE_RE.findall(content)
        if matches:
            return matches[-1].strip()
        for marker in ("import ", "from ", "def ", "n = ", "t = ", "for "):
            idx = content.find(marker)
            if idx > 0:
                return content[idx:].strip()
        return content

    def build_execution_payload(
        self, instance: dict[str, Any], submission: str
    ) -> dict[str, Any]:
        """Build a harness that feeds each test case via stdin and checks stdout."""
        test_cases = instance.get("public_test_cases", [])
        if isinstance(test_cases, str):
            test_cases = json.loads(test_cases)

        runner = textwrap.dedent("""\
            import sys, io

            _SUBMISSION = {submission!r}
            _TEST_CASES = {test_cases!r}

            _all_passed = True
            for _i, _tc in enumerate(_TEST_CASES):
                _stdin_data = _tc.get("input", "")
                _expected = _tc.get("output", "").strip()
                _old_stdin, _old_stdout = sys.stdin, sys.stdout
                sys.stdin = io.StringIO(_stdin_data)
                sys.stdout = _captured = io.StringIO()
                try:
                    exec(_SUBMISSION, {{"__name__": "__main__"}})
                except SystemExit:
                    pass
                except Exception as _e:
                    sys.stdin, sys.stdout = _old_stdin, _old_stdout
                    print(f"Case {{_i}}: RUNTIME ERROR: {{_e}}", file=sys.stderr)
                    _all_passed = False
                    continue
                finally:
                    sys.stdin, sys.stdout = _old_stdin, _old_stdout
                _actual = _captured.getvalue().strip()
                if _actual != _expected:
                    print(f"Case {{_i}}: WRONG ANSWER", file=sys.stderr)
                    print(f"  Expected: {{_expected[:200]}}", file=sys.stderr)
                    print(f"  Got:      {{_actual[:200]}}", file=sys.stderr)
                    _all_passed = False

            if not _all_passed:
                sys.exit(1)
        """).format(submission=submission, test_cases=test_cases)

        return {"language": "python", "code": runner, "timeout_seconds": 30}

    def score(
        self,
        instance: dict[str, Any],
        submission: str,
        execution_result: ExecutionResult | None,
    ) -> ScoringResult:
        if execution_result is None:
            return ScoringResult(
                score=0.0, passed=False, details={"reason": "no_execution"}
            )
        if execution_result.timed_out:
            return ScoringResult(
                score=0.0, passed=False, details={"reason": "timeout"}
            )
        passed = execution_result.exit_code == 0
        return ScoringResult(
            score=1.0 if passed else 0.0,
            passed=passed,
            details={
                "exit_code": execution_result.exit_code,
                "stdout_snippet": execution_result.stdout[:500],
                "stderr_snippet": execution_result.stderr[:500],
                "task_id": instance.get("question_id", instance.get("id", "")),
            },
        )
