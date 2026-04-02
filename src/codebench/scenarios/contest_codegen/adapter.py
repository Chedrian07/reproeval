"""Scenario adapter for competitive programming / contest code generation.

Handles LiveCodeBench and similar datasets with two test modes:
- **stdin**: Codeforces-style, program reads stdin and writes stdout
- **functional**: LeetCode-style, class Solution with method calls
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

_SYSTEM_PROMPT_STDIN = textwrap.dedent("""\
    You are an expert competitive programmer. Solve the given problem.
    Write a complete Python program that reads from stdin and writes to stdout.
    Return ONLY the Python code — no markdown fences, no explanations.
""").strip()

_SYSTEM_PROMPT_FUNCTIONAL = textwrap.dedent("""\
    You are an expert competitive programmer. Solve the given problem.
    Complete the given class method. Return ONLY the Python code including
    the full class definition — no markdown fences, no explanations.
""").strip()

_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)

# Common typing imports that LeetCode solutions need
_TYPING_PREAMBLE = (
    "from typing import List, Optional, Tuple, Dict, Set\n"
    "from collections import defaultdict, Counter, deque\n"
    "from itertools import accumulate, combinations, permutations\n"
    "from functools import lru_cache\n"
    "from heapq import heappush, heappop\n"
    "import math, bisect, sys\n"
)


def _detect_test_type(instance: dict[str, Any]) -> str:
    """Detect whether this instance uses stdin or functional tests."""
    test_cases = instance.get("public_test_cases", [])
    if isinstance(test_cases, str):
        test_cases = json.loads(test_cases)
    if test_cases and test_cases[0].get("testtype") == "functional":
        return "functional"
    return "stdin"


def _extract_method_name(starter_code: str) -> str:
    """Extract the method name from a LeetCode starter code."""
    m = re.search(r"def (\w+)\(self", starter_code)
    return m.group(1) if m else ""


class ContestCodegenAdapter(ScenarioAdapter):
    """Adapter for ``contest_codegen`` scenarios (LiveCodeBench, etc.)."""

    @property
    def scenario_type(self) -> ScenarioType:
        return ScenarioType.CONTEST_CODEGEN

    def format_prompt(self, instance: dict[str, Any]) -> ProviderRequest:
        content = instance.get("question_content", "")
        title = instance.get("question_title", "")
        starter = instance.get("starter_code", "")
        test_type = _detect_test_type(instance)

        test_cases = instance.get("public_test_cases", [])
        if isinstance(test_cases, str):
            test_cases = json.loads(test_cases)

        # Build examples section
        examples = ""
        for i, tc in enumerate(test_cases[:3], 1):
            inp = tc.get("input", "").strip()
            out = tc.get("output", "").strip()
            examples += f"\n\nExample {i}:\nInput: {inp}\nOutput: {out}"

        prompt = f"# {title}\n\n{content}"
        if starter:
            prompt += f"\n\nComplete this code:\n```python\n{starter}\n```"
        if examples:
            prompt += examples

        system = (
            _SYSTEM_PROMPT_FUNCTIONAL
            if test_type == "functional"
            else _SYSTEM_PROMPT_STDIN
        )

        return ProviderRequest(
            prompt=prompt,
            system_prompt=system,
            metadata={
                "task_id": instance.get("question_id", instance.get("id", "")),
                "platform": instance.get("platform", ""),
                "test_type": test_type,
            },
        )

    def extract_submission(self, response: ProviderResponse) -> str:
        content = response.content.strip()
        if not content:
            return ""
        matches = _FENCE_RE.findall(content)
        if matches:
            return matches[-1].strip()
        for marker in ("class ", "import ", "from ", "def ", "n = ", "t = "):
            idx = content.find(marker)
            if idx > 0:
                return content[idx:].strip()
        return content

    def build_execution_payload(
        self, instance: dict[str, Any], submission: str
    ) -> dict[str, Any]:
        test_type = _detect_test_type(instance)
        if test_type == "functional":
            return self._build_functional_payload(instance, submission)
        return self._build_stdin_payload(instance, submission)

    def _build_functional_payload(
        self, instance: dict[str, Any], submission: str
    ) -> dict[str, Any]:
        """LeetCode-style: instantiate Solution, call method, compare."""
        test_cases = instance.get("public_test_cases", [])
        if isinstance(test_cases, str):
            test_cases = json.loads(test_cases)

        starter = instance.get("starter_code", "")
        method_name = _extract_method_name(starter)

        lines = [
            _TYPING_PREAMBLE,
            "",
            "# LLM submission",
            submission,
            "",
            "import sys, json",
            "",
            f"_method = {method_name!r}",
            f"_test_cases = {json.dumps(test_cases)}",
            "",
            "_sol = Solution()",
            "_fn = getattr(_sol, _method)",
            "_all_passed = True",
            "",
            "for _i, _tc in enumerate(_test_cases):",
            "    _raw_input = _tc.get('input', '')",
            "    _expected_str = _tc.get('output', '').strip()",
            "    try:",
            "        _args = json.loads('[' + _raw_input + ']')",
            "    except json.JSONDecodeError:",
            "        _args = [_raw_input]",
            "    try:",
            "        _result = _fn(*_args)",
            "    except Exception as _e:",
            "        print(f'Case {_i}: RUNTIME ERROR: {_e}', file=sys.stderr)",
            "        _all_passed = False",
            "        continue",
            "    try:",
            "        _expected = json.loads(_expected_str)",
            "    except (json.JSONDecodeError, ValueError):",
            "        _expected = _expected_str",
            "    if _result != _expected:",
            "        print(f'Case {_i}: WRONG ANSWER', file=sys.stderr)",
            "        print(f'  Expected: {_expected!r}', file=sys.stderr)",
            "        print(f'  Got:      {_result!r}', file=sys.stderr)",
            "        _all_passed = False",
            "",
            "if not _all_passed:",
            "    sys.exit(1)",
        ]
        return {"language": "python", "code": "\n".join(lines), "timeout_seconds": 30}

    def _build_stdin_payload(
        self, instance: dict[str, Any], submission: str
    ) -> dict[str, Any]:
        """Codeforces-style: feed stdin, compare stdout."""
        test_cases = instance.get("public_test_cases", [])
        if isinstance(test_cases, str):
            test_cases = json.loads(test_cases)

        lines = [
            "import sys, io",
            "",
            f"_SUBMISSION = {submission!r}",
            f"_TEST_CASES = {json.dumps(test_cases)}",
            "",
            "_all_passed = True",
            "for _i, _tc in enumerate(_TEST_CASES):",
            '    _stdin_data = _tc.get("input", "")',
            '    _expected = _tc.get("output", "").strip()',
            "    _old_stdin, _old_stdout = sys.stdin, sys.stdout",
            "    sys.stdin = io.StringIO(_stdin_data)",
            "    sys.stdout = _captured = io.StringIO()",
            "    try:",
            '        exec(_SUBMISSION, {"__name__": "__main__"})',
            "    except SystemExit:",
            "        pass",
            "    except Exception as _e:",
            "        sys.stdin, sys.stdout = _old_stdin, _old_stdout",
            "        print(f'Case {_i}: RUNTIME ERROR: {_e}', file=sys.stderr)",
            "        _all_passed = False",
            "        continue",
            "    finally:",
            "        sys.stdin, sys.stdout = _old_stdin, _old_stdout",
            "    _actual = _captured.getvalue().strip()",
            "    if _actual != _expected:",
            "        print(f'Case {_i}: WRONG ANSWER', file=sys.stderr)",
            "        print(f'  Expected: {_expected[:200]}', file=sys.stderr)",
            "        print(f'  Got:      {_actual[:200]}', file=sys.stderr)",
            "        _all_passed = False",
            "",
            "if not _all_passed:",
            "    sys.exit(1)",
        ]
        return {"language": "python", "code": "\n".join(lines), "timeout_seconds": 30}

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
