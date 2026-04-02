"""Scenario adapter for code reasoning benchmarks.

Handles CRUXEval and similar datasets where the task is to predict the output
of a given function call without executing code. Scoring is done by running
the code to verify the LLM's prediction matches actual output.
"""

from __future__ import annotations

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
    You are an expert at reasoning about Python code.
    Predict the exact output of the given function call.
    Return ONLY the output value — no explanation, no code, no markdown.
""").strip()

_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


class CodeReasoningAdapter(ScenarioAdapter):
    """Adapter for ``code_reasoning`` scenarios (CRUXEval, etc.)."""

    @property
    def scenario_type(self) -> ScenarioType:
        return ScenarioType.CODE_REASONING

    def format_prompt(self, instance: dict[str, Any]) -> ProviderRequest:
        code = instance.get("code", "")
        input_val = instance.get("input", "")

        prompt = textwrap.dedent(f"""\
            Given the following Python function:

            ```python
            {code}
            ```

            What is the return value of `f({input_val})`?

            Return ONLY the exact Python value (e.g., `[1, 2, 3]` or `"hello"`).
        """)

        return ProviderRequest(
            prompt=prompt,
            system_prompt=_SYSTEM_PROMPT,
            metadata={"task_id": instance.get("id", "")},
        )

    def extract_submission(self, response: ProviderResponse) -> str:
        content = response.content.strip()
        if not content:
            return ""
        # Strip markdown fences if present
        matches = _FENCE_RE.findall(content)
        if matches:
            return matches[-1].strip()
        # Strip backtick wrapping: `value`
        if content.startswith("`") and content.endswith("`"):
            content = content[1:-1].strip()
        return content

    def build_execution_payload(
        self, instance: dict[str, Any], submission: str
    ) -> dict[str, Any]:
        """Build a verifier that runs the actual code and compares output.

        Instead of trusting the LLM's prediction blindly, we execute
        the code and compare the real output against the LLM's answer.
        """
        code = instance.get("code", "")
        input_val = instance.get("input", "")

        lines = [
            "import sys",
            "",
            "# The original function",
            code,
            "",
            "# Compute actual output",
            f"_actual = f({input_val})",
            "",
            "# The LLM's predicted output",
            "try:",
            f"    _predicted = eval({submission!r})",
            "except Exception:",
            f"    print('Cannot parse LLM prediction: ' + {submission!r}, file=sys.stderr)",
            "    sys.exit(1)",
            "",
            "if _actual == _predicted:",
            "    sys.exit(0)",
            "else:",
            "    print(f'Expected: {_actual!r}', file=sys.stderr)",
            "    print(f'Predicted: {_predicted!r}', file=sys.stderr)",
            "    sys.exit(1)",
        ]

        return {"language": "python", "code": "\n".join(lines), "timeout_seconds": 10}

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
                "task_id": instance.get("id", ""),
            },
        )
