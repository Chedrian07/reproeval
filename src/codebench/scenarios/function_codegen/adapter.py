"""Scenario adapter for function-level code generation benchmarks.

Handles HumanEval+, MBPP+, BigCodeBench-Hard, and similar datasets where the
task is to produce a standalone function body that satisfies a test suite.
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
    You are an expert Python programmer. Complete the function according to
    the docstring specification. Return ONLY the Python code — no markdown
    fences, no explanations.
""").strip()

# Regex to strip markdown code fences from LLM output.
_FENCE_RE = re.compile(
    r"```(?:python|py)?\s*\n(.*?)```",
    re.DOTALL,
)


class FunctionCodegenAdapter(ScenarioAdapter):
    """Adapter for ``function_codegen`` scenarios."""

    @property
    def scenario_type(self) -> ScenarioType:
        return ScenarioType.FUNCTION_CODEGEN

    # -- prompt formatting ------------------------------------------------

    def format_prompt(self, instance: dict[str, Any]) -> ProviderRequest:
        """Build a :class:`ProviderRequest` from a dataset instance.

        Expected instance keys: ``task_id``, ``prompt``, ``entry_point`` (optional).
        """
        prompt = instance["prompt"]
        entry_point = instance.get("entry_point", "")
        user_prompt = f"Complete the following Python function.\n\n{prompt}"
        return ProviderRequest(
            prompt=user_prompt,
            system_prompt=_SYSTEM_PROMPT,
            temperature=0.0,
            metadata={
                "task_id": instance.get("task_id", ""),
                "entry_point": entry_point,
            },
        )

    # -- submission extraction --------------------------------------------

    def extract_submission(self, response: ProviderResponse) -> str:
        """Extract clean Python source from the provider response.

        If the response contains markdown fences, takes the LAST code block
        (LLMs often put the real solution last after explanations).
        Otherwise strips leading non-code preamble heuristically.
        """
        content = response.content.strip()
        if not content:
            return ""

        # Find all fenced code blocks, take the last one
        matches = _FENCE_RE.findall(content)
        if matches:
            return matches[-1].strip()

        # No fences — try to strip preamble text before first 'def ' or 'import '
        for marker in ("def ", "import ", "from "):
            idx = content.find(marker)
            if idx > 0:
                return content[idx:].strip()

        return content

    # -- execution payload ------------------------------------------------

    def build_execution_payload(
        self,
        instance: dict[str, Any],
        submission: str,
    ) -> dict[str, Any]:
        """Combine the candidate function with the test harness.

        Handles two test formats:
        - HumanEval-style: test has ``def check(candidate):`` wrapper
          → append ``check(entry_point)`` to invoke it
        - MBPP-style: test has inline assertions (no check wrapper)
          → use test code as-is, no extra call needed
        """
        test_code = instance.get("test", "")
        entry_point = instance.get("entry_point", "")

        parts = [submission, "", test_code]

        # Only append check(entry_point) if the test defines a check() wrapper
        if entry_point and "def check(" in test_code:
            parts.append("")
            parts.append(f"check({entry_point})")

        full_code = "\n".join(parts)

        return {
            "language": "python",
            "code": full_code,
            "entry_point": entry_point,
            "timeout_seconds": 30,
        }

    # -- scoring ----------------------------------------------------------

    def score(
        self,
        instance: dict[str, Any],
        submission: str,
        execution_result: ExecutionResult | None,
    ) -> ScoringResult:
        """Pass/fail scoring based on sandbox exit code."""
        if execution_result is None:
            return ScoringResult(
                score=0.0,
                passed=False,
                details={"reason": "no_execution"},
            )

        if execution_result.timed_out:
            return ScoringResult(
                score=0.0,
                passed=False,
                details={"reason": "timeout"},
            )

        passed = execution_result.exit_code == 0
        return ScoringResult(
            score=1.0 if passed else 0.0,
            passed=passed,
            details={
                "exit_code": execution_result.exit_code,
                "stdout_snippet": execution_result.stdout[:500],
                "stderr_snippet": execution_result.stderr[:500],
                "task_id": instance.get("task_id", ""),
            },
        )
