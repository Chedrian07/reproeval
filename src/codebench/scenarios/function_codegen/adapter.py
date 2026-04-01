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

        Supports multiple prompt field names:
        - ``prompt`` (HumanEval+, MBPP+)
        - ``instruct_prompt`` (BigCodeBench — natural language instruction)
        - ``complete_prompt`` (BigCodeBench — code completion with docstring)
        """
        prompt = (
            instance.get("prompt")
            or instance.get("instruct_prompt")
            or instance.get("complete_prompt")
            or ""
        )
        if not prompt:
            raise KeyError("No prompt field found in instance")
        entry_point = instance.get("entry_point", "")
        user_prompt = f"Complete the following Python function.\n\n{prompt}"
        return ProviderRequest(
            prompt=user_prompt,
            system_prompt=_SYSTEM_PROMPT,
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

        Also prepends any import statements from the original prompt so that
        type annotations (``List``, ``Tuple``, etc.) are available even when
        the LLM omits them from its response.
        """
        test_code = instance.get("test", "")
        entry_point = instance.get("entry_point", "")
        prompt = instance.get("prompt", "")

        # Extract imports from the original prompt and ensure they are present
        prompt_imports = _extract_imports(prompt)
        submission_imports = _extract_imports(submission)
        missing_imports = [i for i in prompt_imports if i not in submission_imports]

        parts: list[str] = []
        if missing_imports:
            parts.extend(missing_imports)
            parts.append("")
        parts.append(submission)
        parts.append("")
        parts.append(test_code)

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


def _extract_imports(code: str) -> list[str]:
    """Extract import lines from source code."""
    imports: list[str] = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            imports.append(stripped)
    return imports
