"""Scenario adapter for repository-level patch generation benchmarks.

Handles SWE-bench Lite, SWE-bench Verified, and similar datasets where the
task is to produce a unified diff that resolves a GitHub issue.

Note: Full SWE-bench evaluation requires cloning repos and running test suites
in Docker. This adapter implements a simplified version that asks for a patch
and scores based on structural validity (applies cleanly). For full evaluation,
use the official SWE-bench harness.
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
    You are an expert software engineer. Given a GitHub issue and repository
    context, produce a unified diff (patch) that resolves the issue.
    Return ONLY the unified diff — no explanations, no markdown fences around
    the diff itself. Start with --- and +++ lines.
""").strip()

_FENCE_RE = re.compile(r"```(?:diff|patch)?\s*\n(.*?)```", re.DOTALL)
_DIFF_START_RE = re.compile(r"^---\s", re.MULTILINE)


class RepoPatchAdapter(ScenarioAdapter):
    """Adapter for ``repo_patch`` scenarios (SWE-bench, etc.)."""

    @property
    def scenario_type(self) -> ScenarioType:
        return ScenarioType.REPO_PATCH

    def format_prompt(self, instance: dict[str, Any]) -> ProviderRequest:
        repo = instance.get("repo", "")
        problem = instance.get("problem_statement", "")
        hints = instance.get("hints_text", "")
        version = instance.get("version", "")

        prompt = f"## Repository: {repo}"
        if version:
            prompt += f" (version {version})"
        prompt += f"\n\n## Issue\n\n{problem}"
        if hints:
            prompt += f"\n\n## Hints\n\n{hints}"
        prompt += (
            "\n\n## Task\n\n"
            "Produce a unified diff (patch) that resolves the issue above. "
            "The patch should apply cleanly to the repository."
        )

        return ProviderRequest(
            prompt=prompt,
            system_prompt=_SYSTEM_PROMPT,
            metadata={
                "task_id": instance.get("instance_id", instance.get("id", "")),
                "repo": repo,
            },
        )

    def extract_submission(self, response: ProviderResponse) -> str:
        content = response.content.strip()
        if not content:
            return ""
        # Try fenced diff block
        matches = _FENCE_RE.findall(content)
        if matches:
            return matches[-1].strip()
        # Try to find start of diff
        m = _DIFF_START_RE.search(content)
        if m:
            return content[m.start():].strip()
        return content

    def build_execution_payload(
        self, instance: dict[str, Any], submission: str
    ) -> dict[str, Any]:
        """Validate that the generated patch is syntactically valid.

        This is a lightweight check — it verifies the patch looks like a
        valid unified diff. Full repo-level test execution requires the
        official SWE-bench Docker harness.
        """
        gold_patch = instance.get("patch", "")

        verifier = textwrap.dedent(f"""\
            import sys, re

            submission = {submission!r}
            gold_patch = {gold_patch!r}

            # Check 1: submission is non-empty
            if not submission.strip():
                print("Empty patch", file=sys.stderr)
                sys.exit(1)

            # Check 2: looks like a unified diff
            has_minus = "---" in submission
            has_plus = "+++" in submission
            has_hunk = bool(re.search(r"^@@\\s", submission, re.MULTILINE))

            if not (has_minus and has_plus and has_hunk):
                print("Not a valid unified diff format", file=sys.stderr)
                sys.exit(1)

            # Check 3: modifies at least one file
            files_changed = re.findall(r"^\\+\\+\\+\\s+[ab]/(.+)", submission, re.MULTILINE)
            if not files_changed:
                print("No files changed in patch", file=sys.stderr)
                sys.exit(1)

            # Check 4: gold patch file overlap (if gold available)
            if gold_patch:
                gold_files = set(re.findall(
                    r"^\\+\\+\\+\\s+[ab]/(.+)", gold_patch, re.MULTILINE
                ))
                sub_files = set(files_changed)
                overlap = gold_files & sub_files
                if not overlap:
                    print(f"No file overlap with gold patch", file=sys.stderr)
                    print(f"  Gold files: {{sorted(gold_files)}}", file=sys.stderr)
                    print(f"  Submission files: {{sorted(sub_files)}}", file=sys.stderr)
                    sys.exit(1)

            print(f"Valid patch: {{len(files_changed)}} file(s) modified")
            sys.exit(0)
        """)

        return {"language": "python", "code": verifier, "timeout_seconds": 10}

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
                "task_id": instance.get("instance_id", instance.get("id", "")),
            },
        )
