"""Local-process sandbox runner -- FOR TESTING ONLY.

.. warning::

    **NOT SAFE FOR UNTRUSTED CODE.**

    This runner executes code directly on the host via ``subprocess.run``.
    It exists solely as a lightweight fallback so that the test-suite and
    local development workflows do not require a running Docker daemon.

    **Never** use this runner against code you do not fully trust.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
import time

from codebench.core.config.settings import SandboxConfig
from codebench.core.interfaces.sandbox import SandboxRunner
from codebench.core.models.common import ExecutionResult

# Language -> (filename, command-prefix list)
_LANGUAGE_MAP: dict[str, tuple[str, list[str]]] = {
    "python": ("solution.py", ["python3"]),
    "javascript": ("solution.js", ["node"]),
    "bash": ("solution.sh", ["bash"]),
    "ruby": ("solution.rb", ["ruby"]),
}

_DEFAULT_MAX_OUTPUT_BYTES = 1_000_000


class LocalProcessRunner(SandboxRunner):
    """Execute code in a local subprocess.

    .. danger::

        This runner provides **no isolation whatsoever**.  It is intended
        *only* for testing with trusted, fixture-level programs.  For any
        real benchmark workload use :class:`DockerSandboxRunner`.
    """

    def __init__(self, config: SandboxConfig | None = None) -> None:
        self._config = config or SandboxConfig(backend="local_process")

    @property
    def name(self) -> str:
        return "local_process"

    async def execute(
        self,
        code: str,
        *,
        language: str = "python",
        timeout_seconds: int = 30,
        memory_limit_mb: int = 512,
        network_enabled: bool = False,
        env: dict[str, str] | None = None,
        workdir_files: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """Run *code* in a subprocess and return structured results."""
        return await asyncio.to_thread(
            self._execute_sync,
            code,
            language=language,
            timeout_seconds=timeout_seconds,
            env=env,
            workdir_files=workdir_files,
        )

    async def cleanup(self) -> None:
        """No-op -- nothing to clean up for local processes."""

    async def check_health(self) -> bool:
        """Always healthy -- the host OS is assumed available."""
        return True

    # ------------------------------------------------------------------

    def _execute_sync(
        self,
        code: str,
        *,
        language: str,
        timeout_seconds: int,
        env: dict[str, str] | None,
        workdir_files: dict[str, str] | None,
    ) -> ExecutionResult:
        max_output = self._config.max_output_bytes

        if language not in _LANGUAGE_MAP:
            return ExecutionResult(
                exit_code=1,
                stdout="",
                stderr=f"Unsupported language: {language}",
                duration_ms=0.0,
                timed_out=False,
                metadata={"error": "unsupported_language"},
            )

        filename, cmd_prefix = _LANGUAGE_MAP[language]

        tmp_dir = tempfile.mkdtemp(prefix="codebench_local_")
        try:
            # Write solution.
            solution_path = os.path.join(tmp_dir, filename)
            with open(solution_path, "w", encoding="utf-8") as fh:
                fh.write(code)

            # Write extra files.
            if workdir_files:
                for rel_path, content in workdir_files.items():
                    dest = os.path.realpath(os.path.join(tmp_dir, rel_path))
                    real_tmp = os.path.realpath(tmp_dir)
                    if not dest.startswith(real_tmp + os.sep) and dest != real_tmp:
                        return ExecutionResult(
                            exit_code=1,
                            stdout="",
                            stderr=f"Path escapes sandbox: {rel_path}",
                            duration_ms=0.0,
                            timed_out=False,
                            metadata={"error": "path_escape"},
                        )
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    with open(dest, "w", encoding="utf-8") as fh:
                        fh.write(content)

            # Build a minimal environment to avoid leaking host secrets.
            # Include Python-related vars so installed packages are found.
            allowed_env_keys = (
                "PATH",
                "HOME",
                "LANG",
                "TMPDIR",
                "PYTHONPATH",
                "PYTHONHOME",
                "VIRTUAL_ENV",
            )
            minimal_env = {k: os.environ[k] for k in allowed_env_keys if k in os.environ}
            if env:
                run_env = {**minimal_env, **env}
            else:
                run_env = minimal_env

            cmd = [*cmd_prefix, solution_path]

            timed_out = False
            t0 = time.monotonic()
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    timeout=timeout_seconds,
                    cwd=tmp_dir,
                    env=run_env,
                )
                elapsed_ms = (time.monotonic() - t0) * 1000.0
                exit_code = proc.returncode
                raw_stdout = proc.stdout
                raw_stderr = proc.stderr
            except subprocess.TimeoutExpired:
                elapsed_ms = (time.monotonic() - t0) * 1000.0
                timed_out = True
                exit_code = -1
                raw_stdout = b""
                raw_stderr = b"Process timed out"

            stdout = self._truncate(raw_stdout, max_output)
            stderr = self._truncate(raw_stderr, max_output)

            return ExecutionResult(
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                duration_ms=elapsed_ms,
                timed_out=timed_out,
                metadata={"runner": "local_process"},
            )
        finally:
            import shutil

            shutil.rmtree(tmp_dir, ignore_errors=True)

    @staticmethod
    def _truncate(data: bytes, max_bytes: int) -> str:
        """Decode and optionally truncate output."""
        if len(data) > max_bytes:
            truncated = data[:max_bytes]
            suffix = f"\n... [truncated at {max_bytes} bytes, total {len(data)}]"
            return truncated.decode("utf-8", errors="replace") + suffix
        return data.decode("utf-8", errors="replace")
