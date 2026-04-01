"""Docker-backed sandbox runner for executing untrusted code in isolation.

This module provides the primary production sandbox backend. All benchmark
code is treated as untrusted and executed inside a short-lived Docker
container with:

- Network disabled by default (--network=none)
- Memory limits enforced by cgroup
- Timeout enforcement via container stop + kill
- Read-only root filesystem where practical
- Mounted working directory scoped to a single temp dir
- Output truncation to prevent memory exhaustion in the harness
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
from typing import Any

import docker
import docker.errors

from codebench.core.config.settings import SandboxConfig
from codebench.core.interfaces.sandbox import SandboxRunner
from codebench.core.models.common import ExecutionResult

logger = logging.getLogger(__name__)

# Language-specific filename and run command mappings.
_LANGUAGE_MAP: dict[str, tuple[str, list[str]]] = {
    "python": ("solution.py", ["python", "/workspace/solution.py"]),
    "javascript": ("solution.js", ["node", "/workspace/solution.js"]),
    "bash": ("solution.sh", ["bash", "/workspace/solution.sh"]),
    "ruby": ("solution.rb", ["ruby", "/workspace/solution.rb"]),
}

_DEFAULT_MAX_OUTPUT_BYTES = 1_000_000


class DockerSandboxRunner(SandboxRunner):
    """Execute code inside a disposable Docker container.

    Security posture
    ----------------
    * Benchmark execution is **offline by default** (``network_enabled=False``
      translates to ``--network=none``).
    * All submitted code is treated as *untrusted*.
    * Memory and CPU-time limits are enforced at the container level.
    * The host filesystem is never exposed beyond a single temporary
      directory that is bind-mounted read-only where possible.
    * stdout / stderr are truncated to ``max_output_bytes`` to prevent
      a malicious program from inflating harness memory.
    """

    def __init__(self, config: SandboxConfig | None = None) -> None:
        self._config = config or SandboxConfig()
        self._client: docker.DockerClient | None = None

    # -- SandboxRunner interface ------------------------------------------------

    @property
    def name(self) -> str:
        return "docker"

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
        """Run *code* inside a Docker container and return structured results."""
        return await asyncio.to_thread(
            self._execute_sync,
            code,
            language=language,
            timeout_seconds=timeout_seconds,
            memory_limit_mb=memory_limit_mb,
            network_enabled=network_enabled,
            env=env,
            workdir_files=workdir_files,
        )

    async def cleanup(self) -> None:
        """Close the Docker client if it was opened."""
        if self._client is not None:
            self._client.close()
            self._client = None

    async def check_health(self) -> bool:
        """Return ``True`` when the Docker daemon is reachable."""
        try:
            client = self._get_client()
            return client.ping()
        except Exception:  # noqa: BLE001
            return False

    # -- Internal helpers -------------------------------------------------------

    def _get_client(self) -> docker.DockerClient:
        if self._client is None:
            self._client = docker.from_env()
        return self._client

    def _execute_sync(
        self,
        code: str,
        *,
        language: str,
        timeout_seconds: int,
        memory_limit_mb: int,
        network_enabled: bool,
        env: dict[str, str] | None,
        workdir_files: dict[str, str] | None,
    ) -> ExecutionResult:
        client = self._get_client()
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

        filename, run_cmd = _LANGUAGE_MAP[language]

        tmp_dir = tempfile.mkdtemp(prefix="codebench_")
        container = None
        try:
            # Write the solution file.
            solution_path = os.path.join(tmp_dir, filename)
            with open(solution_path, "w", encoding="utf-8") as fh:
                fh.write(code)

            # Write any additional workdir files, with path validation.
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
                    dest_dir = os.path.dirname(dest)
                    os.makedirs(dest_dir, exist_ok=True)
                    with open(dest, "w", encoding="utf-8") as fh:
                        fh.write(content)

            # Build container configuration.
            mem_limit = f"{memory_limit_mb}m"
            network_mode = "bridge" if network_enabled else "none"

            container_kwargs: dict[str, Any] = {
                "image": self._config.image,
                "command": run_cmd,
                "volumes": {tmp_dir: {"bind": "/workspace", "mode": "ro"}},
                "working_dir": "/workspace",
                "mem_limit": mem_limit,
                "network_mode": network_mode,
                "detach": True,
                "stdout": True,
                "stderr": True,
                "pids_limit": 256,
                "nano_cpus": 1_000_000_000,
                "cap_drop": ["ALL"],
                "read_only": True,
                "user": "nobody",
            }
            if env:
                container_kwargs["environment"] = env

            # ---- Run container ----
            t0 = time.monotonic()
            container = client.containers.run(**container_kwargs)

            try:
                result = container.wait(timeout=timeout_seconds)
                elapsed_ms = (time.monotonic() - t0) * 1000.0
                timed_out = False
            except Exception:  # noqa: BLE001  (requests.exceptions.* or similar)
                elapsed_ms = (time.monotonic() - t0) * 1000.0
                timed_out = True
                try:
                    container.stop(timeout=2)
                except Exception:  # noqa: BLE001
                    import contextlib

                    with contextlib.suppress(Exception):
                        container.kill()
                result = {"StatusCode": -1}

            exit_code: int = result.get("StatusCode", -1)

            # Capture logs.
            raw_stdout = container.logs(stdout=True, stderr=False)
            raw_stderr = container.logs(stdout=False, stderr=True)

            stdout = self._truncate(raw_stdout, max_output)
            stderr = self._truncate(raw_stderr, max_output)

            return ExecutionResult(
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                duration_ms=elapsed_ms,
                timed_out=timed_out,
                metadata={
                    "image": self._config.image,
                    "network_enabled": network_enabled,
                    "memory_limit_mb": memory_limit_mb,
                },
            )

        except docker.errors.ImageNotFound:
            return ExecutionResult(
                exit_code=1,
                stdout="",
                stderr=f"Docker image not found: {self._config.image}",
                duration_ms=0.0,
                timed_out=False,
                metadata={"error": "image_not_found"},
            )
        except docker.errors.APIError as exc:
            return ExecutionResult(
                exit_code=1,
                stdout="",
                stderr=f"Docker API error: {exc}",
                duration_ms=0.0,
                timed_out=False,
                metadata={"error": "docker_api_error"},
            )
        finally:
            # Remove the container.
            if container is not None:
                try:
                    container.remove(force=True)
                except Exception:  # noqa: BLE001
                    logger.warning("Failed to remove container %s", container.id)

            # Remove the temp dir.
            self._rm_tree(tmp_dir)

    # -- Utilities --------------------------------------------------------------

    @staticmethod
    def _truncate(data: bytes, max_bytes: int) -> str:
        """Decode and truncate output, appending a marker when truncated."""
        if len(data) > max_bytes:
            truncated = data[:max_bytes]
            suffix = f"\n... [truncated at {max_bytes} bytes, total {len(data)}]"
            return truncated.decode("utf-8", errors="replace") + suffix
        return data.decode("utf-8", errors="replace")

    @staticmethod
    def _rm_tree(path: str) -> None:
        """Best-effort recursive removal of *path*."""
        import shutil

        shutil.rmtree(path, ignore_errors=True)
