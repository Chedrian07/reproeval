"""Unit tests for DockerSandboxRunner.

All Docker interactions are mocked -- these tests do **not** require a
running Docker daemon.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from codebench.core.config.settings import SandboxConfig
from codebench.core.models.common import ExecutionResult
from codebench.sandbox.docker.runner import DockerSandboxRunner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_container(
    stdout: bytes = b"",
    stderr: bytes = b"",
    exit_code: int = 0,
    wait_side_effect: Exception | None = None,
) -> MagicMock:
    """Return a mock Docker container with sensible defaults."""
    container = MagicMock()
    container.id = "mock_container_id"

    if wait_side_effect is not None:
        container.wait.side_effect = wait_side_effect
    else:
        container.wait.return_value = {"StatusCode": exit_code}

    def _logs(*, stdout: bool = True, stderr: bool = True) -> bytes:  # noqa: FBT001,FBT002
        # The real API separates stdout/stderr via boolean flags.
        if stdout and not stderr:
            return _make_mock_container._stdout  # type: ignore[attr-defined]
        if stderr and not stdout:
            return _make_mock_container._stderr  # type: ignore[attr-defined]
        return _make_mock_container._stdout + _make_mock_container._stderr  # type: ignore[attr-defined]

    # Stash on the factory so the closure can read them.
    _make_mock_container._stdout = stdout  # type: ignore[attr-defined]
    _make_mock_container._stderr = stderr  # type: ignore[attr-defined]

    container.logs = MagicMock(side_effect=_logs)
    container.stop = MagicMock()
    container.kill = MagicMock()
    container.remove = MagicMock()
    return container


def _make_runner(config: SandboxConfig | None = None) -> DockerSandboxRunner:
    """Build a runner with a mocked Docker client."""
    runner = DockerSandboxRunner(config=config)
    mock_client = MagicMock()
    mock_client.ping.return_value = True
    runner._client = mock_client
    return runner


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDockerSandboxRunnerName:
    def test_name_is_docker(self) -> None:
        runner = DockerSandboxRunner()
        assert runner.name == "docker"


@pytest.mark.unit
class TestDockerExecute:
    """Tests for execute() with mocked container runs."""

    async def test_simple_execution(self) -> None:
        runner = _make_runner()
        container = _make_mock_container(stdout=b"hello world\n", exit_code=0)
        runner._client.containers.run.return_value = container  # type: ignore[union-attr]

        result = await runner.execute('print("hello world")')

        assert isinstance(result, ExecutionResult)
        assert result.exit_code == 0
        assert "hello world" in result.stdout
        assert result.timed_out is False
        container.remove.assert_called_once_with(force=True)

    async def test_nonzero_exit_code(self) -> None:
        runner = _make_runner()
        container = _make_mock_container(stderr=b"error\n", exit_code=1)
        runner._client.containers.run.return_value = container  # type: ignore[union-attr]

        result = await runner.execute("bad code")

        assert result.exit_code == 1
        assert "error" in result.stderr

    async def test_unsupported_language(self) -> None:
        runner = _make_runner()

        result = await runner.execute("code", language="cobol")

        assert result.exit_code == 1
        assert "Unsupported language" in result.stderr
        assert result.metadata.get("error") == "unsupported_language"


@pytest.mark.unit
class TestDockerTimeout:
    """Timeout enforcement."""

    async def test_timeout_produces_timed_out_result(self) -> None:
        runner = _make_runner()
        container = _make_mock_container(
            wait_side_effect=Exception("timeout"),
        )
        # After timeout the runner reads logs from the container.
        container.logs.side_effect = None
        container.logs.return_value = b""
        runner._client.containers.run.return_value = container  # type: ignore[union-attr]

        result = await runner.execute("while True: pass", timeout_seconds=1)

        assert result.timed_out is True
        assert result.exit_code == -1
        # Container should have been stopped.
        container.stop.assert_called_once()


@pytest.mark.unit
class TestDockerMemoryLimit:
    """Memory limit is forwarded to Docker."""

    async def test_memory_limit_passed_to_container(self) -> None:
        runner = _make_runner()
        container = _make_mock_container(stdout=b"ok\n")
        runner._client.containers.run.return_value = container  # type: ignore[union-attr]

        await runner.execute("print('ok')", memory_limit_mb=256)

        call_kwargs = runner._client.containers.run.call_args  # type: ignore[union-attr]
        assert call_kwargs.kwargs["mem_limit"] == "256m"


@pytest.mark.unit
class TestDockerNetworkDisabled:
    """Network must be disabled by default."""

    async def test_network_none_by_default(self) -> None:
        runner = _make_runner()
        container = _make_mock_container(stdout=b"ok\n")
        runner._client.containers.run.return_value = container  # type: ignore[union-attr]

        await runner.execute("print('ok')")

        call_kwargs = runner._client.containers.run.call_args  # type: ignore[union-attr]
        assert call_kwargs.kwargs["network_mode"] == "none"

    async def test_network_bridge_when_enabled(self) -> None:
        runner = _make_runner()
        container = _make_mock_container(stdout=b"ok\n")
        runner._client.containers.run.return_value = container  # type: ignore[union-attr]

        await runner.execute("print('ok')", network_enabled=True)

        call_kwargs = runner._client.containers.run.call_args  # type: ignore[union-attr]
        assert call_kwargs.kwargs["network_mode"] == "bridge"


@pytest.mark.unit
class TestDockerPathValidation:
    """Path traversal attacks must be blocked."""

    async def test_path_traversal_rejected(self) -> None:
        runner = _make_runner()

        result = await runner.execute(
            "print('hi')",
            workdir_files={"../../etc/passwd": "pwned"},
        )

        assert result.exit_code == 1
        assert "Path escapes sandbox" in result.stderr
        assert result.metadata.get("error") == "path_escape"

    async def test_normal_subpath_accepted(self) -> None:
        runner = _make_runner()
        container = _make_mock_container(stdout=b"ok\n")
        runner._client.containers.run.return_value = container  # type: ignore[union-attr]

        result = await runner.execute(
            "print('ok')",
            workdir_files={"data/input.txt": "hello"},
        )

        assert result.exit_code == 0


@pytest.mark.unit
class TestDockerOutputTruncation:
    """Large outputs must be truncated."""

    async def test_stdout_truncated(self) -> None:
        config = SandboxConfig(max_output_bytes=1024)
        runner = _make_runner(config=config)

        big_output = b"x" * 5000
        container = _make_mock_container(stdout=big_output)

        # Override logs to return the big output for stdout.
        def logs_fn(*, stdout: bool = True, stderr: bool = True) -> bytes:  # noqa: FBT001,FBT002
            if stdout and not stderr:
                return big_output
            if stderr and not stdout:
                return b""
            return big_output

        container.logs = MagicMock(side_effect=logs_fn)
        runner._client.containers.run.return_value = container  # type: ignore[union-attr]

        result = await runner.execute("print('x' * 5000)")

        # The stdout should be truncated and contain the truncation marker.
        assert "truncated" in result.stdout.lower()
        # The first 1024 bytes should still be present.
        assert result.stdout.startswith("x" * 1024)


@pytest.mark.unit
class TestDockerCleanup:
    """Cleanup releases Docker client."""

    async def test_cleanup_closes_client(self) -> None:
        runner = _make_runner()
        mock_client = runner._client

        await runner.cleanup()

        mock_client.close.assert_called_once()  # type: ignore[union-attr]
        assert runner._client is None

    async def test_cleanup_noop_when_no_client(self) -> None:
        runner = DockerSandboxRunner()
        # Should not raise.
        await runner.cleanup()


@pytest.mark.unit
class TestDockerHealthCheck:
    """Health check verifies Docker daemon."""

    async def test_healthy_when_ping_succeeds(self) -> None:
        runner = _make_runner()
        assert await runner.check_health() is True

    async def test_unhealthy_when_ping_fails(self) -> None:
        runner = DockerSandboxRunner()
        mock_client = MagicMock()
        mock_client.ping.side_effect = ConnectionError("daemon not running")
        runner._client = mock_client

        assert await runner.check_health() is False


@pytest.mark.unit
class TestDockerImageNotFound:
    """Image-not-found error is handled gracefully."""

    async def test_image_not_found_returns_error_result(self) -> None:
        import docker.errors

        runner = _make_runner()
        runner._client.containers.run.side_effect = docker.errors.ImageNotFound(  # type: ignore[union-attr]
            "not found"
        )

        result = await runner.execute("print('hi')")

        assert result.exit_code == 1
        assert "image not found" in result.stderr.lower()
        assert result.metadata.get("error") == "image_not_found"


@pytest.mark.unit
class TestDockerPathTraversalRealpath:
    """Path traversal via normpath-only tricks must be blocked (resolved via realpath)."""

    async def test_normpath_traversal_via_dot_segments(self) -> None:
        """normpath resolves '..' but does not follow symlinks; realpath does both."""
        runner = _make_runner()

        # A path that normpath might consider valid if the tmp dir name
        # prefix-matches another directory, e.g. /tmp/codebench_X/../../../etc/passwd
        result = await runner.execute(
            "print('hi')",
            workdir_files={"sub/../../../etc/passwd": "pwned"},
        )

        assert result.exit_code == 1
        assert result.metadata.get("error") == "path_escape"

    async def test_absolute_path_rejected(self) -> None:
        runner = _make_runner()

        result = await runner.execute(
            "print('hi')",
            workdir_files={"/etc/passwd": "pwned"},
        )

        assert result.exit_code == 1
        assert result.metadata.get("error") == "path_escape"


@pytest.mark.unit
class TestDockerContainerHardening:
    """Container kwargs must include security hardening flags."""

    async def test_hardening_kwargs_present(self) -> None:
        runner = _make_runner()
        container = _make_mock_container(stdout=b"ok\n")
        runner._client.containers.run.return_value = container  # type: ignore[union-attr]

        await runner.execute("print('ok')")

        call_kwargs = runner._client.containers.run.call_args  # type: ignore[union-attr]
        assert call_kwargs.kwargs["pids_limit"] == 256
        assert call_kwargs.kwargs["nano_cpus"] == 1_000_000_000
        assert call_kwargs.kwargs["cap_drop"] == ["ALL"]
        assert call_kwargs.kwargs["read_only"] is True
        assert call_kwargs.kwargs["user"] == "nobody"


@pytest.mark.unit
class TestDockerEnvVars:
    """Environment variables are passed to the container."""

    async def test_env_vars_forwarded(self) -> None:
        runner = _make_runner()
        container = _make_mock_container(stdout=b"ok\n")
        runner._client.containers.run.return_value = container  # type: ignore[union-attr]

        await runner.execute(
            "import os; print(os.environ['FOO'])",
            env={"FOO": "bar"},
        )

        call_kwargs = runner._client.containers.run.call_args  # type: ignore[union-attr]
        assert call_kwargs.kwargs["environment"] == {"FOO": "bar"}
