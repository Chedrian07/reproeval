"""Unit tests for LocalProcessRunner.

These tests exercise real subprocess execution with safe, trusted fixture
programs.  No Docker daemon is required.
"""

from __future__ import annotations

import pytest

from codebench.core.config.settings import SandboxConfig
from codebench.core.models.common import ExecutionResult
from codebench.sandbox.docker.local_runner import LocalProcessRunner

# Re-use fixture programs.
from tests.fixtures.sandbox_programs import (
    EXIT_CODE_42,
    INFINITE_LOOP,
    LARGE_OUTPUT,
    SIMPLE_ADD,
    SIMPLE_HELLO,
    STDERR_OUTPUT,
    SYNTAX_ERROR,
)


def _make_runner(max_output_bytes: int = 1_000_000) -> LocalProcessRunner:
    config = SandboxConfig(backend="local_process", max_output_bytes=max_output_bytes)
    return LocalProcessRunner(config=config)


@pytest.mark.unit
class TestLocalRunnerName:
    def test_name(self) -> None:
        runner = LocalProcessRunner()
        assert runner.name == "local_process"


@pytest.mark.unit
class TestLocalSimpleExecution:
    async def test_hello_world(self) -> None:
        runner = _make_runner()
        result = await runner.execute(SIMPLE_HELLO)

        assert isinstance(result, ExecutionResult)
        assert result.exit_code == 0
        assert result.stdout.strip() == "hello world"
        assert result.timed_out is False

    async def test_add_function(self) -> None:
        runner = _make_runner()
        result = await runner.execute(SIMPLE_ADD)

        assert result.exit_code == 0
        assert result.stdout.strip() == "5"

    async def test_exit_code_42(self) -> None:
        runner = _make_runner()
        result = await runner.execute(EXIT_CODE_42)

        assert result.exit_code == 42


@pytest.mark.unit
class TestLocalTimeout:
    async def test_infinite_loop_times_out(self) -> None:
        runner = _make_runner()
        result = await runner.execute(INFINITE_LOOP, timeout_seconds=2)

        assert result.timed_out is True
        assert result.exit_code == -1
        assert result.duration_ms > 0


@pytest.mark.unit
class TestLocalStderr:
    async def test_syntax_error_produces_stderr(self) -> None:
        runner = _make_runner()
        result = await runner.execute(SYNTAX_ERROR)

        assert result.exit_code != 0
        assert "SyntaxError" in result.stderr

    async def test_stderr_capture(self) -> None:
        runner = _make_runner()
        result = await runner.execute(STDERR_OUTPUT)

        assert result.exit_code == 0
        assert "stdout line" in result.stdout
        assert "stderr line" in result.stderr


@pytest.mark.unit
class TestLocalOutputTruncation:
    async def test_large_output_truncated(self) -> None:
        runner = _make_runner(max_output_bytes=1024)
        result = await runner.execute(LARGE_OUTPUT)

        assert result.exit_code == 0
        assert "truncated" in result.stdout.lower()
        # Should start with the expected character.
        assert result.stdout.startswith("x" * 100)


@pytest.mark.unit
class TestLocalUnsupportedLanguage:
    async def test_unsupported_language(self) -> None:
        runner = _make_runner()
        result = await runner.execute("puts 'hi'", language="cobol")

        assert result.exit_code == 1
        assert "Unsupported language" in result.stderr


@pytest.mark.unit
class TestLocalPathValidation:
    async def test_traversal_blocked(self) -> None:
        runner = _make_runner()
        result = await runner.execute(
            SIMPLE_HELLO,
            workdir_files={"../../../etc/evil": "pwned"},
        )

        assert result.exit_code == 1
        assert "Path escapes sandbox" in result.stderr

    async def test_valid_subpath_accepted(self) -> None:
        runner = _make_runner()
        code = """\
with open("subdir/data.txt") as f:
    print(f.read().strip())
"""
        result = await runner.execute(
            code,
            workdir_files={"subdir/data.txt": "hello from file"},
        )
        assert result.exit_code == 0
        assert "hello from file" in result.stdout


@pytest.mark.unit
class TestLocalPathTraversalRealpath:
    """Path traversal via normpath-only tricks must be blocked (resolved via realpath)."""

    async def test_normpath_traversal_via_dot_segments(self) -> None:
        runner = _make_runner()

        result = await runner.execute(
            SIMPLE_HELLO,
            workdir_files={"sub/../../../etc/passwd": "pwned"},
        )

        assert result.exit_code == 1
        assert result.metadata.get("error") == "path_escape"

    async def test_absolute_path_rejected(self) -> None:
        runner = _make_runner()

        result = await runner.execute(
            SIMPLE_HELLO,
            workdir_files={"/etc/passwd": "pwned"},
        )

        assert result.exit_code == 1
        assert result.metadata.get("error") == "path_escape"


@pytest.mark.unit
class TestLocalHealthCheck:
    async def test_always_healthy(self) -> None:
        runner = _make_runner()
        assert await runner.check_health() is True


@pytest.mark.unit
class TestLocalCleanup:
    async def test_cleanup_is_noop(self) -> None:
        runner = _make_runner()
        # Should not raise.
        await runner.cleanup()


@pytest.mark.unit
class TestLocalEnvVars:
    async def test_env_vars_available(self) -> None:
        runner = _make_runner()
        code = """\
import os
print(os.environ.get("CODEBENCH_TEST_VAR", "NOT_SET"))
"""
        result = await runner.execute(code, env={"CODEBENCH_TEST_VAR": "it_works"})

        assert result.exit_code == 0
        assert "it_works" in result.stdout


@pytest.mark.unit
class TestLocalEnvIsolation:
    """The local runner must NOT inherit the full host environment."""

    async def test_host_secrets_not_leaked(self) -> None:
        """OPENAI_API_KEY (or any non-allowed var) must not be visible."""
        import os

        # Temporarily set a secret in the host env to prove it is filtered.
        sentinel = "__CODEBENCH_SECRET_TEST__"
        os.environ[sentinel] = "leaked"
        try:
            runner = _make_runner()
            code = f"""\
import os
print("{sentinel}" in os.environ)
"""
            result = await runner.execute(code)
            assert result.exit_code == 0
            assert result.stdout.strip() == "False"
        finally:
            os.environ.pop(sentinel, None)

    async def test_minimal_env_without_explicit_env(self) -> None:
        """Even when no env kwarg is passed, the runner uses a minimal env."""
        import os

        sentinel = "__CODEBENCH_MINIMAL_TEST__"
        os.environ[sentinel] = "should_not_appear"
        try:
            runner = _make_runner()
            code = f"""\
import os
print("{sentinel}" in os.environ)
"""
            result = await runner.execute(code)
            assert result.exit_code == 0
            assert result.stdout.strip() == "False"
        finally:
            os.environ.pop(sentinel, None)


@pytest.mark.unit
class TestLocalDuration:
    async def test_duration_is_positive(self) -> None:
        runner = _make_runner()
        result = await runner.execute(SIMPLE_HELLO)

        assert result.duration_ms > 0
