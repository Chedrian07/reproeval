"""Tests for configuration models."""

import pytest
from pydantic import ValidationError

from codebench.core.config.settings import (
    DatasetConfig,
    ProviderConfig,
    RunConfig,
    SandboxConfig,
)
from codebench.core.models.common import ScenarioType, TrackType


@pytest.mark.unit
class TestProviderConfig:
    def test_defaults(self) -> None:
        cfg = ProviderConfig(name="test", model="gpt-4")
        assert cfg.temperature == 0.0
        assert cfg.max_tokens == 40960
        assert cfg.api_key_env == ""
        assert cfg.base_url is None

    def test_custom(self) -> None:
        cfg = ProviderConfig(
            name="custom",
            model="claude-3",
            api_key_env="MY_KEY",
            base_url="https://api.example.com",
            temperature=0.7,
        )
        assert cfg.base_url == "https://api.example.com"


@pytest.mark.unit
class TestSandboxConfig:
    def test_defaults(self) -> None:
        cfg = SandboxConfig()
        assert cfg.backend == "docker"
        assert cfg.timeout_seconds == 30
        assert cfg.network_enabled is False
        assert cfg.image == "codebench-sandbox"

    def test_custom(self) -> None:
        cfg = SandboxConfig(timeout_seconds=60, memory_limit_mb=1024, network_enabled=True)
        assert cfg.timeout_seconds == 60
        assert cfg.network_enabled is True


@pytest.mark.unit
class TestSandboxConfigBounds:
    """Field constraints must reject out-of-range values."""

    def test_timeout_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SandboxConfig(timeout_seconds=0)

    def test_timeout_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SandboxConfig(timeout_seconds=-1)

    def test_timeout_too_large_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SandboxConfig(timeout_seconds=601)

    def test_memory_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SandboxConfig(memory_limit_mb=0)

    def test_memory_below_minimum_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SandboxConfig(memory_limit_mb=32)

    def test_memory_above_maximum_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SandboxConfig(memory_limit_mb=20000)

    def test_max_output_below_minimum_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SandboxConfig(max_output_bytes=100)

    def test_max_output_above_maximum_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SandboxConfig(max_output_bytes=200_000_000)

    def test_valid_boundary_values_accepted(self) -> None:
        cfg = SandboxConfig(timeout_seconds=1, memory_limit_mb=64, max_output_bytes=1024)
        assert cfg.timeout_seconds == 1
        assert cfg.memory_limit_mb == 64
        assert cfg.max_output_bytes == 1024

    def test_valid_upper_boundary_values_accepted(self) -> None:
        cfg = SandboxConfig(
            timeout_seconds=600, memory_limit_mb=16384, max_output_bytes=100_000_000
        )
        assert cfg.timeout_seconds == 600
        assert cfg.memory_limit_mb == 16384
        assert cfg.max_output_bytes == 100_000_000


@pytest.mark.unit
class TestRunConfig:
    def test_minimal(self) -> None:
        cfg = RunConfig(
            provider=ProviderConfig(name="test", model="gpt-4"),
            dataset=DatasetConfig(name="humaneval_plus", version="1.0"),
            scenario_type=ScenarioType.FUNCTION_CODEGEN,
        )
        assert cfg.track_type == TrackType.LM_ONLY
        assert cfg.dry_run is False
        assert cfg.harness_version == "0.1.0"

    def test_serialization(self) -> None:
        cfg = RunConfig(
            provider=ProviderConfig(name="test", model="gpt-4"),
            dataset=DatasetConfig(name="test", version="1.0"),
            scenario_type=ScenarioType.REPO_PATCH,
            track_type=TrackType.STANDARD_TOOLS,
            seed=42,
        )
        data = cfg.model_dump(mode="json")
        cfg2 = RunConfig.model_validate(data)
        assert cfg2.seed == 42
        assert cfg2.track_type == TrackType.STANDARD_TOOLS
