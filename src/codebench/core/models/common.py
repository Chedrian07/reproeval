"""Common models shared across the codebench framework."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class TrackType(StrEnum):
    LM_ONLY = "lm_only"
    STANDARD_TOOLS = "standard_tools"
    OPEN_AGENT = "open_agent"


class ScenarioType(StrEnum):
    REPO_PATCH = "repo_patch"
    CONTEST_CODEGEN = "contest_codegen"
    FUNCTION_CODEGEN = "function_codegen"
    CODE_REASONING = "code_reasoning"
    TERMINAL_AGENT = "terminal_agent"


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class ProviderRequest(BaseModel):
    """Provider-neutral request envelope."""

    prompt: str
    system_prompt: str | None = None
    temperature: float = 0.0
    max_tokens: int = 4096
    seed: int | None = None
    stop_sequences: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProviderResponse(BaseModel):
    """Provider-neutral response envelope."""

    content: str
    model: str
    usage: TokenUsage
    latency_ms: float
    raw_response: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TokenUsage(BaseModel):
    """Token usage from a provider call."""

    input_tokens: int
    output_tokens: int
    total_tokens: int | None = None

    def model_post_init(self, __context: Any) -> None:
        if self.total_tokens is None:
            self.total_tokens = self.input_tokens + self.output_tokens


class ExecutionResult(BaseModel):
    """Result from sandbox code execution."""

    exit_code: int
    stdout: str
    stderr: str
    duration_ms: float
    timed_out: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScoringResult(BaseModel):
    """Result from scoring/judging a benchmark instance."""

    score: float
    max_score: float = 1.0
    passed: bool
    details: dict[str, Any] = Field(default_factory=dict)
    judge_reasoning: str | None = None


class RunManifest(BaseModel):
    """Metadata for a complete benchmark run, enabling reproducibility and replay."""

    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    dataset_name: str
    dataset_version: str
    scenario_type: ScenarioType
    track_type: TrackType
    provider_name: str
    model_name: str
    harness_version: str
    prompt_version: str | None = None
    seed: int | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    status: RunStatus = RunStatus.PENDING
    total_instances: int = 0
    completed_instances: int = 0
    results_summary: dict[str, Any] = Field(default_factory=dict)


class InstanceResult(BaseModel):
    """Result for a single benchmark instance within a run."""

    instance_id: str
    run_id: str
    dataset_instance_id: str
    provider_response: ProviderResponse | None = None
    execution_result: ExecutionResult | None = None
    scoring_result: ScoringResult | None = None
    status: RunStatus = RunStatus.PENDING
    error: str | None = None
    artifacts_path: str | None = None
