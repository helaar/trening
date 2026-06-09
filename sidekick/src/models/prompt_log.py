from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, Field, field_validator

from utils.datetime_utils import ensure_utc


class PromptLogEntry(BaseModel):
    """A single LLM call or tool invocation captured during a crew run.

    Captures the literal messages/responses and tool inputs/outputs as emitted
    by CrewAI's event bus — ground truth for what an agent actually saw and did,
    in chronological order, so admins can trace the call hierarchy (which agent
    ran which task, when it called the LLM vs. a tool, and what came back).
    """

    run_id: str
    athlete_id: int
    crew_name: str
    kind: Literal["llm_call", "tool_call"] = "llm_call"
    agent_role: str | None = None
    task_name: str | None = None

    # llm_call fields
    model: str | None = None
    call_type: str | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    response: str | None = None

    # tool_call fields
    tool_name: str | None = None
    tool_args: dict[str, Any] | str | None = None
    tool_output: str | None = None
    tool_error: str | None = None

    created_at: AwareDatetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("created_at", mode="before")
    @classmethod
    def _utc(cls, v):
        return ensure_utc(v)
