from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class PromptLogEntry(BaseModel):
    """A single LLM call captured during a crew run, for admin inspection.

    Captures the literal messages sent to the model and the response received,
    as emitted by CrewAI's event bus — ground truth for what an agent actually saw.
    """

    run_id: str
    athlete_id: int
    crew_name: str
    agent_role: str | None = None
    task_name: str | None = None
    model: str | None = None
    call_type: str | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    response: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
