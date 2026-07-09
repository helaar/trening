from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum

from pydantic import AwareDatetime, BaseModel, Field, field_validator, model_validator

from utils.datetime_utils import ensure_utc

_RECENT_TTL_DAYS = 30
_LONG_TERM_TTL_DAYS = 365
_MAX_CONTENT_CHARS = 600


def clamp_memory_content(content: str, max_chars: int = _MAX_CONTENT_CHARS) -> str:
    """Truncate memory content to max_chars at a word boundary.

    Defensive backstop only, not a substitute for the 1-3 sentence rule enforced in
    the extraction/consolidation task prompts — this just bounds a single runaway
    LLM update from growing a memory's content unboundedly.
    """
    if len(content) <= max_chars:
        return content
    marker = " […truncated]"
    truncated = content[: max_chars - len(marker)].rsplit(" ", 1)[0]
    return truncated + marker


class MemoryScope(str, Enum):
    RECENT = "recent"
    LONG_TERM = "long_term"


class MemoryCategory(str, Enum):
    RECOVERY = "recovery"
    HABIT = "habit"
    PERFORMANCE = "performance"
    RISK = "risk"
    GOAL = "goal"


class Memory(BaseModel):
    memory_id: str
    athlete_id: int
    scope: MemoryScope
    category: MemoryCategory
    content: str = Field(description="1-3 sentence natural language observation")
    confidence: float = Field(ge=0.0, le=1.0)
    importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Standing significance to coaching, independent of evidence strength",
    )
    evidence_dates: list[str] = Field(default_factory=list, description="YYYY-MM-DD dates supporting this memory")
    created_at: AwareDatetime
    updated_at: AwareDatetime
    expires_at: AwareDatetime
    active: bool = True

    @field_validator("created_at", "updated_at", "expires_at", mode="before")
    @classmethod
    def _utc(cls, v):
        return ensure_utc(v)

    @model_validator(mode="before")
    @classmethod
    def compute_expires_at(cls, values: dict) -> dict:
        if "expires_at" not in values or values.get("expires_at") is None:
            scope = values.get("scope")
            updated_at = values.get("updated_at") or datetime.now(timezone.utc)
            ttl = _LONG_TERM_TTL_DAYS if scope == MemoryScope.LONG_TERM or scope == "long_term" else _RECENT_TTL_DAYS
            values["expires_at"] = updated_at + timedelta(days=ttl)
        return values

    def refresh_expiry(self) -> "Memory":
        ttl = _LONG_TERM_TTL_DAYS if self.scope == MemoryScope.LONG_TERM else _RECENT_TTL_DAYS
        return self.model_copy(update={"expires_at": self.updated_at + timedelta(days=ttl)})
