"""Typed agent / task / philosophy definitions stored in `crew_definitions`.

Each document carries a `type` discriminator and the fields specific to that
type. This collection is the single source of truth for crew prompt content —
bootstrapped from the legacy YAML by scripts/seed_crew_definitions.py and edited
via the admin interface. Identity is the (type, name) pair; for philosophies
`name` is the slug and `display_name` holds the human-facing label.
"""
import re
from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, Field, TypeAdapter, field_validator

from utils.datetime_utils import ensure_utc

_VALID_LLM_RE = re.compile(
    r"^(anthropic/|openai/|azure/|bedrock/|vertex_ai/)?"
    r"(claude-|gpt-|o1-|o3-|o4-)"
)


class _Base(BaseModel):
    name: str
    updated_at: AwareDatetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("updated_at", mode="before")
    @classmethod
    def _utc(cls, v):
        return ensure_utc(v)


class AgentDoc(_Base):
    type: Literal["agent"] = "agent"
    role: str
    goal: str
    backstory: str
    llm_model: str

    @field_validator("llm_model")
    @classmethod
    def _validate_llm(cls, v: str) -> str:
        if not _VALID_LLM_RE.match(v):
            raise ValueError(
                f"Unrecognized llm_model '{v}'. Expected a claude-* or gpt-*/o*-* "
                "model name, optionally prefixed with 'anthropic/' or 'openai/'."
            )
        return v


class TaskDoc(_Base):
    type: Literal["task"] = "task"
    description: str
    expected_output: str


class PhilosophyDoc(_Base):
    type: Literal["philosophy"] = "philosophy"
    display_name: str
    intensity_targets: str = ""
    coach_guidance: str = ""
    analyst_guidance: str = ""


CrewDefinition = Annotated[
    AgentDoc | TaskDoc | PhilosophyDoc,
    Field(discriminator="type"),
]

crew_definition_adapter: TypeAdapter[CrewDefinition] = TypeAdapter(CrewDefinition)
