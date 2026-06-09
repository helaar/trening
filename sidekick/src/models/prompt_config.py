import re
from datetime import datetime, timezone

from pydantic import AwareDatetime, BaseModel, Field, field_validator

from utils.datetime_utils import ensure_utc

_VALID_LLM_RE = re.compile(
    r"^(anthropic/|openai/|azure/|bedrock/|vertex_ai/)?"
    r"(claude-|gpt-|o1-|o3-|o4-)"
)


class PromptConfig(BaseModel):
    key: str
    value: str
    updated_at: AwareDatetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("updated_at", mode="before")
    @classmethod
    def _utc(cls, v):
        return ensure_utc(v)


class PromptConfigUpdate(BaseModel):
    key: str
    value: str

    @field_validator("value", mode="after")
    @classmethod
    def validate_llm_model(cls, v: str, info) -> str:
        if str(info.data.get("key", "")).endswith(".llm_model"):
            if not _VALID_LLM_RE.match(v):
                raise ValueError(
                    f"Unrecognized llm_model '{v}'. "
                    "Expected a claude-* or gpt-*/o*-* model name, "
                    "optionally prefixed with 'anthropic/' or 'openai/'."
                )
        return v
