from datetime import datetime, timezone

from pydantic import BaseModel, Field


class PromptConfig(BaseModel):
    key: str
    value: str
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PromptConfigUpdate(BaseModel):
    key: str
    value: str
