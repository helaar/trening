"""Pydantic models for coach role definitions."""

from pydantic import BaseModel, Field

class Coach(BaseModel):
    """Model for a single coach role definition."""
    role: str
    goal: str
    backstory: str
    llm_model: str | None = None
    reasoning_steps: int | None = None

class TaskDescription(BaseModel):
    """Class representing a task description loaded from YAML."""
    description: str
    expected_output: str 
    markdown: bool | None = False
    output_file: str | None = None

class CommonKnowledge(BaseModel):

    rule: str
    accept: str
    reject: str


class Plan(BaseModel):
    date: str
    activities: list[str] = Field(default_factory=list)