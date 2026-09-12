from datetime import datetime, timezone
from enum import Enum
from typing import Any
from pydantic import AwareDatetime, BaseModel, Field, field_validator
from utils.datetime_utils import ensure_utc


class TaskStatus(str, Enum):
    """Task status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskType(str, Enum):
    """Task type enumeration."""
    TRAINING_ANALYSIS = "training_analysis"
    DAILY_LLM_ANALYSIS = "daily_llm_analysis"
    MEMORY_CONSOLIDATION = "memory_consolidation"


class TaskStepStatus(str, Enum):
    """Status of a single named sub-step within a task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskStep(BaseModel):
    """A named, independently-tracked sub-step of a task (e.g. one crew task)."""
    key: str = Field(..., description="Stable identifier for this step")
    label: str = Field(..., description="Human-readable label shown in the UI")
    status: TaskStepStatus = Field(default=TaskStepStatus.PENDING)


class Task(BaseModel):
    """Model for asynchronous task tracking."""

    task_id: str = Field(..., description="Unique task identifier")
    athlete_id: int = Field(..., description="Athlete ID who initiated the task")
    task_type: TaskType = Field(..., description="Type of task")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current task status")
    progress: float = Field(default=0.0, description="Task progress (0.0 to 1.0)")
    steps: list[TaskStep] | None = Field(
        default=None, description="Named sub-steps for tasks that report granular progress"
    )
    parameters: dict[str, Any] = Field(default_factory=dict, description="Task input parameters")
    result: dict[str, Any] | None = Field(default=None, description="Task result data")
    error: str | None = Field(default=None, description="Error message if failed")
    created_at: AwareDatetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: AwareDatetime | None = Field(default=None, description="When task started running")
    completed_at: AwareDatetime | None = Field(default=None, description="When task completed/failed")

    @field_validator("created_at", "started_at", "completed_at", mode="before")
    @classmethod
    def _utc(cls, v):
        return ensure_utc(v)
    
    @property
    def duration_seconds(self) -> float | None:
        """Calculate task duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class TaskCreateRequest(BaseModel):
    """Request model for creating a new task."""
    task_type: TaskType = Field(..., description="Type of task to create")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Task parameters")


class TaskResponse(BaseModel):
    """Response model for task information."""
    task_id: str
    status: TaskStatus
    progress: float
    steps: list[TaskStep] | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: AwareDatetime
    started_at: AwareDatetime | None = None
    completed_at: AwareDatetime | None = None

    @field_validator("created_at", "started_at", "completed_at", mode="before")
    @classmethod
    def _utc(cls, v):
        return ensure_utc(v)
    duration_seconds: float | None = None
