from datetime import datetime, timezone
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Task status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskType(str, Enum):
    """Task type enumeration."""
    TRAINING_ANALYSIS = "training_analysis"


class Task(BaseModel):
    """Model for asynchronous task tracking."""
    
    task_id: str = Field(..., description="Unique task identifier")
    athlete_id: int = Field(..., description="Athlete ID who initiated the task")
    task_type: TaskType = Field(..., description="Type of task")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current task status")
    progress: float = Field(default=0.0, description="Task progress (0.0 to 1.0)")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Task input parameters")
    result: dict[str, Any] | None = Field(default=None, description="Task result data")
    error: str | None = Field(default=None, description="Error message if failed")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = Field(default=None, description="When task started running")
    completed_at: datetime | None = Field(default=None, description="When task completed/failed")
    
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
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_seconds: float | None = None
