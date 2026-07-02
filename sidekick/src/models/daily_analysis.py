from datetime import datetime, timezone
from typing import Any

from pydantic import AwareDatetime, BaseModel, Field, field_validator

from utils.datetime_utils import ensure_utc

from models.crew_outputs import CoachingOutput, RestitutionAnalysisOutput, WorkoutAnalysisOutput


class DailyAnalysisResult(BaseModel):
    athlete_id: int
    date: str
    workout_analysis: WorkoutAnalysisOutput | None = None
    restitution_analysis: RestitutionAnalysisOutput | None = None
    coaching_feedback: CoachingOutput | None = None
    # Deterministic weekly polarization verdict (polarized philosophy only). Plain dict as
    # produced by _weekly_philosophy_assessment; None when no philosophy / not polarized.
    weekly_philosophy_assessment: dict[str, Any] | None = None
    analyzed_at: AwareDatetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("analyzed_at", mode="before")
    @classmethod
    def _utc(cls, v):
        return ensure_utc(v)
