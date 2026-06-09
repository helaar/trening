from datetime import datetime, timezone

from pydantic import AwareDatetime, BaseModel, Field, field_validator

from utils.datetime_utils import ensure_utc

from models.crew_outputs import CoachingOutput, RestitutionAnalysisOutput, WorkoutAnalysisOutput


class DailyAnalysisResult(BaseModel):
    athlete_id: int
    date: str
    workout_analysis: WorkoutAnalysisOutput | None = None
    restitution_analysis: RestitutionAnalysisOutput | None = None
    coaching_feedback: CoachingOutput | None = None
    analyzed_at: AwareDatetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("analyzed_at", mode="before")
    @classmethod
    def _utc(cls, v):
        return ensure_utc(v)
