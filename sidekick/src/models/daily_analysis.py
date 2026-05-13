from datetime import datetime, timezone

from pydantic import BaseModel, Field

from models.crew_outputs import CoachingOutput, RestitutionAnalysisOutput, WorkoutAnalysisOutput


class DailyAnalysisResult(BaseModel):
    athlete_id: int
    date: str
    workout_analysis: WorkoutAnalysisOutput | None = None
    restitution_analysis: RestitutionAnalysisOutput | None = None
    coaching_feedback: CoachingOutput | None = None
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
