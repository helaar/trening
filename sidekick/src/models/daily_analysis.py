from datetime import datetime, timezone
from pydantic import BaseModel, Field


class DailyAnalysisResult(BaseModel):
    athlete_id: int
    date: str
    workout_analysis: str
    coaching_feedback: str
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
