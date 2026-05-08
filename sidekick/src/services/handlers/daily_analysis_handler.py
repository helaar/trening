import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from crew.daily_analysis import DailyAnalysisInput, run_daily_analysis
from database.athlete_repository import AthleteRepository
from database.daily_analysis_repository import DailyAnalysisRepository
from database.plan_repository import PlanRepository
from database.task_repository import TaskRepository
from database.workout_repository import WorkoutRepository
from models.daily_analysis import DailyAnalysisResult
from services.handlers.base import TaskHandler

logger = logging.getLogger(__name__)


class DailyAnalysisHandler(TaskHandler):
    """Handler for daily LLM multi-agent analysis tasks."""

    def __init__(
        self,
        task_repo: TaskRepository,
        athlete_repo: AthleteRepository,
        workout_repo: WorkoutRepository,
        plan_repo: PlanRepository,
        daily_analysis_repo: DailyAnalysisRepository,
    ):
        self.task_repo = task_repo
        self.athlete_repo = athlete_repo
        self.workout_repo = workout_repo
        self.plan_repo = plan_repo
        self.daily_analysis_repo = daily_analysis_repo

    async def execute(
        self,
        task_id: str,
        athlete_id: int,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        date_str = parameters.get("date")
        if not date_str:
            raise ValueError("Parameter 'date' is required (YYYY-MM-DD)")

        logger.info("Starting daily LLM analysis for task %s, athlete %s, date %s", task_id, athlete_id, date_str)

        athlete = await self.athlete_repo.get_athlete(athlete_id)
        if not athlete:
            raise ValueError(f"Athlete {athlete_id} not found")
        await self.task_repo.update_task_progress(task_id, 0.1)

        activity_date = datetime.fromisoformat(date_str)
        workout_analyses = await self.workout_repo.get_analyses_for_date(athlete_id, activity_date)
        logger.info("Retrieved %d workout analyses for %s", len(workout_analyses), date_str)
        await self.task_repo.update_task_progress(task_id, 0.2)

        planned_activities = await self.plan_repo.get_for_date(athlete_id, date_str)
        logger.info("Retrieved %d planned activities for %s", len(planned_activities), date_str)
        await self.task_repo.update_task_progress(task_id, 0.3)

        analysis_input = DailyAnalysisInput(
            athlete=athlete,
            workout_analyses=workout_analyses,
            planned_activities=planned_activities,
            date=date_str,
        )

        crew_result = await asyncio.to_thread(run_daily_analysis, analysis_input)
        await self.task_repo.update_task_progress(task_id, 0.9)

        stored = DailyAnalysisResult(
            athlete_id=athlete_id,
            date=date_str,
            workout_analysis=crew_result["workout_analysis"],
            coaching_feedback=crew_result["coaching_feedback"],
        )
        await self.daily_analysis_repo.upsert(stored)
        logger.info("Stored daily analysis result for athlete %s on %s", athlete_id, date_str)

        return {
            "analysis_type": "daily_llm_analysis",
            "date": date_str,
            "workout_analysis": crew_result["workout_analysis"],
            "coaching_feedback": crew_result["coaching_feedback"],
            "completed_at": stored.analyzed_at.isoformat(),
        }
