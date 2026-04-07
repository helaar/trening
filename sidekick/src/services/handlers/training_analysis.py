import logging
from datetime import datetime, timezone
from typing import Any

from database.athlete_repository import AthleteRepository
from database.task_repository import TaskRepository
from database.workout_repository import WorkoutRepository
from services.handlers.base import TaskHandler

logger = logging.getLogger(__name__)


class TrainingAnalysisHandler(TaskHandler):
    """Handler for training analysis tasks."""

    def __init__(
        self,
        task_repo: TaskRepository,
        athlete_repo: AthleteRepository,
        workout_repo: WorkoutRepository,
    ):
        self.task_repo = task_repo
        self.athlete_repo = athlete_repo
        self.workout_repo = workout_repo

    async def execute(
        self,
        task_id: str,
        athlete_id: int,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute training analysis task.

        Step 1: retrieve athlete information from the database.
        Step 2: retrieve stored workouts for the requested date.
        Subsequent steps (CrewAI analysis) will be added incrementally.
        """
        logger.info("Starting training analysis for task %s, athlete %s", task_id, athlete_id)

        # Step 1 – athlete information
        athlete = await self.athlete_repo.get_athlete(athlete_id)
        if not athlete:
            raise ValueError(f"Athlete {athlete_id} not found")

        await self.task_repo.update_task_progress(task_id, 0.1)
        logger.info("Retrieved athlete info for %s %s", athlete.firstname, athlete.lastname)

        # Step 2 – workout analyses for the requested date
        date_str = parameters.get("date")
        if not date_str:
            raise ValueError("Parameter 'date' is required (YYYY-MM-DD)")

        activity_date = datetime.fromisoformat(date_str)
        analyses = await self.workout_repo.get_analyses_for_date(athlete_id, activity_date)
        logger.info("Retrieved %d workout analyses for %s", len(analyses), date_str)

        await self.task_repo.update_task_progress(task_id, 0.2)

        return {
            "analysis_type": "training_analysis",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "athlete": athlete.model_dump(mode="json"),
            "activities": analyses,
            "parameters_used": parameters,
        }
