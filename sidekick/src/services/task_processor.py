import asyncio
import logging
from datetime import datetime, timezone
from typing import Any
import uuid

from database.athlete_repository import AthleteRepository
from database.task_repository import TaskRepository
from database.workout_repository import WorkoutRepository
from models.task import Task, TaskStatus, TaskType

logger = logging.getLogger(__name__)


class TaskProcessor:
    """Background task processor for long-running operations."""

    def __init__(
        self,
        task_repo: TaskRepository,
        athlete_repo: AthleteRepository,
        workout_repo: WorkoutRepository,
    ):
        self.task_repo = task_repo
        self.athlete_repo = athlete_repo
        self.workout_repo = workout_repo

    async def execute_training_analysis(
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

    async def process_task(self, task_id: str) -> None:
        """Process a task in the background."""
        try:
            task = await self.task_repo.get_task(task_id)
            if not task:
                logger.error("Task %s not found", task_id)
                return

            await self.task_repo.update_task_status(task_id, TaskStatus.RUNNING)

            if task.task_type == TaskType.TRAINING_ANALYSIS:
                result = await self.execute_training_analysis(task_id, task.athlete_id, task.parameters)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")

            await self.task_repo.update_task_result(task_id, result)
            await self.task_repo.update_task_progress(task_id, 1.0)
            await self.task_repo.update_task_status(task_id, TaskStatus.COMPLETED)

            logger.info("Task %s completed successfully", task_id)

        except Exception as e:
            logger.error("Task %s failed: %s", task_id, str(e), exc_info=True)
            await self.task_repo.update_task_status(task_id, TaskStatus.FAILED, error=str(e))

    def start_task_in_background(self, task_id: str) -> None:
        """Start task processing in the background."""
        asyncio.create_task(self.process_task(task_id))
        logger.info("Task %s started in background", task_id)


async def create_task(
    task_repo: TaskRepository,
    athlete_id: int,
    task_type: TaskType,
    parameters: dict[str, Any],
) -> Task:
    """Create a new task and return it."""
    task = Task(
        task_id=str(uuid.uuid4()),
        athlete_id=athlete_id,
        task_type=task_type,
        parameters=parameters,
    )

    await task_repo.create_task(task)
    logger.info("Created task %s of type %s for athlete %s", task.task_id, task_type, athlete_id)

    return task
