import asyncio
import logging
from typing import Any
import uuid

from database.athlete_repository import AthleteRepository
from database.task_repository import TaskRepository
from database.workout_repository import WorkoutRepository
from models.task import Task, TaskStatus, TaskType
from services.handlers import TaskHandler, TrainingAnalysisHandler

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
        self._handlers: dict[TaskType, TaskHandler] = {
            TaskType.TRAINING_ANALYSIS: TrainingAnalysisHandler(task_repo, athlete_repo, workout_repo),
        }

    async def process_task(self, task_id: str) -> None:
        """Process a task in the background."""
        try:
            task = await self.task_repo.get_task(task_id)
            if not task:
                logger.error("Task %s not found", task_id)
                return

            await self.task_repo.update_task_status(task_id, TaskStatus.RUNNING)

            handler = self._handlers.get(task.task_type)
            if not handler:
                raise ValueError(f"Unknown task type: {task.task_type}")

            result = await handler.execute(task_id, task.athlete_id, task.parameters)

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
