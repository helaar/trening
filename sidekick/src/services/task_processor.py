import asyncio
import logging
from typing import Any
import uuid

from database.athlete_repository import AthleteRepository
from database.crew_definition_repository import CrewDefinitionRepository
from database.daily_analysis_repository import DailyAnalysisRepository
from database.daily_entry_repository import DailyEntryRepository
from database.memory_repository import MemoryRepository
from database.plan_repository import PlanRepository
from database.prompt_log_repository import PromptLogRepository
from database.task_repository import TaskRepository
from database.workout_repository import WorkoutRepository
from models.task import Task, TaskStatus, TaskType
from services.handlers import DailyAnalysisHandler, MemoryConsolidationHandler, TaskHandler, TrainingAnalysisHandler

logger = logging.getLogger(__name__)


class TaskProcessor:
    """Background task processor for long-running operations."""

    # Shared across instances: TaskProcessor is constructed per-request via DI,
    # but background tasks must be tracked for the lifetime of the process.
    _running_tasks: dict[str, asyncio.Task] = {}

    def __init__(
        self,
        task_repo: TaskRepository,
        athlete_repo: AthleteRepository,
        workout_repo: WorkoutRepository,
        plan_repo: PlanRepository,
        daily_analysis_repo: DailyAnalysisRepository,
        daily_entry_repo: DailyEntryRepository,
        memory_repo: MemoryRepository,
        crew_def_repo: CrewDefinitionRepository,
        prompt_log_repo: PromptLogRepository | None = None,
    ):
        self.task_repo = task_repo
        self._handlers: dict[TaskType, TaskHandler] = {
            TaskType.TRAINING_ANALYSIS: TrainingAnalysisHandler(task_repo, athlete_repo, workout_repo),
            TaskType.DAILY_LLM_ANALYSIS: DailyAnalysisHandler(task_repo, athlete_repo, workout_repo, plan_repo, daily_analysis_repo, daily_entry_repo, memory_repo, crew_def_repo, prompt_log_repo),
            TaskType.MEMORY_CONSOLIDATION: MemoryConsolidationHandler(task_repo, athlete_repo, memory_repo, daily_analysis_repo, crew_def_repo),
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

        except asyncio.CancelledError:
            logger.error("Task %s was cancelled", task_id)
            await self.task_repo.update_task_status(task_id, TaskStatus.FAILED, error="Task was cancelled")
            raise
        except Exception as e:
            logger.error("Task %s failed: %s", task_id, str(e), exc_info=True)
            await self.task_repo.update_task_status(task_id, TaskStatus.FAILED, error=str(e))

    def start_task_in_background(self, task_id: str) -> None:
        """Start task processing in the background."""
        task = asyncio.create_task(self.process_task(task_id))
        TaskProcessor._running_tasks[task_id] = task
        task.add_done_callback(lambda _: TaskProcessor._running_tasks.pop(task_id, None))
        logger.info("Task %s started in background", task_id)

    @staticmethod
    async def shutdown() -> None:
        """Cancel and await all in-flight background tasks during app shutdown."""
        tasks = list(TaskProcessor._running_tasks.values())
        if not tasks:
            return
        logger.info("Cancelling %d in-flight task(s) during shutdown", len(tasks))
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


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
