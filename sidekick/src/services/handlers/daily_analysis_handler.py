import asyncio
import logging
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any

from crew.daily_analysis import DailyAnalysisInput, run_daily_analysis
from database.athlete_repository import AthleteRepository
from database.crew_definition_repository import CrewDefinitionRepository
from database.daily_analysis_repository import DailyAnalysisRepository
from database.daily_entry_repository import DailyEntryRepository
from database.memory_repository import MemoryRepository
from database.plan_repository import PlanRepository
from database.prompt_log_repository import PromptLogRepository
from database.task_repository import TaskRepository
from database.workout_repository import WorkoutRepository
from models.daily_analysis import DailyAnalysisResult
from models.memory import Memory, MemoryScope
from services.handlers.base import TaskHandler

logger = logging.getLogger(__name__)

_RESTITUTION_WINDOW_DAYS = 14


class DailyAnalysisHandler(TaskHandler):
    """Handler for daily LLM multi-agent analysis tasks."""

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
        self.athlete_repo = athlete_repo
        self.workout_repo = workout_repo
        self.plan_repo = plan_repo
        self.daily_analysis_repo = daily_analysis_repo
        self.daily_entry_repo = daily_entry_repo
        self.memory_repo = memory_repo
        self.crew_def_repo = crew_def_repo
        self.prompt_log_repo = prompt_log_repo

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
        restitution_start = (
            date.fromisoformat(date_str) - timedelta(days=_RESTITUTION_WINDOW_DAYS - 1)
        ).isoformat()

        agents = {a.name: a for a in await self.crew_def_repo.get_by_type("agent")}
        tasks = {t.name: t for t in await self.crew_def_repo.get_by_type("task")}
        philosophy = None
        if athlete.settings.training_philosophy:
            philosophy = await self.crew_def_repo.get_philosophy(athlete.settings.training_philosophy)

        (
            workout_analyses,
            planned_activities,
            daily_entries,
            recent_workout_analyses,
            active_memories,
            upcoming_races,
        ) = await asyncio.gather(
            self.workout_repo.get_analyses_for_date(athlete_id, activity_date),
            self.plan_repo.get_for_date(athlete_id, date_str),
            self.daily_entry_repo.get_range(athlete_id, restitution_start, date_str),
            self.workout_repo.get_analyses_for_range(
                athlete_id,
                datetime.fromisoformat(restitution_start),
                activity_date,
            ),
            self.memory_repo.get_active(athlete_id),
            self.plan_repo.get_races_from(athlete_id, date_str),
        )
        logger.info(
            "Retrieved %d workout analyses, %d plans, %d daily entries, "
            "%d recent analyses, %d memories, %d upcoming races for %s",
            len(workout_analyses),
            len(planned_activities),
            len(daily_entries),
            len(recent_workout_analyses),
            len(active_memories),
            len(upcoming_races),
            date_str,
        )
        await self.task_repo.update_task_progress(task_id, 0.3)

        analysis_input = DailyAnalysisInput(
            athlete=athlete,
            workout_analyses=workout_analyses,
            planned_activities=planned_activities,
            date=date_str,
            daily_entries=daily_entries,
            recent_workout_analyses=recent_workout_analyses,
            active_memories=active_memories,
            upcoming_races=upcoming_races,
            agents=agents,
            tasks=tasks,
            philosophy=philosophy,
        )

        crew_result = await asyncio.to_thread(run_daily_analysis, analysis_input)
        await self.task_repo.update_task_progress(task_id, 0.9)

        if self.prompt_log_repo:
            try:
                await self.prompt_log_repo.insert_many(crew_result.get("prompt_log_entries", []))
            except Exception:
                logger.exception("Failed to persist prompt log entries for task %s", task_id)

        stored = DailyAnalysisResult(
            athlete_id=athlete_id,
            date=date_str,
            workout_analysis=crew_result["workout_analysis"],
            restitution_analysis=crew_result["restitution_analysis"],
            coaching_feedback=crew_result["coaching_feedback"],
        )
        await self.daily_analysis_repo.upsert(stored)
        logger.info("Stored daily analysis result for athlete %s on %s", athlete_id, date_str)

        await self._apply_memory_extraction(athlete_id, date_str, crew_result.get("memory_extraction"), active_memories)

        return {
            "analysis_type": "daily_llm_analysis",
            "date": date_str,
            "workout_analysis": stored.workout_analysis.model_dump() if stored.workout_analysis else None,
            "restitution_analysis": stored.restitution_analysis.model_dump() if stored.restitution_analysis else None,
            "coaching_feedback": stored.coaching_feedback.model_dump() if stored.coaching_feedback else None,
            "completed_at": stored.analyzed_at.isoformat(),
        }

    async def _apply_memory_extraction(
        self,
        athlete_id: int,
        date_str: str,
        extraction,
        existing_memories: list[Memory],
    ) -> None:
        from models.crew_outputs import MemoryExtractionOutput
        if not isinstance(extraction, MemoryExtractionOutput):
            logger.warning("Memory extraction output missing or invalid for athlete %s on %s", athlete_id, date_str)
            return

        existing_by_id = {m.memory_id: m for m in existing_memories}
        now = datetime.now(timezone.utc)

        for draft in extraction.new_memories:
            memory = Memory(
                memory_id=str(uuid.uuid4()),
                athlete_id=athlete_id,
                scope=MemoryScope(draft.scope),
                category=draft.category,
                content=draft.content,
                confidence=draft.confidence,
                importance=draft.importance,
                evidence_dates=draft.evidence_dates,
                created_at=now,
                updated_at=now,
                expires_at=None,  # computed by model_validator
            )
            await self.memory_repo.upsert(memory)

        for update in extraction.updated_memories:
            existing = existing_by_id.get(update.memory_id)
            if not existing:
                logger.warning("Memory update targets unknown memory_id=%s", update.memory_id)
                continue
            updated = existing.model_copy(update={
                "content": update.content,
                "confidence": update.confidence,
                "importance": update.importance,
                "evidence_dates": update.evidence_dates,
                "updated_at": now,
            }).refresh_expiry()
            await self.memory_repo.upsert(updated)

        for memory_id in extraction.deactivated_memory_ids:
            await self.memory_repo.deactivate(memory_id)

        logger.info(
            "Memory extraction applied for athlete %s: +%d new, %d updated, %d deactivated",
            athlete_id, len(extraction.new_memories), len(extraction.updated_memories), len(extraction.deactivated_memory_ids),
        )
