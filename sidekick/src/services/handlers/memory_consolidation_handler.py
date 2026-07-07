import asyncio
import json
import logging
import os
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any

from crewai import Crew, Task
from crewai.tools import BaseTool

from database.athlete_repository import AthleteRepository
from database.crew_definition_repository import CrewDefinitionRepository
from database.daily_analysis_repository import DailyAnalysisRepository
from database.memory_repository import MemoryRepository
from database.prompt_log_repository import PromptLogRepository
from database.task_repository import TaskRepository
from models.crew_definition import AgentDoc, TaskDoc
from models.crew_outputs import MemoryConsolidationOutput
from models.memory import Memory, MemoryScope, clamp_memory_content
from models.prompt_log import PromptLogEntry, RunUsage
from utils.datetime_utils import to_athlete_tz
from models.task import TaskStatus, TaskType
from crew.daily_analysis import _make_agent, require_definition
from crew.prompt_logging import capture_prompt_log, drain_prompt_log
from crew.usage import collect_run_usage
from services.handlers.base import TaskHandler
from utils.duration import parse_iso8601_duration

logger = logging.getLogger(__name__)

_CONSOLIDATION_WINDOW_DAYS = 30


class _ConsolidationDataTool(BaseTool):
    name: str = "get_consolidation_data"
    description: str = (
        "Retrieve all active memories and recent daily analysis summaries for review. "
        "Returns JSON with 'active_memories' and 'recent_analyses' keys. Call this first."
    )
    _payload: str = ""

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, payload: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_payload", payload)

    def _run(self, **kwargs: Any) -> str:
        return self._payload


class MemoryConsolidationHandler(TaskHandler):
    """Handler for periodic memory consolidation tasks."""

    def __init__(
        self,
        task_repo: TaskRepository,
        athlete_repo: AthleteRepository,
        memory_repo: MemoryRepository,
        daily_analysis_repo: DailyAnalysisRepository,
        crew_def_repo: CrewDefinitionRepository,
        prompt_log_repo: PromptLogRepository | None = None,
    ):
        self.task_repo = task_repo
        self.athlete_repo = athlete_repo
        self.memory_repo = memory_repo
        self.daily_analysis_repo = daily_analysis_repo
        self.crew_def_repo = crew_def_repo
        self.prompt_log_repo = prompt_log_repo

    async def execute(self, task_id: str, athlete_id: int, parameters: dict[str, Any]) -> dict[str, Any]:
        from config import settings

        logger.info("Starting memory consolidation for task %s, athlete %s", task_id, athlete_id)

        min_age = parse_iso8601_duration(settings.consolidation_min_age)
        if min_age.total_seconds() > 0:
            recent_tasks = await self.task_repo.get_tasks_by_athlete(
                athlete_id, limit=20, status=TaskStatus.COMPLETED
            )
            last_run = next(
                (t for t in recent_tasks if t.task_type == TaskType.MEMORY_CONSOLIDATION and t.task_id != task_id),
                None,
            )
            if last_run and last_run.completed_at:
                elapsed = datetime.now(timezone.utc) - last_run.completed_at
                if elapsed < min_age:
                    next_eligible = last_run.completed_at + min_age
                    logger.info(
                        "Skipping consolidation for athlete %s: last run was %s ago (min age %s)",
                        athlete_id, elapsed, min_age,
                    )
                    return {
                        "analysis_type": "memory_consolidation",
                        "skipped": True,
                        "reason": "min_age_not_elapsed",
                        "next_eligible_at": next_eligible.isoformat(),
                    }

        window_days = parameters.get("window_days", _CONSOLIDATION_WINDOW_DAYS)
        end_date = date.today().isoformat()
        start_date = (date.today() - timedelta(days=window_days - 1)).isoformat()

        athlete, active_memories, recent_analyses = await asyncio.gather(
            self.athlete_repo.get_athlete(athlete_id),
            self.memory_repo.get_active(athlete_id),
            self._get_recent_analysis_summaries(athlete_id, start_date, end_date),
        )
        tz_str = athlete.settings.timezone if athlete else "UTC"
        await self.task_repo.update_task_progress(task_id, 0.2)

        payload = json.dumps(
            {
                "active_memories": [
                    {
                        "memory_id": m.memory_id,
                        "scope": m.scope,
                        "category": m.category,
                        "content": m.content,
                        "confidence": m.confidence,
                        "importance": m.importance,
                        "evidence_dates": m.evidence_dates,
                        "created_at": to_athlete_tz(m.created_at, tz_str).isoformat(),
                        "updated_at": to_athlete_tz(m.updated_at, tz_str).isoformat(),
                    }
                    for m in active_memories
                ],
                "recent_analyses": recent_analyses,
            },
            default=str,
        )

        agents = {a.name: a for a in await self.crew_def_repo.get_by_type("agent")}
        tasks = {t.name: t for t in await self.crew_def_repo.get_by_type("task")}
        agent_def = require_definition(agents, "memory_consolidator", "agent")
        task_def = require_definition(tasks, "memory_consolidation_task", "task")

        result, prompt_log_entries, run_usage = await asyncio.to_thread(
            self._run_consolidation_crew,
            payload,
            athlete_id,
            window_days,
            settings,
            agent_def,
            task_def,
        )
        await self.task_repo.update_task_progress(task_id, 0.8)

        if self.prompt_log_repo:
            try:
                await self.prompt_log_repo.insert_many(prompt_log_entries)
                await self.prompt_log_repo.insert_usage(run_usage)
            except Exception:
                logger.exception("Failed to persist prompt log entries for task %s", task_id)

        if result:
            await self._apply_consolidation(athlete_id, result, {m.memory_id: m for m in active_memories})

        return {
            "analysis_type": "memory_consolidation",
            "window_days": window_days,
            "memories_reviewed": len(active_memories),
            "updates": len(result.updates) if result else 0,
            "promotions": len(result.promotions) if result else 0,
            "deactivations": len(result.deactivations) if result else 0,
            "new_long_term": len(result.new_long_term) if result else 0,
        }

    async def _get_recent_analysis_summaries(self, athlete_id: int, start_date: str, end_date: str) -> list[dict]:
        return await self.daily_analysis_repo.get_summaries_for_range(athlete_id, start_date, end_date)

    def _run_consolidation_crew(
        self,
        payload: str,
        athlete_id: int,
        window_days: int,
        settings: Any,
        agent_def: AgentDoc,
        task_def: TaskDoc,
    ) -> tuple[MemoryConsolidationOutput | None, list[PromptLogEntry], RunUsage]:
        if settings.anthropic_api_key:
            os.environ.setdefault("ANTHROPIC_API_KEY", settings.anthropic_api_key)
        if settings.openai_api_key:
            os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)

        data_tool = _ConsolidationDataTool(payload=payload)
        agent = _make_agent(agent_def, tools=[data_tool], default_llm=settings.llm_model)
        task_inputs = {"athlete_name": f"athlete {athlete_id}", "window_days": window_days}
        task = Task(
            description=task_def.description.format(**task_inputs),
            expected_output=task_def.expected_output.format(**task_inputs),
            agent=agent,
            output_pydantic=MemoryConsolidationOutput,
        )
        crew = Crew(agents=[agent], tasks=[task], verbose=True)
        with capture_prompt_log(athlete_id, "memory_consolidation", crew) as prompt_log_run_id:
            result = crew.kickoff()
        prompt_log_entries = drain_prompt_log(prompt_log_run_id)
        run_usage = collect_run_usage(crew, athlete_id, "memory_consolidation", prompt_log_run_id)

        output: MemoryConsolidationOutput | None = None
        if result.tasks_output:
            pydantic_output = result.tasks_output[0].pydantic
            if isinstance(pydantic_output, MemoryConsolidationOutput):
                output = pydantic_output
            else:
                logger.warning(
                    "Memory consolidation pydantic output missing, raw=%r",
                    result.tasks_output[0].raw[:200],
                )
        return output, prompt_log_entries, run_usage

    async def _apply_consolidation(
        self,
        athlete_id: int,
        result: MemoryConsolidationOutput,
        existing_by_id: dict[str, Memory],
    ) -> None:
        now = datetime.now(timezone.utc)

        for update in result.updates:
            existing = existing_by_id.get(update.memory_id)
            if not existing:
                continue
            updated = existing.model_copy(update={
                "content": clamp_memory_content(update.content),
                "confidence": update.confidence,
                "importance": update.importance,
                "evidence_dates": update.evidence_dates,
                "updated_at": now,
            }).refresh_expiry()
            await self.memory_repo.upsert(updated)

        for memory_id in result.promotions:
            existing = existing_by_id.get(memory_id)
            if not existing:
                continue
            promoted = existing.model_copy(update={
                "scope": MemoryScope.LONG_TERM,
                "updated_at": now,
            }).refresh_expiry()
            await self.memory_repo.upsert(promoted)

        for memory_id in result.deactivations:
            await self.memory_repo.deactivate(memory_id)

        for draft in result.new_long_term:
            memory = Memory(
                memory_id=str(uuid.uuid4()),
                athlete_id=athlete_id,
                scope=MemoryScope.LONG_TERM,
                category=draft.category,
                content=clamp_memory_content(draft.content),
                confidence=draft.confidence,
                importance=draft.importance,
                evidence_dates=draft.evidence_dates,
                created_at=now,
                updated_at=now,
                expires_at=None,
            )
            await self.memory_repo.upsert(memory)

        logger.info(
            "Memory consolidation applied for athlete %s: %d updated, %d promoted, %d deactivated, %d new long-term",
            athlete_id, len(result.updates), len(result.promotions), len(result.deactivations), len(result.new_long_term),
        )
