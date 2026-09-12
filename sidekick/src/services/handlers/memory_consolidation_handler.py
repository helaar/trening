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
from models.crew_outputs import (
    MemoryConsolidationDetailOutput,
    MemoryTriageOutput,
)
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

# Pass 2 (content rewriting) only ever sees memories triage flagged "update", plus new
# pattern drafts — batching keeps that call's output bounded regardless of how many of
# them there are, unlike pass 1 which reviews the whole bank in one shot but only emits
# a compact decision per memory.
_CONSOLIDATION_DETAIL_BATCH_SIZE = 12


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


class _ConsolidationDetailDataTool(BaseTool):
    name: str = "get_consolidation_detail_data"
    description: str = (
        "Retrieve the memories flagged for content rewriting, new pattern drafts, and the "
        "analysis window data. Returns JSON with 'memories_to_update', 'new_patterns', and "
        "'recent_analyses' keys. Call this first."
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
    """Handler for periodic memory consolidation tasks.

    Runs as two crew passes: a triage pass reviews the whole memory bank and emits a
    compact keep/update/deactivate/promote/merge decision per memory (bounded output
    regardless of bank size), then a detail pass rewrites content only for the subset
    triage flagged "update" (batched, so that output stays bounded too). Splitting this
    way keeps either pass's output well under the LLM's max_tokens cap even as the bank
    grows — a single combined pass previously truncated mid-JSON once the bank held
    enough memories to review.
    """

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
        existing_by_id = {m.memory_id: m for m in active_memories}
        await self.task_repo.update_task_progress(task_id, 0.2)

        triage_payload = json.dumps(
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
        triage_task_def = require_definition(tasks, "memory_triage_task", "task")
        detail_task_def = require_definition(tasks, "memory_consolidation_detail_task", "task")

        triage, triage_log_entries, triage_usage = await asyncio.to_thread(
            self._run_triage_crew,
            triage_payload,
            athlete_id,
            window_days,
            settings,
            agent_def,
            triage_task_def,
        )
        await self._persist_run_log(triage_log_entries, triage_usage)
        await self.task_repo.update_task_progress(task_id, 0.5)

        promotions = 0
        deactivations = 0
        updates = 0
        new_long_term = 0

        if triage:
            promotions, deactivations = await self._apply_triage_decisions(
                athlete_id, triage, existing_by_id
            )

            detail_batches = self._build_detail_batches(triage, existing_by_id, recent_analyses)
            for batch_payload in detail_batches:
                detail, detail_log_entries, detail_usage = await asyncio.to_thread(
                    self._run_detail_crew,
                    batch_payload,
                    athlete_id,
                    window_days,
                    settings,
                    agent_def,
                    detail_task_def,
                )
                await self._persist_run_log(detail_log_entries, detail_usage)
                if detail:
                    await self._apply_detail_output(athlete_id, detail, existing_by_id)
                    updates += len(detail.updates)
                    new_long_term += len(detail.new_long_term)

        await self.task_repo.update_task_progress(task_id, 0.9)

        return {
            "analysis_type": "memory_consolidation",
            "window_days": window_days,
            "memories_reviewed": len(active_memories),
            "updates": updates,
            "promotions": promotions,
            "deactivations": deactivations,
            "new_long_term": new_long_term,
        }

    async def _get_recent_analysis_summaries(self, athlete_id: int, start_date: str, end_date: str) -> list[dict]:
        return await self.daily_analysis_repo.get_summaries_for_range(athlete_id, start_date, end_date)

    async def _persist_run_log(self, entries: list[PromptLogEntry], usage: RunUsage) -> None:
        if not self.prompt_log_repo:
            return
        try:
            await self.prompt_log_repo.insert_many(entries)
            await self.prompt_log_repo.insert_usage(usage)
        except Exception:
            logger.exception("Failed to persist prompt log entries")

    def _ensure_api_keys(self, settings: Any) -> None:
        if settings.anthropic_api_key:
            os.environ.setdefault("ANTHROPIC_API_KEY", settings.anthropic_api_key)
        if settings.openai_api_key:
            os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)

    def _run_triage_crew(
        self,
        payload: str,
        athlete_id: int,
        window_days: int,
        settings: Any,
        agent_def: AgentDoc,
        task_def: TaskDoc,
    ) -> tuple[MemoryTriageOutput | None, list[PromptLogEntry], RunUsage]:
        self._ensure_api_keys(settings)

        data_tool = _ConsolidationDataTool(payload=payload)
        agent = _make_agent(agent_def, tools=[data_tool], default_llm=settings.llm_model)
        task_inputs = {"athlete_name": f"athlete {athlete_id}", "window_days": window_days}
        task = Task(
            description=task_def.description.format(**task_inputs),
            expected_output=task_def.expected_output.format(**task_inputs),
            agent=agent,
            output_pydantic=MemoryTriageOutput,
        )
        crew = Crew(agents=[agent], tasks=[task], verbose=True)
        with capture_prompt_log(athlete_id, "memory_triage", crew) as prompt_log_run_id:
            result = crew.kickoff()
        prompt_log_entries = drain_prompt_log(prompt_log_run_id)
        run_usage = collect_run_usage(crew, athlete_id, "memory_triage", prompt_log_run_id)

        output: MemoryTriageOutput | None = None
        if result.tasks_output:
            pydantic_output = result.tasks_output[0].pydantic
            if isinstance(pydantic_output, MemoryTriageOutput):
                output = pydantic_output
            else:
                logger.warning(
                    "Memory triage pydantic output missing, raw=%r",
                    result.tasks_output[0].raw[:200],
                )
        return output, prompt_log_entries, run_usage

    def _run_detail_crew(
        self,
        payload: str,
        athlete_id: int,
        window_days: int,
        settings: Any,
        agent_def: AgentDoc,
        task_def: TaskDoc,
    ) -> tuple[MemoryConsolidationDetailOutput | None, list[PromptLogEntry], RunUsage]:
        self._ensure_api_keys(settings)

        data_tool = _ConsolidationDetailDataTool(payload=payload)
        agent = _make_agent(agent_def, tools=[data_tool], default_llm=settings.llm_model)
        task_inputs = {"athlete_name": f"athlete {athlete_id}", "window_days": window_days}
        task = Task(
            description=task_def.description.format(**task_inputs),
            expected_output=task_def.expected_output.format(**task_inputs),
            agent=agent,
            output_pydantic=MemoryConsolidationDetailOutput,
        )
        crew = Crew(agents=[agent], tasks=[task], verbose=True)
        with capture_prompt_log(
            athlete_id, "memory_consolidation_detail", crew
        ) as prompt_log_run_id:
            result = crew.kickoff()
        prompt_log_entries = drain_prompt_log(prompt_log_run_id)
        run_usage = collect_run_usage(
            crew, athlete_id, "memory_consolidation_detail", prompt_log_run_id
        )

        output: MemoryConsolidationDetailOutput | None = None
        if result.tasks_output:
            pydantic_output = result.tasks_output[0].pydantic
            if isinstance(pydantic_output, MemoryConsolidationDetailOutput):
                output = pydantic_output
            else:
                logger.warning(
                    "Memory consolidation detail pydantic output missing, raw=%r",
                    result.tasks_output[0].raw[:200],
                )
        return output, prompt_log_entries, run_usage

    def _build_detail_batches(
        self,
        triage: MemoryTriageOutput,
        existing_by_id: dict[str, Memory],
        recent_analyses: list[dict],
    ) -> list[str]:
        """Group triage's "update" decisions + new pattern flags into bounded-size batches.

        Each batch becomes one pass-2 crew call, so its own output — the only part of
        this job that writes prose — stays bounded regardless of how large the bank is.
        """
        merged_from: dict[str, list[Memory]] = {}
        for decision in triage.decisions:
            if decision.action == "merge" and decision.merge_into:
                source = existing_by_id.get(decision.memory_id)
                if source:
                    merged_from.setdefault(decision.merge_into, []).append(source)

        update_items: list[dict[str, Any]] = []
        for decision in triage.decisions:
            if decision.action != "update":
                continue
            existing = existing_by_id.get(decision.memory_id)
            if not existing:
                continue
            update_items.append(
                {
                    "memory_id": existing.memory_id,
                    "scope": existing.scope,
                    "category": existing.category,
                    "content": existing.content,
                    "confidence": existing.confidence,
                    "importance": existing.importance,
                    "evidence_dates": existing.evidence_dates,
                    "triage_reason": decision.reason,
                    "merged_from": [
                        {
                            "memory_id": m.memory_id,
                            "content": m.content,
                            "evidence_dates": m.evidence_dates,
                        }
                        for m in merged_from.get(existing.memory_id, [])
                    ],
                }
            )

        new_pattern_items = [p.model_dump() for p in triage.new_patterns]

        work_items: list[tuple[str, dict[str, Any]]] = [("update", item) for item in update_items]
        work_items += [("new_pattern", item) for item in new_pattern_items]

        batches: list[str] = []
        for i in range(0, len(work_items), _CONSOLIDATION_DETAIL_BATCH_SIZE):
            chunk = work_items[i : i + _CONSOLIDATION_DETAIL_BATCH_SIZE]
            batches.append(
                json.dumps(
                    {
                        "memories_to_update": [item for kind, item in chunk if kind == "update"],
                        "new_patterns": [item for kind, item in chunk if kind == "new_pattern"],
                        "recent_analyses": recent_analyses,
                    },
                    default=str,
                )
            )
        return batches

    async def _apply_triage_decisions(
        self,
        athlete_id: int,
        triage: MemoryTriageOutput,
        existing_by_id: dict[str, Memory],
    ) -> tuple[int, int]:
        """Apply the decisions that never need an LLM to execute: promote/deactivate/merge
        are pure metadata operations on the existing memory."""
        now = datetime.now(timezone.utc)
        promotions = 0
        deactivations = 0

        for decision in triage.decisions:
            if decision.action == "promote":
                existing = existing_by_id.get(decision.memory_id)
                if not existing:
                    continue
                promoted = existing.model_copy(
                    update={"scope": MemoryScope.LONG_TERM, "updated_at": now}
                ).refresh_expiry()
                await self.memory_repo.upsert(promoted)
                promotions += 1
            elif decision.action in ("deactivate", "merge"):
                await self.memory_repo.deactivate(decision.memory_id)
                deactivations += 1
            # "keep" and "update" need no direct DB action here — "update" is handled
            # once the detail pass rewrites its content.

        logger.info(
            "Memory triage applied for athlete %s: %d promoted, %d deactivated/merged",
            athlete_id, promotions, deactivations,
        )
        return promotions, deactivations

    async def _apply_detail_output(
        self,
        athlete_id: int,
        detail: MemoryConsolidationDetailOutput,
        existing_by_id: dict[str, Memory],
    ) -> None:
        now = datetime.now(timezone.utc)

        for update in detail.updates:
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

        for draft in detail.new_long_term:
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
            "Memory consolidation detail applied for athlete %s: %d updated, %d new long-term",
            athlete_id, len(detail.updates), len(detail.new_long_term),
        )
