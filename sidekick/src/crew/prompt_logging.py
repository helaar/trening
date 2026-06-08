"""Captures the literal LLM prompts/responses sent during crew runs.

Subscribes to CrewAI's event bus for `LLMCallCompletedEvent`, which carries the
exact `messages` array sent to the model alongside its `response` — ground
truth for what an agent actually saw, as opposed to a theoretical
reconstruction of agent/task/data composition (which would drift from CrewAI's
internal prompt assembly across versions).

Captured calls are buffered per-run (via a contextvar, so concurrent runs and
async-executed tasks don't cross-contaminate) and persisted by the caller after
`crew.kickoff()` returns, since event handlers run on CrewAI's own thread pool
and cannot perform async Mongo writes directly.
"""

import logging
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator

from crewai.events import BaseEventListener, LLMCallCompletedEvent

from models.prompt_log import PromptLogEntry

logger = logging.getLogger(__name__)


@dataclass
class _RunBuffer:
    run_id: str
    athlete_id: int
    crew_name: str
    entries: list[PromptLogEntry] = field(default_factory=list)


_current_run: ContextVar["_RunBuffer | None"] = ContextVar("_current_run", default=None)


@contextmanager
def capture_prompt_log(athlete_id: int, crew_name: str) -> Iterator[str]:
    """Buffer LLM calls made during the wrapped block under a fresh run_id.

    Usage:
        with capture_prompt_log(athlete_id, "daily_analysis") as run_id:
            crew.kickoff()
        await prompt_log_repo.insert_many(drain_prompt_log(run_id))
    """
    run_id = str(uuid.uuid4())
    buffer = _RunBuffer(run_id=run_id, athlete_id=athlete_id, crew_name=crew_name)
    token = _current_run.set(buffer)
    try:
        yield run_id
    finally:
        _current_run.reset(token)


def drain_prompt_log(run_id: str) -> list[PromptLogEntry]:
    """Pop and return the entries captured for `run_id` (call after kickoff() returns)."""
    buffer = _current_run.get()
    if buffer is None or buffer.run_id != run_id:
        return []
    entries, buffer.entries = buffer.entries, []
    return entries


class _PromptLogListener(BaseEventListener):
    """Process-global listener; buffers calls into whichever run is active on the emitting thread."""

    def setup_listeners(self, bus) -> None:
        @bus.on(LLMCallCompletedEvent)
        def _on_completed(source: Any, event: LLMCallCompletedEvent) -> None:
            buffer = _current_run.get()
            if buffer is None:
                return
            try:
                messages = event.messages
                if isinstance(messages, str):
                    messages = [{"role": "user", "content": messages}]
                buffer.entries.append(
                    PromptLogEntry(
                        run_id=buffer.run_id,
                        athlete_id=buffer.athlete_id,
                        crew_name=buffer.crew_name,
                        agent_role=getattr(event.from_agent, "role", None),
                        task_name=getattr(event.from_task, "name", None)
                        or getattr(event.from_task, "description", None),
                        model=event.model,
                        call_type=getattr(event.call_type, "value", event.call_type),
                        messages=messages or [],
                        response=event.response
                        if isinstance(event.response, str)
                        else str(event.response),
                    )
                )
            except Exception:
                logger.exception("Failed to buffer prompt log entry for run %s", buffer.run_id)


_listener: _PromptLogListener | None = None


def register_prompt_log_listener() -> None:
    """Register the process-global event listener. Call once at startup."""
    global _listener
    if _listener is None:
        _listener = _PromptLogListener()
