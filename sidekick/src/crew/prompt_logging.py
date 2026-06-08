"""Captures the literal LLM prompts/responses sent during crew runs.

Subscribes to CrewAI's event bus for `LLMCallCompletedEvent`, which carries the
exact `messages` array sent to the model alongside its `response` — ground
truth for what an agent actually saw, as opposed to a theoretical
reconstruction of agent/task/data composition (which would drift from CrewAI's
internal prompt assembly across versions).

Calls are correlated to a run by `task_id`/`agent_id` (the `Task`/`Agent` UUIDs,
which the event carries as plain strings — CrewAI nulls out the actual
`from_task`/`from_agent` objects on construction), not by contextvars:
CrewAI executes `async_execution=True` tasks on bare `threading.Thread`s, which
do not inherit the caller's context, so a contextvar set before `kickoff()`
would be invisible to those threads. UUIDs survive the thread hop.

Captured calls are buffered per-run and persisted by the caller after
`crew.kickoff()` returns, since event handlers run on CrewAI's own thread pool
and cannot perform async Mongo writes directly.
"""

import logging
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator

from crewai.events import (
    BaseEventListener,
    LLMCallCompletedEvent,
    ToolUsageErrorEvent,
    ToolUsageFinishedEvent,
)

from models.prompt_log import PromptLogEntry

if TYPE_CHECKING:
    from crewai import Crew

logger = logging.getLogger(__name__)


@dataclass
class _RunBuffer:
    run_id: str
    athlete_id: int
    crew_name: str
    entries: list[PromptLogEntry] = field(default_factory=list)


_lock = threading.Lock()
_registry: dict[str, _RunBuffer] = {}
_buffers_by_run: dict[str, _RunBuffer] = {}


@contextmanager
def capture_prompt_log(athlete_id: int, crew_name: str, crew: "Crew") -> Iterator[str]:
    """Buffer LLM calls made by `crew` during the wrapped block under a fresh run_id.

    Usage:
        with capture_prompt_log(athlete_id, "daily_analysis", crew) as run_id:
            crew.kickoff()
        await prompt_log_repo.insert_many(drain_prompt_log(run_id))
    """
    run_id = str(uuid.uuid4())
    buffer = _RunBuffer(run_id=run_id, athlete_id=athlete_id, crew_name=crew_name)
    keys: list[str] = []
    for task in crew.tasks:
        keys.append(str(task.id))
        agent = getattr(task, "agent", None)
        if agent is not None:
            keys.append(str(agent.id))
    with _lock:
        _buffers_by_run[run_id] = buffer
        for key in keys:
            _registry[key] = buffer
    try:
        yield run_id
    finally:
        with _lock:
            for key in keys:
                _registry.pop(key, None)


def drain_prompt_log(run_id: str) -> list[PromptLogEntry]:
    """Pop and return the entries captured for `run_id` (call after kickoff() returns)."""
    with _lock:
        buffer = _buffers_by_run.pop(run_id, None)
    if buffer is None:
        return []
    return buffer.entries


def _lookup_buffer(event: Any) -> "_RunBuffer | None":
    task_id = getattr(event, "task_id", None)
    agent_id = getattr(event, "agent_id", None)
    with _lock:
        return _registry.get(task_id) or _registry.get(agent_id)


def _stringify(value: Any) -> str:
    return value if isinstance(value, str) else str(value)


class _PromptLogListener(BaseEventListener):
    """Process-global listener; correlates calls to a run via task_id/agent_id."""

    def setup_listeners(self, bus) -> None:
        @bus.on(LLMCallCompletedEvent)
        def _on_llm_completed(source: Any, event: LLMCallCompletedEvent) -> None:
            buffer = _lookup_buffer(event)
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
                        kind="llm_call",
                        agent_role=getattr(event, "agent_role", None),
                        task_name=getattr(event, "task_name", None),
                        model=event.model,
                        call_type=getattr(event.call_type, "value", event.call_type),
                        messages=messages or [],
                        response=_stringify(event.response),
                    )
                )
            except Exception:
                logger.exception("Failed to buffer LLM call entry for run %s", buffer.run_id)

        @bus.on(ToolUsageFinishedEvent)
        def _on_tool_finished(source: Any, event: ToolUsageFinishedEvent) -> None:
            buffer = _lookup_buffer(event)
            if buffer is None:
                return
            try:
                buffer.entries.append(
                    PromptLogEntry(
                        run_id=buffer.run_id,
                        athlete_id=buffer.athlete_id,
                        crew_name=buffer.crew_name,
                        kind="tool_call",
                        agent_role=getattr(event, "agent_role", None),
                        task_name=getattr(event, "task_name", None),
                        tool_name=event.tool_name,
                        tool_args=event.tool_args,
                        tool_output=_stringify(event.output),
                    )
                )
            except Exception:
                logger.exception("Failed to buffer tool usage entry for run %s", buffer.run_id)

        @bus.on(ToolUsageErrorEvent)
        def _on_tool_error(source: Any, event: ToolUsageErrorEvent) -> None:
            buffer = _lookup_buffer(event)
            if buffer is None:
                return
            try:
                buffer.entries.append(
                    PromptLogEntry(
                        run_id=buffer.run_id,
                        athlete_id=buffer.athlete_id,
                        crew_name=buffer.crew_name,
                        kind="tool_call",
                        agent_role=getattr(event, "agent_role", None),
                        task_name=getattr(event, "task_name", None),
                        tool_name=event.tool_name,
                        tool_args=event.tool_args,
                        tool_error=_stringify(event.error),
                    )
                )
            except Exception:
                logger.exception(
                    "Failed to buffer tool usage error entry for run %s", buffer.run_id
                )


_listener: _PromptLogListener | None = None


def register_prompt_log_listener() -> None:
    """Register the process-global event listener. Call once at startup."""
    global _listener
    if _listener is None:
        _listener = _PromptLogListener()
