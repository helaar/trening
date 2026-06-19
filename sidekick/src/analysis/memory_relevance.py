"""Relevance scoring and selection for coach memories.

This is the single source of truth for how active memories are ranked and
trimmed. Both the daily-analysis crew (the coach's ``get_athlete_memories``
tool) and the athlete-facing memory bank view consume these helpers, so the
ordering a user sees matches the ordering the coach reasons over.

The functions are pure: they take already-loaded ``Memory`` objects plus a
``DayContext`` and return ranked plain dicts. No I/O lives here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from models.memory import _LONG_TERM_TTL_DAYS, _RECENT_TTL_DAYS, Memory, MemoryCategory, MemoryScope

_MAX_MEMORIES = 14
_CORE_MEMORIES = 6

# Category gets a full context match (1.0) in each of these situations.
_CONTEXT_BOOSTS: dict[str, set[MemoryCategory]] = {
    "readiness:low": {MemoryCategory.RECOVERY, MemoryCategory.RISK},
    "demand:hard": {MemoryCategory.PERFORMANCE, MemoryCategory.RISK},
    "demand:easy": {MemoryCategory.HABIT, MemoryCategory.RECOVERY},
    "demand:rest": {MemoryCategory.HABIT, MemoryCategory.RECOVERY},
    "phase:taper": {MemoryCategory.GOAL, MemoryCategory.PERFORMANCE},
}


@dataclass(frozen=True)
class DayContext:
    """Deterministic snapshot of today's situation used to score memory relevance."""

    readiness: str = "normal"  # low | normal | high
    demand: str = "easy"  # rest | easy | hard
    phase: str = "normal"  # taper | normal

    def boosted_categories(self) -> set[MemoryCategory]:
        boosted: set[MemoryCategory] = set()
        boosted |= _CONTEXT_BOOSTS.get(f"readiness:{self.readiness}", set())
        boosted |= _CONTEXT_BOOSTS.get(f"demand:{self.demand}", set())
        boosted |= _CONTEXT_BOOSTS.get(f"phase:{self.phase}", set())
        return boosted


def _context_match(memory: Memory, ctx: DayContext) -> float:
    if memory.category in ctx.boosted_categories():
        return 1.0
    if memory.category == MemoryCategory.GOAL or (
        memory.category == MemoryCategory.HABIT and memory.scope == MemoryScope.LONG_TERM
    ):
        return 0.5  # durable baseline floor — never fully dropped
    return 0.3


def _recency(memory: Memory, today: date) -> float:
    ttl = _LONG_TERM_TTL_DAYS if memory.scope == MemoryScope.LONG_TERM else _RECENT_TTL_DAYS
    age_days = max(0, (today - memory.updated_at.date()).days)
    return max(0.0, 1.0 - age_days / ttl)


def _score_memory(memory: Memory, ctx: DayContext, today: date) -> float:
    return (
        0.30 * memory.confidence
        + 0.30 * memory.importance
        + 0.15 * _recency(memory, today)
        + 0.25 * _context_match(memory, ctx)
    )


def _is_core_candidate(memory: Memory) -> bool:
    return memory.category in (MemoryCategory.GOAL, MemoryCategory.RISK) or (
        memory.category == MemoryCategory.HABIT and memory.scope == MemoryScope.LONG_TERM
    )


def select_relevant_memories(
    memories: list[Memory], ctx: DayContext, analysis_date: str
) -> list[dict[str, Any]]:
    """Hybrid, leaning-wide selection.

    A stable durable core (goals, long-term habits, active risks) is always present so the
    coach can track habits and progress over time; today's situation then fills the
    remaining slots and orders the whole set. Deterministic for fixed inputs.
    """
    today = date.fromisoformat(analysis_date)
    scored = {id(m): _score_memory(m, ctx, today) for m in memories}

    core_pool = sorted(
        (m for m in memories if _is_core_candidate(m)),
        key=lambda m: (m.importance * m.confidence, m.updated_at),
        reverse=True,
    )
    core = core_pool[:_CORE_MEMORIES]
    core_ids = {id(m) for m in core}

    remaining = sorted(
        (m for m in memories if id(m) not in core_ids),
        key=lambda m: (scored[id(m)], m.updated_at),
        reverse=True,
    )
    selected = (core + remaining)[:_MAX_MEMORIES]
    selected.sort(key=lambda m: (scored[id(m)], m.updated_at), reverse=True)

    return [
        {
            "scope": m.scope,
            "category": m.category,
            "content": m.content,
            "confidence": m.confidence,
        }
        for m in selected
    ]
