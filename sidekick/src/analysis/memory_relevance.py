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
_MAX_RECOVERY_MEMORIES = 8

# Guarantees the coach keeps some sense of the athlete's normal habits/performance even
# during a sustained recovery/risk episode, where RECOVERY/RISK memories can otherwise
# legitimately win every scored slot via the readiness:low context boost below. Excludes
# long-term HABIT, which is already a core candidate (see _is_core_candidate).
_MIN_DIVERSITY_SLOTS = 3
_DIVERSITY_CATEGORIES = (MemoryCategory.HABIT, MemoryCategory.PERFORMANCE)

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

    # Diversity floor: if a sustained crisis has crowded HABIT/PERFORMANCE out of the
    # non-core slots entirely, backfill from the top-scored ones not already selected,
    # evicting the lowest-scored non-core, non-diversity selections to make room. A
    # no-op when diversity memories already naturally make the cut.
    selected_ids = {id(m) for m in selected}
    diversity_selected = sum(1 for m in selected if m.category in _DIVERSITY_CATEGORIES)
    if diversity_selected < _MIN_DIVERSITY_SLOTS:
        needed = _MIN_DIVERSITY_SLOTS - diversity_selected
        diversity_candidates = sorted(
            (m for m in remaining if id(m) not in selected_ids and m.category in _DIVERSITY_CATEGORIES),
            key=lambda m: (scored[id(m)], m.updated_at),
            reverse=True,
        )[:needed]
        if diversity_candidates:
            non_core_selected = sorted(
                (m for m in selected if id(m) not in core_ids),
                key=lambda m: (scored[id(m)], m.updated_at),
            )
            evictable = [m for m in non_core_selected if m.category not in _DIVERSITY_CATEGORIES]
            for new_m, evict_m in zip(diversity_candidates, evictable):
                selected.remove(evict_m)
                selected.append(new_m)

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


def select_recovery_memories(memories: list[Memory], analysis_date: str) -> list[dict[str, Any]]:
    """Recovery/risk memories for the restitution analyst, ranked and trimmed.

    Distilled, recency-weighted context (e.g. a recent illness) the analyst uses to
    explain anomalies in today's metrics — not the coach's wide, goal-inclusive set.
    Scored against a low-readiness context so recovery/risk memories get their context
    boost; deterministic for fixed inputs.
    """
    today = date.fromisoformat(analysis_date)
    ctx = DayContext(readiness="low")
    recovery = [m for m in memories if m.category in (MemoryCategory.RECOVERY, MemoryCategory.RISK)]
    recovery.sort(key=lambda m: (_score_memory(m, ctx, today), m.updated_at), reverse=True)

    return [
        {
            "scope": m.scope,
            "category": m.category,
            "content": m.content,
            "confidence": m.confidence,
        }
        for m in recovery[:_MAX_RECOVERY_MEMORIES]
    ]
