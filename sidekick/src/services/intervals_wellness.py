"""Fitness/fatigue/form (CTL/ATL/TSB), sourced live from Intervals.icu's wellness data.

Sidekick has never computed CTL/ATL itself (the earlier TrainingLoadAnalysis/
WeeklySummary models were plain TSS/HRSS accumulation, not exponentially-weighted
load, and were deleted as dead code in Phase 2a) — per the same "don't reinvent
it" philosophy as Phases 2-3, this reads Intervals.icu's own computation instead.
TSB is the one derived value computed locally (ctl - atl), since that formula is
a universal training-load definition, not an Intervals.icu-specific field.

No local caching: same rationale as intervals_calendar.py — wellness changes daily
and call volume here is low (once per Insights-page load), so a live read is both
simpler and more correct than a cached one.

Field names below (ctl/atl/rampRate/id) are Intervals.icu's documented wellness
response fields; pending final confirmation against a real captured response via
scripts/spike_intervals_icu.py's "Wellness" section before being treated as settled
(see the Phase 2 precedent in analysis/intervals_mapping.py, where field names were
confirmed the same way before being trusted).
"""

import logging
from datetime import date, timedelta
from typing import Any

import httpx

from clients.intervals_icu.client import IntervalsIcuClient, IntervalsIcuError, IntervalsIcuNotConfigured
from models.athlete import AthleteSettings
from models.fitness import FitnessPoint

logger = logging.getLogger(__name__)


def map_wellness_entry(entry: dict[str, Any]) -> FitnessPoint:
    ctl = entry.get("ctl")
    atl = entry.get("atl")
    tsb = ctl - atl if ctl is not None and atl is not None else None
    return FitnessPoint(
        date=str(entry.get("id") or ""),
        ctl=ctl,
        atl=atl,
        tsb=tsb,
        ramp_rate=entry.get("rampRate"),
    )


async def _fetch_wellness(
    settings: AthleteSettings | None, oldest: date, newest: date
) -> list[dict[str, Any]]:
    if not settings:
        return []
    try:
        client = IntervalsIcuClient.from_athlete_settings(settings)
    except IntervalsIcuNotConfigured:
        return []
    try:
        return await client.get_wellness(oldest, newest)
    except (IntervalsIcuError, httpx.HTTPError) as e:
        logger.warning("Intervals.icu wellness fetch failed for %s..%s: %s", oldest, newest, e)
        return []


async def get_fitness_series(
    settings: AthleteSettings | None, start: str, end: str
) -> list[FitnessPoint]:
    entries = await _fetch_wellness(settings, date.fromisoformat(start), date.fromisoformat(end))
    points = [map_wellness_entry(e) for e in entries]
    points.sort(key=lambda p: p.date)
    return points


def default_start(end: str, days: int) -> str:
    return (date.fromisoformat(end) - timedelta(days=days)).isoformat()
