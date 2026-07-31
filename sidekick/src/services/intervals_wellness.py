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

Field names below (ctl/atl/rampRate/id/sleepSecs/hrv/restingHR) are Intervals.icu's
documented wellness response fields; pending final confirmation against a real
captured response via scripts/spike_intervals_icu.py's "Wellness" section before
being treated as settled (see the Phase 2 precedent in analysis/intervals_mapping.py,
where field names were confirmed the same way before being trusted).

Phase 5 also sources the restitution check-in's objective fields (sleep_hours, hrv,
resting_hr) from the same wellness entries — see map_wellness_entry_to_restitution
and merge_restitution. Subjective fields (sleep_quality, readiness, comment) are
never touched here: Intervals.icu has no equivalent to the athlete's own self-rating.
"""

import logging
from datetime import date, timedelta
from typing import Any

import httpx

from clients.intervals_icu.client import IntervalsIcuClient, IntervalsIcuError, IntervalsIcuNotConfigured
from models.athlete import AthleteSettings
from models.daily_entry import DailyEntry, Restitution, RestitutionSync
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


def map_wellness_entry_to_restitution(entry: dict[str, Any]) -> RestitutionSync:
    sleep_secs = entry.get("sleepSecs")
    return RestitutionSync(
        date=str(entry.get("id") or ""),
        sleep_hours=sleep_secs / 3600 if isinstance(sleep_secs, (int, float)) else None,
        hrv=entry.get("hrv"),
        resting_hr=entry.get("restingHR"),
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


async def get_restitution_sync_series(
    settings: AthleteSettings | None, start: str, end: str
) -> list[RestitutionSync]:
    entries = await _fetch_wellness(settings, date.fromisoformat(start), date.fromisoformat(end))
    points = [map_wellness_entry_to_restitution(e) for e in entries]
    points.sort(key=lambda p: p.date)
    return points


def merge_restitution(
    entry: DailyEntry | None,
    athlete_id: int,
    date_str: str,
    sync: RestitutionSync | None,
) -> DailyEntry | None:
    """Fill empty objective restitution fields from a live Intervals.icu reading.

    Only fills sleep_hours/hrv/resting_hr, and only where the manually-entered
    value is None — an explicit manual entry always wins. sleep_quality/
    readiness/comment are never touched. Returns entry unchanged if sync is None.
    """
    if sync is None:
        return entry

    restitution = entry.restitution if entry else None

    def fill(manual: float | int | None, synced: float | int | None):
        return manual if manual is not None else synced

    merged = Restitution(
        sleep_hours=fill(restitution.sleep_hours if restitution else None, sync.sleep_hours),
        sleep_quality=restitution.sleep_quality if restitution else None,
        hrv=fill(restitution.hrv if restitution else None, sync.hrv),
        resting_hr=fill(restitution.resting_hr if restitution else None, sync.resting_hr),
        readiness=restitution.readiness if restitution else None,
        comment=restitution.comment if restitution else None,
    )

    if entry is None:
        return DailyEntry(athlete_id=athlete_id, date=date_str, restitution=merged)
    return entry.model_copy(update={"restitution": merged})


def default_start(end: str, days: int) -> str:
    return (date.fromisoformat(end) - timedelta(days=days)).isoformat()
