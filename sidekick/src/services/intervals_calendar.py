"""Read-only planned-workout calendar, sourced live from Intervals.icu.

Replaces PlanRepository/training_plans entirely — sidekick no longer stores
planned workouts of its own; the athlete authors/edits them in Intervals.icu
(per the user's explicit decision) and this module maps its calendar events
onto the same PlannedActivity shape the rest of the app already expects, so
crew/daily_analysis.py's race/taper logic needs no changes.

No local caching: planned workouts are live, editable-elsewhere state, so a
live read on each call is more correct than a cached one, and call volume
here is low (once per calendar-page-load, once per daily-coaching-run).
"""

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx

from clients.intervals_icu.client import IntervalsIcuClient, IntervalsIcuError, IntervalsIcuNotConfigured
from models.athlete import AthleteSettings
from models.plan import PlannedActivity

logger = logging.getLogger(__name__)

# Intervals.icu's `get_events` needs bounded dates; this stands in for the
# "unbounded forward" race fetch PlanRepository.get_races_from did — the LLM,
# not a fixed window, decides which upcoming races are worth mentioning.
_RACES_FORWARD_WINDOW_DAYS = 730

# Matches PlannedActivity.sport's Literal values; case-insensitive on the
# Intervals.icu `type` field. Unrecognized types default to "other", same
# defensive convention as the retired TrainingPeaks ICS client.
_SPORT_MAP: dict[str, str] = {
    "ride": "cycling",
    "virtualride": "cycling",
    "gravelride": "cycling",
    "mountainbikeride": "cycling",
    "run": "running",
    "trailrun": "running",
    "virtualrun": "running",
    "weighttraining": "strength",
    "workout": "strength",
    "nordicski": "skiing_cross",
    "backcountryski": "skiing_cross",
    "alpineski": "skiing_alpine",
    "rest": "day_off",
    "dayoff": "day_off",
}

# RACE_A implies the season-goal label too (the standard A/B/C goal-race
# convention: A races are season goals). RACE_B/RACE_C get "race" only.
_RACE_CATEGORY_PRIORITY: dict[str, str] = {"RACE_A": "A", "RACE_B": "B", "RACE_C": "C"}

# Substring match against name/description, case-insensitive — preserves
# crew/daily_analysis.py's existing hard-demand detection (which checks
# labels for "interval"/"intervals"/"hard"/"key"/"race") without requiring
# any change there, since Intervals.icu events have no equivalent free-text
# label list to read directly.
_HARD_KEYWORDS = ("interval", "intervals", "hard", "key")


def map_event_to_planned_activity(event: dict[str, Any], athlete_id: int) -> PlannedActivity:
    event_id = str(event.get("id"))
    sport = _SPORT_MAP.get(str(event.get("type") or "").lower(), "other")

    category = str(event.get("category") or "")
    priority = _RACE_CATEGORY_PRIORITY.get(category)

    name = event.get("name") or "Planned workout"
    description = event.get("description")
    text = f"{name} {description or ''}".lower()

    labels: list[str] = []
    if priority:
        labels.append("race")
        if priority == "A":
            labels.append("seasongoal")
    if any(keyword in text for keyword in _HARD_KEYWORDS):
        labels.append("hard")

    date_str = str(event.get("start_date_local") or event.get("start_date") or "")[:10]

    moving_time = event.get("moving_time")
    estimated_duration_min = (
        int(moving_time / 60) if isinstance(moving_time, (int, float)) and moving_time else None
    )

    raw_tss = event.get("icu_training_load") or event.get("load_target")
    estimated_tss = int(raw_tss) if isinstance(raw_tss, (int, float)) else None

    updated_raw = event.get("updated")
    updated_at = _parse_datetime(updated_raw) if updated_raw else datetime.now(timezone.utc)

    return PlannedActivity(
        id=event_id,
        athlete_id=athlete_id,
        date=date_str,
        sport=sport,
        name=name,
        description=description,
        purpose=None,
        labels=labels,
        estimated_duration_min=estimated_duration_min,
        estimated_tss=estimated_tss,
        external_reference=event_id,
        race_priority=priority,
        created_at=updated_at,
        updated_at=updated_at,
    )


def _parse_datetime(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.now(timezone.utc)


async def _fetch_events(settings: AthleteSettings | None, oldest: date, newest: date) -> list[dict[str, Any]]:
    if not settings:
        return []
    try:
        client = IntervalsIcuClient.from_athlete_settings(settings)
    except IntervalsIcuNotConfigured:
        return []
    try:
        return await client.get_events(oldest, newest)
    except (IntervalsIcuError, httpx.HTTPError) as e:
        logger.warning("Intervals.icu events fetch failed for %s..%s: %s", oldest, newest, e)
        return []


async def get_for_date(
    athlete_id: int, settings: AthleteSettings | None, target_date: str
) -> list[PlannedActivity]:
    return await get_for_range(athlete_id, settings, target_date, target_date)


async def get_for_range(
    athlete_id: int, settings: AthleteSettings | None, start: str, end: str
) -> list[PlannedActivity]:
    events = await _fetch_events(settings, date.fromisoformat(start), date.fromisoformat(end))
    return [map_event_to_planned_activity(e, athlete_id) for e in events]


async def get_races_from(
    athlete_id: int, settings: AthleteSettings | None, start: str
) -> list[PlannedActivity]:
    start_date = date.fromisoformat(start)
    end_date = start_date + timedelta(days=_RACES_FORWARD_WINDOW_DAYS)
    events = await _fetch_events(settings, start_date, end_date)
    races = [
        a for e in events if "race" in (a := map_event_to_planned_activity(e, athlete_id)).labels
    ]
    races.sort(key=lambda a: a.date)
    return races
