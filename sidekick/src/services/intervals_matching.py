"""Cross-reference Intervals.icu activities to Strava activities.

The athlete's devices (Garmin bike computer/HR strap/power meter/watch, Zwift)
upload to Strava and Intervals.icu independently rather than through a single
chained hop, so an Intervals.icu activity often has no Strava-id back-reference
at all even though the "same" workout exists in both systems. Both matching
tiers below are expected to fire regularly — fuzzy matching is not a rare
fallback.

Field names below were confirmed against a real Intervals.icu API response
(scripts/spike_intervals_icu.py, run 2026-07-30):
- Activities do carry a dedicated `strava_id` field (not a regex-parsed guess).
- There is no polyline/map field on this Intervals.icu endpoint, so fuzzy
  matching uses `distance` (present on both sides, in meters) instead of route
  similarity. This is unrelated to commute detection, which runs on Strava's
  own polyline via clients/strava/polyline.py and is unaffected.
- `start_date`/`moving_time`/`elapsed_time` match Strava's own field names
  exactly, so no separate candidate list was needed for those.

Confirmed against the athlete's real activity history (2026-07-30): `strava_id`
is populated on older activities synced via a Strava connection (source
"UPLOAD"), but is null on every activity since the athlete switched to
Intervals.icu's direct Garmin Connect integration (source "GARMIN_CONNECT",
2026-07-14 onward) — so for ongoing activity ingestion, fuzzy matching is the
dominant path, not a rare fallback, exactly as this module is designed to
handle.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Literal

from models.intervals_activity import IntervalsActivityRaw
from models.strava_activity import StravaActivityRaw

logger = logging.getLogger(__name__)

_STRAVA_ID_KEYS = ("strava_id",)
_INTERVALS_ID_KEYS = ("id",)
_START_TIME_KEYS = ("start_date",)
_DURATION_KEYS = ("moving_time", "elapsed_time")

# --- matching tolerances -----------------------------------------------------

START_TIME_TOLERANCE = timedelta(minutes=5)
DURATION_RELATIVE_TOLERANCE = 0.05
DURATION_ABS_FLOOR_SECONDS = 60.0
DISTANCE_RELATIVE_TOLERANCE = 0.05
DISTANCE_ABS_FLOOR_METERS = 200.0


def _extract_strava_id(intervals_activity: dict[str, Any]) -> int | None:
    for key in _STRAVA_ID_KEYS:
        value = intervals_activity.get(key)
        if value:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return None


def _extract_intervals_id(intervals_activity: dict[str, Any]) -> str | None:
    for key in _INTERVALS_ID_KEYS:
        value = intervals_activity.get(key)
        if value:
            return str(value)
    return None


def _parse_start_time(data: dict[str, Any]) -> datetime | None:
    for key in _START_TIME_KEYS:
        raw = data.get(key)
        if raw:
            try:
                return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            except ValueError:
                continue
    return None


def _extract_duration_seconds(data: dict[str, Any]) -> float | None:
    for key in _DURATION_KEYS:
        value = data.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _extract_distance_meters(data: dict[str, Any]) -> float | None:
    value = data.get("distance")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_fuzzy_candidate(
    intervals_start: datetime | None,
    intervals_duration: float | None,
    intervals_distance: float | None,
    strava: StravaActivityRaw,
) -> bool:
    strava_start = _parse_start_time(strava.raw_data)
    if strava_start is None or intervals_start is None:
        return False
    if abs((strava_start - intervals_start).total_seconds()) > START_TIME_TOLERANCE.total_seconds():
        return False

    strava_duration = _extract_duration_seconds(strava.raw_data)
    if strava_duration is not None and intervals_duration is not None:
        tolerance = max(DURATION_ABS_FLOOR_SECONDS, strava_duration * DURATION_RELATIVE_TOLERANCE)
        if abs(strava_duration - intervals_duration) > tolerance:
            return False

    strava_distance = _extract_distance_meters(strava.raw_data)
    if strava_distance and intervals_distance:
        tolerance = max(DISTANCE_ABS_FLOOR_METERS, strava_distance * DISTANCE_RELATIVE_TOLERANCE)
        if abs(strava_distance - intervals_distance) > tolerance:
            return False

    return True


def match_intervals_activities(
    intervals_activities: list[dict[str, Any]],
    strava_activities: list[StravaActivityRaw],
    athlete_id: int,
    fetched_at: datetime,
) -> list[IntervalsActivityRaw]:
    """Match each Intervals.icu activity to a Strava activity, where possible.

    Never guesses under ambiguity: if a Strava-id back-reference is absent and
    more than one Strava activity fits the fuzzy tolerance window equally well
    (or none does), the record is stored with match_method="ambiguous" and
    strava_activity_id left unset, rather than risk a wrong cross-reference
    that would silently corrupt a later merge.
    """
    results: list[IntervalsActivityRaw] = []

    for intervals_activity in intervals_activities:
        intervals_id = _extract_intervals_id(intervals_activity)
        if intervals_id is None:
            logger.warning("Skipping Intervals.icu activity with no id field: %r", intervals_activity)
            continue

        strava_id = _extract_strava_id(intervals_activity)
        match_method: Literal["strava_id", "fuzzy_date_duration_distance", "ambiguous"]
        resolved_strava_id: int | None

        if strava_id is not None:
            match_method = "strava_id"
            resolved_strava_id = strava_id
        else:
            intervals_start = _parse_start_time(intervals_activity)
            intervals_duration = _extract_duration_seconds(intervals_activity)
            intervals_distance = _extract_distance_meters(intervals_activity)
            candidates = [
                s
                for s in strava_activities
                if _is_fuzzy_candidate(intervals_start, intervals_duration, intervals_distance, s)
            ]
            if len(candidates) == 1:
                match_method = "fuzzy_date_duration_distance"
                resolved_strava_id = candidates[0].activity_id
            else:
                if len(candidates) > 1:
                    logger.info(
                        "Ambiguous match for Intervals.icu activity %s: %d candidate Strava activities",
                        intervals_id,
                        len(candidates),
                    )
                match_method = "ambiguous"
                resolved_strava_id = None

        results.append(
            IntervalsActivityRaw(
                athlete_id=athlete_id,
                strava_activity_id=resolved_strava_id,
                intervals_activity_id=intervals_id,
                match_method=match_method,
                raw_data=intervals_activity,
                fetched_at=fetched_at,
            )
        )

    return results
