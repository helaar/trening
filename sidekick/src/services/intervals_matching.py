"""Cross-reference Intervals.icu activities to Strava activities.

The athlete's devices (Garmin bike computer/HR strap/power meter/watch, Zwift)
upload to Strava and Intervals.icu independently rather than through a single
chained hop, so an Intervals.icu activity often has no Strava-id back-reference
at all even though the "same" workout exists in both systems. Both matching
tiers below are expected to fire regularly — fuzzy matching is not a rare
fallback.

Field names used to extract a Strava-id back-reference, start time, duration,
and polyline from an Intervals.icu activity payload are best-effort guesses
(the Intervals.icu docs domain returned 403 to automated fetches during
planning). Verify these against a real API response (see
scripts/spike_intervals_icu.py) and adjust the candidate-key tuples below if
needed — the extraction helpers are isolated exactly so that's a one-place fix.
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Any, Literal

from clients.strava.polyline import is_route_match
from models.intervals_activity import IntervalsActivityRaw
from models.strava_activity import StravaActivityRaw

logger = logging.getLogger(__name__)

# --- field-name candidates (verify in Phase 0, see module docstring) ---------

_STRAVA_ID_KEYS = ("strava_id", "stravaId", "strava_activity_id")
_EXTERNAL_ID_KEYS = ("external_id", "externalId")
_EXTERNAL_ID_STRAVA_RE = re.compile(r"strava[_-]?(\d+)", re.IGNORECASE)
_INTERVALS_ID_KEYS = ("id", "intervals_id", "intervalsId")
_START_TIME_KEYS = ("start_date", "start_date_local", "startTime")
_DURATION_KEYS = ("moving_time", "movingTime", "elapsed_time", "elapsedTime", "duration")

# --- matching tolerances -----------------------------------------------------

START_TIME_TOLERANCE = timedelta(minutes=5)
DURATION_RELATIVE_TOLERANCE = 0.05
DURATION_ABS_FLOOR_SECONDS = 60.0
ROUTE_MATCH_THRESHOLD_METERS = 100.0


def _extract_strava_id(intervals_activity: dict[str, Any]) -> int | None:
    for key in _STRAVA_ID_KEYS:
        value = intervals_activity.get(key)
        if value:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    for key in _EXTERNAL_ID_KEYS:
        external_id = intervals_activity.get(key)
        if external_id:
            match = _EXTERNAL_ID_STRAVA_RE.search(str(external_id))
            if match:
                return int(match.group(1))
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


def _extract_polyline(data: dict[str, Any]) -> str | None:
    for key in ("polyline", "summary_polyline"):
        value = data.get(key)
        if value:
            return str(value)
    map_data = data.get("map")
    if isinstance(map_data, dict):
        return map_data.get("summary_polyline") or map_data.get("polyline")
    return None


def _is_fuzzy_candidate(
    intervals_start: datetime | None,
    intervals_duration: float | None,
    intervals_polyline: str | None,
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

    strava_polyline = _extract_polyline(strava.raw_data)
    if strava_polyline and intervals_polyline:
        if not is_route_match(
            strava_polyline, intervals_polyline, threshold_meters=ROUTE_MATCH_THRESHOLD_METERS
        ):
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
        match_method: Literal["strava_id", "fuzzy_polyline_date_duration", "ambiguous"]
        resolved_strava_id: int | None

        if strava_id is not None:
            match_method = "strava_id"
            resolved_strava_id = strava_id
        else:
            intervals_start = _parse_start_time(intervals_activity)
            intervals_duration = _extract_duration_seconds(intervals_activity)
            intervals_polyline = _extract_polyline(intervals_activity)
            candidates = [
                s
                for s in strava_activities
                if _is_fuzzy_candidate(intervals_start, intervals_duration, intervals_polyline, s)
            ]
            if len(candidates) == 1:
                match_method = "fuzzy_polyline_date_duration"
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
