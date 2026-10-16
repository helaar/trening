"""Read-only diagnostic: why does a Strava activity get no Intervals.icu match?

For one calendar day, prints the Intervals.icu activities the API returns (live)
next to the Strava activities in sidekick's local cache, and for every pair
says whether services.intervals_matching would accept it as a fuzzy candidate
or which check rejects it (start time, duration, distance). Also shows the
stored intervals_activities records for that day and what match_intervals_activities
would decide now.

No writes: neither the DB nor Intervals.icu is changed.

Run:
  cd sidekick
  PYTHONPATH=src uv run python scripts/diagnose_intervals_match.py \\
      --athlete-id 12345 --date 2026-10-15
"""

import argparse
import asyncio
from datetime import date, datetime, timezone
from typing import Any

from clients.intervals_icu.client import IntervalsIcuClient, IntervalsIcuNotConfigured
from config import settings as app_settings
from database.athlete_repository import AthleteRepository
from database.mongodb import db_manager
from database.workout_repository import WorkoutRepository
from models.strava_activity import StravaActivityRaw
from services.intervals_matching import (
    DISTANCE_ABS_FLOOR_METERS,
    DISTANCE_RELATIVE_TOLERANCE,
    DURATION_ABS_FLOOR_SECONDS,
    DURATION_RELATIVE_TOLERANCE,
    START_TIME_TOLERANCE,
    _extract_distance_meters,
    _extract_duration_seconds,
    _parse_start_time,
    match_intervals_activities,
)

_ICU_FIELDS = (
    "id", "name", "type", "source", "device_name", "start_date", "start_date_local",
    "moving_time", "elapsed_time", "distance", "strava_id",
    "icu_average_watts", "icu_weighted_avg_watts", "icu_training_load",
)


def _rejection_reason(icu: dict[str, Any], strava: StravaActivityRaw) -> str:
    """Mirror of intervals_matching._is_fuzzy_candidate that says which check failed."""
    icu_start = _parse_start_time(icu)
    strava_start = _parse_start_time(strava.raw_data)
    if icu_start is None or strava_start is None:
        return f"no start time (icu={icu_start}, strava={strava_start})"
    start_diff = abs((strava_start - icu_start).total_seconds())
    if start_diff > START_TIME_TOLERANCE.total_seconds():
        return f"start differs by {start_diff:.0f}s (limit {START_TIME_TOLERANCE.total_seconds():.0f}s)"

    icu_duration = _extract_duration_seconds(icu)
    strava_duration = _extract_duration_seconds(strava.raw_data)
    if strava_duration is not None and icu_duration is not None:
        tolerance = max(DURATION_ABS_FLOOR_SECONDS, strava_duration * DURATION_RELATIVE_TOLERANCE)
        if abs(strava_duration - icu_duration) > tolerance:
            return f"duration {icu_duration:.0f}s vs {strava_duration:.0f}s (limit ±{tolerance:.0f}s)"

    icu_distance = _extract_distance_meters(icu)
    strava_distance = _extract_distance_meters(strava.raw_data)
    if strava_distance and icu_distance:
        tolerance = max(DISTANCE_ABS_FLOOR_METERS, strava_distance * DISTANCE_RELATIVE_TOLERANCE)
        if abs(strava_distance - icu_distance) > tolerance:
            return f"distance {icu_distance:.0f}m vs {strava_distance:.0f}m (limit ±{tolerance:.0f}m)"

    return "CANDIDATE"


async def diagnose(athlete_id: int, day: date) -> None:
    await db_manager.connect()
    db = db_manager.db

    athlete = await AthleteRepository(db).get_athlete(athlete_id)
    if athlete is None:
        raise SystemExit(f"Athlete {athlete_id} not found")
    client = IntervalsIcuClient.from_athlete_settings(athlete.settings)

    day_dt = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc)
    strava_activities = await WorkoutRepository(db).get_activities_for_date(athlete_id, day_dt)
    icu_activities = await client.get_activities(day, day)

    print(f"\n== Strava activities in sidekick's cache for {day} ({len(strava_activities)}) ==")
    for s in strava_activities:
        r = s.raw_data
        print(
            f"  {s.activity_id}  {r.get('name')!r}  start={r.get('start_date')}  "
            f"moving={r.get('moving_time')}  elapsed={r.get('elapsed_time')}  distance={r.get('distance')}"
        )

    print(f"\n== Intervals.icu activities from the API for {day} ({len(icu_activities)}) ==")
    for icu in icu_activities:
        print("  " + "  ".join(f"{k}={icu.get(k)!r}" for k in _ICU_FIELDS))
        for s in strava_activities:
            print(f"      vs Strava {s.activity_id}: {_rejection_reason(icu, s)}")

    print("\n== What matching would decide now ==")
    matches = match_intervals_activities(icu_activities, strava_activities, athlete_id, datetime.now(timezone.utc))
    for m in matches:
        print(f"  icu {m.intervals_activity_id} -> strava {m.strava_activity_id} ({m.match_method})")

    print("\n== Stored intervals_activities records for these Intervals.icu ids ==")
    ids = [str(icu.get("id")) for icu in icu_activities]
    async for doc in db["intervals_activities"].find(
        {"athlete_id": athlete_id, "intervals_activity_id": {"$in": ids}}
    ):
        print(
            f"  icu {doc.get('intervals_activity_id')} -> strava {doc.get('strava_activity_id')} "
            f"({doc.get('match_method')}, fetched {doc.get('fetched_at')})"
        )

    await db_manager.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--athlete-id", type=int, default=app_settings.dev_athlete_id)
    parser.add_argument("--date", type=date.fromisoformat, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()

    if args.athlete_id is None:
        raise SystemExit("--athlete-id is required (or set DEV_ATHLETE_ID)")

    try:
        asyncio.run(diagnose(args.athlete_id, args.date))
    except IntervalsIcuNotConfigured as e:
        raise SystemExit(f"Cannot diagnose: {e}") from e


if __name__ == "__main__":
    main()
