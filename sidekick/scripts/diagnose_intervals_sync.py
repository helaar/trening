"""Read-only diagnostic: where is the Strava -> Intervals.icu sync actually breaking?

Prints, per Strava activity in a date range, three things side by side:
  - Whether a matching intervals_activities document exists (the raw
    Intervals.icu cross-reference, populated by services.intervals_matching)
  - What icu_rpe/icu_average_watts look like in that raw document, if any
  - What's cached in workout_analyses (intervals_sync_status/intervals_rpe/
    whether zones is populated) — the value the UI is actually showing

This does not change anything — no writes, no calls to the Intervals.icu API.
It's for isolating whether a "not showing up" report is a matching problem
(no intervals_activities doc at all), a data problem (doc exists but no
icu_rpe on it — meaning Intervals.icu itself doesn't have it, or the athlete
hasn't logged it there), or a caching problem (raw doc has the data, but
workout_analyses is stale).

Run:
  cd sidekick
  PYTHONPATH=src uv run python scripts/diagnose_intervals_sync.py \\
      --athlete-id 12345 --days-back 3
"""

import argparse
import asyncio
from datetime import datetime, timedelta, timezone

from config import settings as app_settings
from database.athlete_repository import AthleteRepository
from database.intervals_activity_repository import IntervalsActivityRepository
from database.mongodb import db_manager
from database.workout_repository import WorkoutRepository


async def diagnose(athlete_id: int, days_back: int) -> None:
    await db_manager.connect()
    db = db_manager.db

    athlete_repo = AthleteRepository(db)
    workout_repo = WorkoutRepository(db)
    intervals_repo = IntervalsActivityRepository(db)

    athlete = await athlete_repo.get_athlete(athlete_id)
    if athlete is None:
        raise SystemExit(f"Athlete {athlete_id} not found")

    icu_settings = athlete.settings.intervals_icu if athlete.settings else None
    print(f"Intervals.icu configured: {bool(icu_settings and icu_settings.api_key)}")

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days_back)
    activities = await workout_repo.get_activities_for_range(athlete_id, start, end)
    print(f"\n{len(activities)} Strava activities in the last {days_back} days\n")

    header = f"{'activity_id':>12} | {'date':>16} | {'name':<24} | {'intervals_match':<15} | {'icu_rpe':>7} | {'cached_status':<14} | {'cached_rpe':>10} | {'cached_zones'}"
    print(header)
    print("-" * len(header))

    for activity in activities:
        raw = await intervals_repo.get_by_strava_activity_id(athlete_id, activity.activity_id)
        if raw is None:
            match_info = "NO MATCH"
            icu_rpe = "-"
        else:
            match_info = raw.match_method
            icu_rpe = str(raw.raw_data.get("icu_rpe"))

        cached = await workout_repo.get_analysis(athlete_id, activity.activity_id)
        if cached is None:
            cached_status, cached_rpe, cached_zones = "no cache", "-", "-"
        else:
            data, _, _ = cached
            cached_status = str(data.get("intervals_sync_status"))
            cached_rpe = str(data.get("intervals_rpe"))
            cached_zones = "present" if data.get("zones") else "None"

        name = (activity.raw_data.get("name") or "")[:24]
        date_str = activity.activity_date.strftime("%Y-%m-%d %H:%M")
        print(
            f"{activity.activity_id:>12} | {date_str:>16} | {name:<24} | {match_info:<15} | "
            f"{icu_rpe:>7} | {cached_status:<14} | {cached_rpe:>10} | {cached_zones}"
        )

    await db_manager.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--athlete-id", type=int, default=app_settings.dev_athlete_id)
    parser.add_argument("--days-back", type=int, default=3)
    args = parser.parse_args()

    if args.athlete_id is None:
        raise SystemExit("--athlete-id is required (or set DEV_ATHLETE_ID)")

    asyncio.run(diagnose(args.athlete_id, args.days_back))


if __name__ == "__main__":
    main()
