"""Backfill/repair: cross-reference existing strava_activities to Intervals.icu.

Populates the `intervals_activities` collection for activities that already
exist in `strava_activities`, so historical activities don't have to wait on
an organic re-fetch to get an Intervals.icu cross-reference. Additive only —
does not touch strava_activities or workout_analyses, and is safe to re-run
(store_activity upserts on athlete_id + intervals_activity_id).

Ongoing maintenance tool, not a one-time script: this is also the fix-up
step after any bug in the matching pipeline (e.g. a period where
get_activities_for_date fed fuzzy-matching an empty candidate pool and
downgraded good matches to "ambiguous") — re-running it for the affected
date range re-derives the correct cross-references once the underlying bug
is fixed.

Run:
  cd sidekick
  PYTHONPATH=src uv run python scripts/backfill_intervals_matches.py \\
      --athlete-id 12345 --start 2024-01-01 --end 2026-07-30
"""

import argparse
import asyncio
import logging
from datetime import date, datetime, timezone

from clients.intervals_icu.client import IntervalsIcuClient, IntervalsIcuNotConfigured
from config import settings as app_settings
from database.athlete_repository import AthleteRepository
from database.intervals_activity_repository import IntervalsActivityRepository
from database.mongodb import db_manager
from database.workout_repository import WorkoutRepository
from services.intervals_matching import match_intervals_activities

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def backfill(athlete_id: int, start: date, end: date) -> None:
    await db_manager.connect()
    db = db_manager.db

    athlete_repo = AthleteRepository(db)
    workout_repo = WorkoutRepository(db)
    intervals_repo = IntervalsActivityRepository(db)

    athlete = await athlete_repo.get_athlete(athlete_id)
    if athlete is None:
        raise SystemExit(f"Athlete {athlete_id} not found")

    client = IntervalsIcuClient.from_athlete_settings(athlete.settings)

    range_start = datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc)
    range_end = datetime.combine(end, datetime.min.time(), tzinfo=timezone.utc)

    strava_activities = await workout_repo.get_activities_for_range(athlete_id, range_start, range_end)
    logger.info("Loaded %d Strava activities for %s..%s", len(strava_activities), start, end)

    intervals_activities = await client.get_activities(start, end)
    logger.info("Fetched %d Intervals.icu activities for %s..%s", len(intervals_activities), start, end)

    matches = match_intervals_activities(
        intervals_activities, strava_activities, athlete_id, datetime.now(timezone.utc)
    )

    counts: dict[str, int] = {}
    for match in matches:
        await intervals_repo.store_activity(match)
        counts[match.match_method] = counts.get(match.match_method, 0) + 1

    logger.info("Stored %d Intervals.icu activity records: %s", len(matches), counts)
    await db_manager.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--athlete-id", type=int, default=app_settings.dev_athlete_id)
    parser.add_argument("--start", type=date.fromisoformat, required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", type=date.fromisoformat, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()

    if args.athlete_id is None:
        raise SystemExit("--athlete-id is required (or set DEV_ATHLETE_ID)")

    try:
        asyncio.run(backfill(args.athlete_id, args.start, args.end))
    except IntervalsIcuNotConfigured as e:
        raise SystemExit(f"Cannot backfill: {e}") from e


if __name__ == "__main__":
    main()
