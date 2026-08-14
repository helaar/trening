"""Link planned sessions to the completed workout that fulfilled them.

Intervals.icu already pairs a calendar event with the activity that completed
it (auto-paired by date/sport, correctable by the athlete in their own UI) and
exposes this as `paired_event_id` on the activity object returned by its
`/activities` endpoint. `WorkoutAnalysisService._sync_intervals_activities_for_date`
already fetches and stores that raw activity data (as `IntervalsActivityRaw.raw_data`)
for essentially every analyzed workout, so this module just reads the field
back out rather than re-deriving the pairing with our own date/sport/duration
heuristics.
"""

from collections import defaultdict
from datetime import datetime
from typing import Any

from database.intervals_activity_repository import IntervalsActivityRepository
from models.plan import PlannedActivity


def group_workouts_by_date(workouts: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group workout-analysis dicts by their session's calendar date (YYYY-MM-DD)."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for workout in workouts:
        start_time = workout.get("session", {}).get("start_time")
        if start_time is None:
            continue
        date = (
            start_time.strftime("%Y-%m-%d")
            if isinstance(start_time, datetime)
            else str(start_time)[:10]
        )
        grouped[date].append(workout)
    return grouped


async def attach_matches(
    athlete_id: int,
    plans: list[PlannedActivity],
    workouts: list[dict[str, Any]],
    intervals_activity_repo: IntervalsActivityRepository,
) -> None:
    """Set `matched_activity_id` on each plan fulfilled by one of `workouts`.

    Workouts without a stored Intervals.icu cross-reference (not yet synced,
    or Intervals.icu not connected) are silently skipped, same degrade-
    gracefully convention `workout_analysis.py` already uses elsewhere.
    """
    plans_by_id = {plan.id: plan for plan in plans}
    if not plans_by_id:
        return

    for workout in workouts:
        activity_id = workout.get("activity_id")
        if activity_id is None:
            continue
        record = await intervals_activity_repo.get_by_strava_activity_id(athlete_id, activity_id)
        if record is None:
            continue
        paired_event_id = record.raw_data.get("paired_event_id")
        if paired_event_id is None:
            continue
        plan = plans_by_id.get(str(paired_event_id))
        if plan is not None:
            plan.matched_activity_id = activity_id


async def attach_matches_for_range(
    athlete_id: int,
    plans_by_date: dict[str, list[PlannedActivity]],
    workouts_by_date: dict[str, list[dict[str, Any]]],
    intervals_activity_repo: IntervalsActivityRepository,
) -> None:
    """Run `attach_matches` per date for a set of plans/workouts spanning a range."""
    for date, plans in plans_by_date.items():
        await attach_matches(
            athlete_id, plans, workouts_by_date.get(date, []), intervals_activity_repo
        )
