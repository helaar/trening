import logging
import re
from dataclasses import dataclass
from datetime import date, timedelta

import httpx
from icalendar import Calendar, Event

logger = logging.getLogger(__name__)

# Sport prefix in SUMMARY → sidekick sport type
_SUMMARY_SPORT_MAP: dict[str, str] = {
    "bike": "cycling",
    "cycling": "cycling",
    "run": "running",
    "running": "running",
    "swim": "other",
    "swimming": "other",
    "strength": "strength",
    "strengthtraining": "strength",
    "crosscountryskiing": "skiing_cross",
    "alpineskiing": "skiing_alpine",
    "rest": "day_off",
    "dayoff": "day_off",
}

# TrainingPeaks CATEGORIES → sidekick sport type (fallback if CATEGORIES present)
_CATEGORY_MAP: dict[str, str] = {k.title(): v for k, v in _SUMMARY_SPORT_MAP.items()}

# "Planned Time: H:MM" or "Planned Time: HH:MM" in DESCRIPTION
_PLANNED_TIME_RE = re.compile(r"Planned Time:\s*(\d+):(\d{2})", re.IGNORECASE)


@dataclass
class TPPlannedWorkout:
    uid: str
    date: date
    sport_type: str
    name: str
    description: str | None
    duration_min: int | None


def _parse_sport_and_name(summary: str) -> tuple[str, str]:
    """Split 'Sport: [CoachPlan: ]Title' into (sport_type, title)."""
    parts = [p.strip() for p in summary.split(":")]
    sport_type = _SUMMARY_SPORT_MAP.get(parts[0].lower().replace(" ", ""), "other")
    # Drop the sport segment; also drop any segment that looks like a coaching plan name
    # (heuristic: segment before the last one that contains no lowercase letters is a label)
    title_parts = parts[1:]
    # If there are 2+ remaining segments, the first is likely a plan/coach name — drop it
    if len(title_parts) >= 2:
        title_parts = title_parts[1:]
    name = ": ".join(title_parts).strip() or summary
    return sport_type, name


def _parse_duration_from_description(description: str | None) -> int | None:
    if not description:
        return None
    m = _PLANNED_TIME_RE.search(description)
    if not m:
        return None
    hours, minutes = int(m.group(1)), int(m.group(2))
    total = hours * 60 + minutes
    return total if total > 0 else None


def _parse_sport_type_from_categories(component: Event) -> str | None:
    categories = component.get("CATEGORIES")
    if categories is None:
        return None
    cats: list[str] = []
    if hasattr(categories, "cats"):
        cats = [str(c) for c in categories.cats]
    elif isinstance(categories, (list, tuple)):
        for item in categories:
            if hasattr(item, "cats"):
                cats.extend(str(c) for c in item.cats)
            else:
                cats.append(str(item))
    else:
        cats = [str(categories)]
    for cat in cats:
        mapped = _CATEGORY_MAP.get(cat)
        if mapped:
            return mapped
    return None


def _parse_duration_min(component: Event) -> int | None:
    duration = component.get("DURATION")
    if duration is None:
        return None
    td: timedelta = duration.dt  # type: ignore[attr-defined]
    total_seconds = int(td.total_seconds())
    return total_seconds // 60 if total_seconds > 0 else None


class TrainingPeaksICalClient:
    """Read-only client that fetches planned workouts from a TrainingPeaks iCal URL."""

    async def get_planned_workouts(
        self,
        ical_url: str,
        start_date: date,
        end_date: date,
    ) -> list[TPPlannedWorkout]:
        # webcal:// is just http(s):// for calendar apps; always fetch over https
        fetch_url = re.sub(r"^webcal://", "https://", ical_url, flags=re.IGNORECASE)

        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            response = await client.get(fetch_url)
            response.raise_for_status()

        cal = Calendar.from_ical(response.content)
        workouts: list[TPPlannedWorkout] = []

        for component in cal.walk():
            if component.name != "VEVENT":
                continue

            dtstart = component.get("DTSTART")
            if dtstart is None:
                continue

            event_date: date = dtstart.dt  # type: ignore[attr-defined]
            if hasattr(event_date, "date"):
                event_date = event_date.date()

            if not (start_date <= event_date <= end_date):
                continue

            uid = str(component.get("UID", ""))
            raw_summary = str(component.get("SUMMARY", "Planned workout"))
            description_raw = component.get("DESCRIPTION")
            description = str(description_raw).strip() if description_raw else None

            sport_type, name = _parse_sport_and_name(raw_summary)
            # CATEGORIES takes precedence over SUMMARY prefix if present
            sport_type = _parse_sport_type_from_categories(component) or sport_type

            # Duration: prefer iCal DURATION field, fall back to "Planned Time:" in description
            duration_min = _parse_duration_min(component) or _parse_duration_from_description(description)

            workouts.append(
                TPPlannedWorkout(
                    uid=uid,
                    date=event_date,
                    sport_type=sport_type,
                    name=name,
                    description=description or None,
                    duration_min=duration_min,
                )
            )

        logger.info(
            "Fetched %d TP planned workouts for %s–%s",
            len(workouts),
            start_date,
            end_date,
        )
        return workouts

