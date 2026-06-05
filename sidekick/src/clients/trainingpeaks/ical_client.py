import logging
import re
from dataclasses import dataclass
from datetime import date, timedelta

import httpx
from icalendar import Calendar, Event

logger = logging.getLogger(__name__)

# TrainingPeaks CATEGORIES → sidekick sport type
_CATEGORY_MAP: dict[str, str] = {
    "Bike": "cycling",
    "Cycling": "cycling",
    "Run": "running",
    "Running": "running",
    "Swim": "other",
    "Swimming": "other",
    "Strength": "strength",
    "StrengthTraining": "strength",
    "CrossCountrySkiing": "skiing_cross",
    "AlpineSkiing": "skiing_alpine",
    "Rest": "day_off",
    "DayOff": "day_off",
}


@dataclass
class TPPlannedWorkout:
    uid: str
    date: date
    sport_type: str
    name: str
    description: str | None
    duration_min: int | None


def _parse_duration_min(component: Event) -> int | None:
    duration = component.get("DURATION")
    if duration is None:
        return None
    td: timedelta = duration.dt  # type: ignore[attr-defined]
    total_seconds = int(td.total_seconds())
    return total_seconds // 60 if total_seconds > 0 else None


def _parse_sport_type(component: Event) -> str:
    categories = component.get("CATEGORIES")
    if categories is None:
        return "other"
    # CATEGORIES can be a single object or a list; normalise to a flat list of strings
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
    return "other"


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
            # DTSTART may be a datetime; normalise to date
            if hasattr(event_date, "date"):
                event_date = event_date.date()

            if not (start_date <= event_date <= end_date):
                continue

            uid = str(component.get("UID", ""))
            summary = str(component.get("SUMMARY", "Planned workout"))
            description_raw = component.get("DESCRIPTION")
            description = str(description_raw).strip() if description_raw else None

            logger.debug(
                "VEVENT fields: %s",
                {k: str(v)[:120] for k, v in component.items()},
            )

            workouts.append(
                TPPlannedWorkout(
                    uid=uid,
                    date=event_date,
                    sport_type=_parse_sport_type(component),
                    name=summary,
                    description=description or None,
                    duration_min=_parse_duration_min(component),
                )
            )

        logger.info(
            "Fetched %d TP planned workouts for %s–%s",
            len(workouts),
            start_date,
            end_date,
        )
        return workouts
