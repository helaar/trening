"""Build a preview of athlete settings as Intervals.icu currently has them.

Read-only: nothing is written here. The settings page shows the athlete a diff
against their saved settings, and saving goes through the normal settings PATCH.
"""

from typing import Literal

from analysis.intervals_settings_mapping import (
    find_sport_settings,
    map_heart_rate_settings,
    map_power_settings,
)
from clients.intervals_icu.client import IntervalsIcuClient
from models.athlete import AthleteSettings, HeartRateSettings, SportSettings

SyncSection = Literal["cycling", "running", "heart_rate"]

# Heart rate uses the bike settings: sidekick has one HR zone set for all sports.
INTERVALS_SPORT_TYPE: dict[SyncSection, str] = {
    "cycling": "Ride",
    "running": "Run",
    "heart_rate": "Ride",
}


async def preview_from_intervals(
    settings: AthleteSettings, section: SyncSection
) -> SportSettings | HeartRateSettings:
    """The section's settings as Intervals.icu has them, mapped to sidekick's shape.

    Raises IntervalsIcuNotConfigured, IntervalsIcuError (incl. rate limiting),
    httpx.HTTPError, or IntervalsSettingsIncomplete; the API layer maps them to
    HTTP errors.
    """
    client = IntervalsIcuClient.from_athlete_settings(settings)
    sport_type = INTERVALS_SPORT_TYPE[section]
    raw = find_sport_settings(await client.get_sport_settings(), sport_type)
    match section:
        case "heart_rate":
            proposed = map_heart_rate_settings(raw, sport_type)
        case "cycling" | "running":
            proposed = map_power_settings(raw, sport_type)
    current = getattr(settings, section)
    if current is None:
        return proposed
    # Measured date/activity are the athlete's own notes; Intervals.icu has no equivalent.
    return proposed.model_copy(
        update={
            "measured_activity": current.measured_activity,
            "measured_date": current.measured_date,
        }
    )
