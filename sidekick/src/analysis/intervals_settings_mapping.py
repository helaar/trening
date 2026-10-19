"""Map Intervals.icu sport settings onto sidekick's AthleteSettings shapes.

Intervals.icu is the master for FTP, LTHR, max HR and zones; sidekick pulls them
on demand (the athlete previews and confirms each sync in the settings page).

Intervals.icu stores zones as a list of upper bounds: power zones as % of FTP
(the last one is typically 999, i.e. open-ended), HR zones as absolute bpm (the
last one is max HR). Sidekick's ZoneDefinition is an inclusive [min, max] range
matched first-to-last, so zones are made contiguous (min = previous max): a value
exactly on a boundary lands in the lower zone, as it does in Intervals.icu, and
fractional values never fall into a gap between zones. The last zone is always
open-ended so efforts above max HR or 999% FTP are still classified.

Measured date/activity are never synced: Intervals.icu doesn't carry them, and
they hold notes the athlete entered (which test, when). The caller keeps the
saved values.

Intervals.icu keeps sport-specific HR zones; sidekick has one HR set for all
sports, taken from the Ride (bike/default) sport settings by the caller.
"""

from typing import Any

from models.athlete import HeartRateSettings, SportSettings, ZoneDefinition

class IntervalsSettingsIncomplete(ValueError):
    """Intervals.icu's sport settings lack a value sidekick requires."""


def find_sport_settings(entries: list[dict[str, Any]], sport_type: str) -> dict[str, Any]:
    """The sport-settings entry whose `types` include sport_type (e.g. "Ride")."""
    for entry in entries:
        if sport_type in (entry.get("types") or []):
            return entry
    raise IntervalsSettingsIncomplete(
        f"Intervals.icu has no sport settings for activity type {sport_type}"
    )


def _zones_from_upper_bounds(
    upper_bounds: list[int | float], names: list[str] | None, scale: float | None
) -> list[ZoneDefinition]:
    names = names or []
    zones: list[ZoneDefinition] = []
    lower = 0
    for i, raw_upper in enumerate(upper_bounds):
        value = raw_upper * scale / 100 if scale is not None else raw_upper
        # Half-up, not Python's banker's rounding: 105% of 250 W is 263 W in Intervals.icu.
        upper = int(value + 0.5)
        is_last = i == len(upper_bounds) - 1
        name = names[i] if i < len(names) and names[i] else f"Z{i + 1}"
        zones.append(ZoneDefinition(name=name, min=lower, max=None if is_last else upper))
        lower = upper
    return zones


def map_power_settings(
    raw: dict[str, Any], sport_type: str
) -> SportSettings:
    """FTP and power zones (in watts) from one Intervals.icu sport-settings entry."""
    ftp = raw.get("ftp")
    if not ftp:
        raise IntervalsSettingsIncomplete(f"Intervals.icu has no FTP set for {sport_type}")
    power_zones = raw.get("power_zones")
    if not power_zones:
        raise IntervalsSettingsIncomplete(
            f"Intervals.icu has no power zones set for {sport_type}"
        )
    return SportSettings(
        ftp=ftp,
        power_zones=_zones_from_upper_bounds(power_zones, raw.get("power_zone_names"), ftp),
    )


def map_heart_rate_settings(
    raw: dict[str, Any], sport_type: str
) -> HeartRateSettings:
    """LTHR, max HR and HR zones (in bpm) from one Intervals.icu sport-settings entry."""
    lthr = raw.get("lthr")
    hr_zones = raw.get("hr_zones")
    if not lthr:
        raise IntervalsSettingsIncomplete(f"Intervals.icu has no LTHR set for {sport_type}")
    if not hr_zones:
        raise IntervalsSettingsIncomplete(f"Intervals.icu has no HR zones set for {sport_type}")
    max_hr = raw.get("max_hr") or hr_zones[-1]
    return HeartRateSettings(
        max=max_hr,
        lt=lthr,
        hr_zones=_zones_from_upper_bounds(hr_zones, raw.get("hr_zone_names"), None),
    )

