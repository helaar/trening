import re
from datetime import timedelta


_PATTERN = re.compile(
    r"^P"
    r"(?:(\d+)W)?"
    r"(?:(\d+)D)?"
    r"(?:T"
    r"(?:(\d+)H)?"
    r"(?:(\d+)M)?"
    r"(?:(\d+(?:\.\d+)?)S)?"
    r")?$"
)


def parse_iso8601_duration(value: str) -> timedelta:
    """Parse an ISO 8601 duration string into a timedelta.

    Supports weeks (P1W), days (P7D), hours/minutes/seconds (PT1H30M),
    and combinations thereof. Raises ValueError on unrecognised input.
    """
    m = _PATTERN.match(value)
    if not m or not any(m.groups()):
        raise ValueError(f"Unrecognised ISO 8601 duration: {value!r}")
    weeks, days, hours, minutes, seconds = m.groups(default="0")
    return timedelta(
        weeks=int(weeks),
        days=int(days),
        hours=int(hours),
        minutes=int(minutes),
        seconds=float(seconds),
    )
