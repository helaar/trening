from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo


def to_athlete_tz(dt: datetime, tz_str: str) -> datetime:
    """Convert an aware UTC datetime to the athlete's local timezone."""
    return dt.astimezone(ZoneInfo(tz_str))


def convert_datetimes_in_obj(obj: Any, tz_str: str) -> Any:
    """Recursively convert all datetime values in a dict/list structure to athlete local tz."""
    if isinstance(obj, datetime):
        if obj.tzinfo is None:
            obj = obj.replace(tzinfo=timezone.utc)
        return obj.astimezone(ZoneInfo(tz_str))
    if isinstance(obj, dict):
        return {k: convert_datetimes_in_obj(v, tz_str) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_datetimes_in_obj(item, tz_str) for item in obj]
    return obj
