"""Daily LLM analysis crew for a single training day."""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from crewai import LLM, Agent, Crew, Task
from crewai.tools import BaseTool
from pydantic import BaseModel, ValidationError

from config import settings
from crew.prompt_logging import capture_prompt_log, drain_prompt_log
from crew.usage import collect_run_usage
from models.athlete import Athlete
from models.crew_definition import AgentDoc, PhilosophyDoc, TaskDoc
from utils.datetime_utils import convert_datetimes_in_obj
from models.crew_outputs import (
    CoachingOutput,
    MemoryExtractionOutput,
    RestitutionAnalysisOutput,
    WorkoutAnalysisOutput,
)
from analysis.memory_relevance import (
    DayContext,
    select_recovery_memories,
    select_relevant_memories,
)
from models.memory import Memory
from models.daily_entry import DailyEntry
from models.plan import PlannedActivity

logger = logging.getLogger(__name__)

_RESTITUTION_WINDOW_DAYS = 14

# The deterministic weekly assessment is specific to this philosophy; other
# philosophies get no weekly classification.
_POLARIZED_SLUG = "polarized_80_20"


@dataclass
class DailyAnalysisInput:
    athlete: Athlete
    workout_analyses: list[dict[str, Any]]
    planned_activities: list[PlannedActivity]
    date: str
    daily_entries: list[DailyEntry] = field(default_factory=list)
    recent_workout_analyses: list[dict[str, Any]] = field(default_factory=list)
    active_memories: list[Memory] = field(default_factory=list)
    upcoming_races: list[PlannedActivity] = field(default_factory=list)
    agents: dict[str, AgentDoc] = field(default_factory=dict)
    tasks: dict[str, TaskDoc] = field(default_factory=dict)
    philosophy: PhilosophyDoc | None = None


def _philosophy_payload(philosophy: PhilosophyDoc | None) -> dict[str, str] | None:
    """Render a philosophy document as the dict shape embedded in LLM prompts."""
    if not philosophy or not philosophy.display_name:
        return None
    return {
        "name": philosophy.display_name,
        "intensity_targets": philosophy.intensity_targets,
        "coach_guidance": philosophy.coach_guidance,
        "analyst_guidance": philosophy.analyst_guidance,
    }


def require_definition(mapping: dict[str, Any], name: str, kind: str) -> Any:
    """Fetch a seeded definition by name, failing fast with a helpful message."""
    try:
        return mapping[name]
    except KeyError:
        raise RuntimeError(
            f"No {kind} '{name}' found in crew_definitions. "
            "Run scripts/seed_crew_definitions.py to seed defaults."
        ) from None


def _compute_intensity_distribution(analysis: dict[str, Any]) -> dict[str, Any]:
    """Compute low/moderate/high % from zone list positions (philosophy-neutral).

    Zone position mapping (0-based):
      0-1 → low (Zone 1 Recovery + Zone 2 Endurance)
        2 → moderate (Zone 3 Tempo)
       3+ → high (Zone 4 Threshold and above)
    """
    zones_data = analysis.get("zones", {})

    def _extract(zone_list: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not zone_list:
            return None
        total_seconds = sum(z.get("seconds", 0) for z in zone_list)
        if not total_seconds:
            return None
        low = sum(z.get("seconds", 0) for z in zone_list[:2])
        moderate = zone_list[2].get("seconds", 0) if len(zone_list) > 2 else 0
        high = sum(z.get("seconds", 0) for z in zone_list[3:])
        return {
            "low_pct": round(low / total_seconds * 100, 1),
            "moderate_pct": round(moderate / total_seconds * 100, 1),
            "high_pct": round(high / total_seconds * 100, 1),
        }

    power_zones = (zones_data.get("power_zones") or {}).get("zones") if zones_data else None
    if power_zones:
        dist = _extract(power_zones)
        if dist:
            return {**dist, "basis": "power"}

    hr_zones = (zones_data.get("hr_zones") or {}).get("zones") if zones_data else None
    if hr_zones:
        dist = _extract(hr_zones)
        if dist:
            return {**dist, "basis": "hr"}

    return {"low_pct": None, "moderate_pct": None, "high_pct": None, "basis": "unavailable"}


def _session_start_date(analysis: dict[str, Any]) -> str | None:
    """YYYY-MM-DD of a session's start_time (datetime or string), or None."""
    start_time = analysis.get("session", {}).get("start_time")
    if not start_time:
        return None
    return start_time.date().isoformat() if hasattr(start_time, "date") else str(start_time)[:10]


# Activities eligible for the HR-fallback path. Cross-country skiing maps to the
# "skiing" category; alpine skiing falls to "other" and is excluded. The power path is
# NOT sport-gated (a power meter already implies an endurance effort), so MTB/gravel/etc.
# still classify — this list only guards the HR path against low-HR strength volume.
_ENDURANCE_CATEGORIES = {"cycling", "running", "skiing"}
_ENDURANCE_OTHER_SPORTS = {"swim", "walk", "hike", "rowing"}


def _is_endurance_sport(analysis: dict[str, Any]) -> bool:
    """True for endurance activities eligible for HR-based classification."""
    session = analysis.get("session", {})
    category = session.get("category")
    if category in _ENDURANCE_CATEGORIES:
        return True
    if category == "other":
        return (session.get("sport") or "").lower() in _ENDURANCE_OTHER_SPORTS
    return False


def _usable_power_zones(analysis: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Power zones if there is a resolvable gray zone (>=3 zones with a Z3 lower edge)."""
    zones = ((analysis.get("zones") or {}).get("power_zones") or {}).get("zones") or []
    if len(zones) >= 3 and zones[2].get("lower") is not None:
        return zones
    return None


def _coarse_bands(zones: list[dict[str, Any]], basis: str) -> dict[str, Any]:
    """Low/moderate/high minutes from raw zone seconds (Z1-2 / Z3 / Z4+), no tolerance."""
    low_s = sum((z.get("seconds") or 0) for z in zones[:2])
    mod_s = zones[2].get("seconds") or 0
    high_s = sum((z.get("seconds") or 0) for z in zones[3:])
    return {
        "low": low_s / 60,
        "moderate": mod_s / 60,
        "high": high_s / 60,
        "effective_gray_zone": mod_s / 60,
        "basis": basis,
    }


def _power_session_bands(
    power_zones: list[dict[str, Any]], histogram: Any, depth_frac: float
) -> dict[str, Any]:
    """Low/moderate/high minutes from the power histogram with a graduated gray zone.

    Only Z3 time in the upper part of the zone (above ``depth_frac`` of the way in) counts
    as drift; time in the lower part is treated as effectively easy. Without a usable
    histogram, falls back to coarse zone seconds (all Z3 counts as moderate).
    """
    seconds = histogram.get("seconds") if isinstance(histogram, dict) else None
    bucket_width = histogram.get("bucket_width") if isinstance(histogram, dict) else None
    if not seconds or not bucket_width or bucket_width <= 0:
        return _coarse_bands(power_zones, "power-zones")

    z3 = power_zones[2]
    z3_lower = z3.get("lower")
    z3_upper = z3.get("upper")  # None if Z3 is the open-ended top zone (unusual)
    min_watts = histogram.get("min_watts") or 0.0
    zone_span = (z3_upper - z3_lower) if z3_upper is not None else 0.0
    depth_watts = z3_lower + depth_frac * zone_span

    low_s = mod_tolerated_s = mod_drift_s = high_s = 0.0
    for i, sec in enumerate(seconds):
        if not sec:
            continue
        center = min_watts + (i + 0.5) * bucket_width
        if center < z3_lower:
            low_s += sec
        elif z3_upper is not None and center >= z3_upper:
            high_s += sec
        elif center >= depth_watts:
            mod_drift_s += sec
        else:
            mod_tolerated_s += sec

    # Border-hugging (lower-Z3) time is effectively easy.
    low_s += mod_tolerated_s

    return {
        "low": low_s / 60,
        "moderate": mod_drift_s / 60,
        "high": high_s / 60,
        "effective_gray_zone": mod_drift_s / 60,
        "basis": "power-histogram",
    }


def _qualifies_for_polarized_classification(analysis: dict[str, Any]) -> bool:
    """True if a session should be counted at all (not strength, has an endurance basis)."""
    if analysis.get("session", {}).get("category") == "strength":
        return False  # strength is not endurance training; ignore entirely
    # Power implies an endurance effort; otherwise require an endurance sport so a
    # low-HR non-endurance session cannot inflate easy volume.
    return _usable_power_zones(analysis) is not None or _is_endurance_sport(analysis)


def _classify_session(analysis: dict[str, Any], depth_frac: float) -> dict[str, float] | None:
    """Low/moderate/high/effective_gray_zone minutes for one qualifying session.

    Returns None when the session has no usable power or HR zones to classify against
    (its duration should be counted as unclassified by the caller).
    """
    power_zones = _usable_power_zones(analysis)
    if power_zones is not None:
        return _power_session_bands(power_zones, analysis.get("power_histogram"), depth_frac)
    hr_zones = ((analysis.get("zones") or {}).get("heart_rate_zones") or {}).get("zones") or []
    if len(hr_zones) >= 3:
        return _coarse_bands(hr_zones, "hr-zones")
    return None


def _day_bands(
    recent_analyses: list[dict[str, Any]], day_iso: str, depth_frac: float
) -> dict[str, Any]:
    """Aggregate low/moderate/high minutes for a single calendar day.

    ``trained`` is True as soon as a qualifying endurance session exists that day, even
    if it couldn't be classified — distinct from ``classified_minutes == 0``.
    """
    low = mod = high = 0.0
    trained = False
    for analysis in recent_analyses:
        if _session_start_date(analysis) != day_iso:
            continue
        if not _qualifies_for_polarized_classification(analysis):
            continue
        trained = True
        bands = _classify_session(analysis, depth_frac)
        if bands is None:
            continue
        low += bands["low"]
        mod += bands["moderate"]
        high += bands["high"]
    classified = low + mod + high
    return {
        "date": day_iso,
        "trained": trained,
        "classified_minutes": round(classified, 1),
        "low_min": round(low, 1),
        "moderate_min": round(mod, 1),
        "high_min": round(high, 1),
        "low_pct": round(low / classified * 100, 1) if classified else None,
        "moderate_pct": round(mod / classified * 100, 1) if classified else None,
        "high_pct": round(high / classified * 100, 1) if classified else None,
    }


def _fmt_day_minutes(day: dict[str, Any]) -> str:
    """Plain-language rundown of a day's classified minutes, e.g. '30m moderate, 45m easy'."""
    parts = [
        f"{day[key]:.0f}m {label}"
        for key, label in (("high_min", "hard"), ("moderate_min", "moderate"), ("low_min", "easy"))
        if day[key] >= 0.5
    ]
    if parts:
        return ", ".join(parts)
    return "an unclassified session" if day["trained"] else "no training"


def _day_over_day_note(today: dict[str, Any], dropped: dict[str, Any]) -> str:
    """Explain the day-to-day shift: today's addition vs. the day rolling out of the window.

    The 7-day window is contiguous, so this week's totals minus last week's totals equal
    today's minutes minus the dropped day's minutes for each band — an exact, cheap
    explanation without recomputing the whole prior week.
    """
    if not today["trained"] and not dropped["trained"]:
        return (
            "No training today, and no session rolled off the 7-day window either — "
            "the week's mix is unchanged from yesterday."
        )

    today_desc = _fmt_day_minutes(today) if today["trained"] else "no training"
    if not dropped["trained"]:
        return (
            f"Today: {today_desc}. No session rolled off the 7-day window, so today's "
            "training is the main driver of any shift."
        )

    dropped_desc = _fmt_day_minutes(dropped)
    note = (
        f"Today: {today_desc}. The {dropped['date']} session ({dropped_desc}) just rolled "
        "off the 7-day window, so the week's mix reflects both changes."
    )
    mod_delta = today["moderate_min"] - dropped["moderate_min"]
    if abs(mod_delta) >= 1:
        sign = "+" if mod_delta > 0 else "-"
        note += f" Moderate {sign}{abs(mod_delta):.0f} min this week as a result."
    return note


def _weekly_philosophy_assessment(
    recent_analyses: list[dict[str, Any]], end_date: str
) -> dict[str, Any]:
    """Deterministic polarized verdict over the trailing 7 days ending on end_date.

    Aggregates per-session bands (with graduated gray-zone tolerance) into a weekly
    distribution and a status + plain-language description the coach can paraphrase. All
    thresholds are config tunables so the verdict can be adjusted without code changes.

    Also reports today's own contribution and a day-over-day note explaining the shift
    from yesterday's window (today's session vs. the day that rolled off 7 days back).
    """
    end = date.fromisoformat(end_date)
    start = end - timedelta(days=6)
    start_iso, end_iso = start.isoformat(), end.isoformat()
    depth_frac = settings.polarized_gray_zone_depth_frac
    dropped_date = (start - timedelta(days=1)).isoformat()

    today_bands = _day_bands(recent_analyses, end_iso, depth_frac)
    dropped_bands = _day_bands(recent_analyses, dropped_date, depth_frac)

    low = mod = high = eff_gray = unclassified_min = weekly_tss = 0.0
    tss_seen = False
    session_count = 0
    days: set[str] = set()

    for analysis in recent_analyses:
        d = _session_start_date(analysis)
        if not d or not (start_iso <= d <= end_iso):
            continue
        if not _qualifies_for_polarized_classification(analysis):
            continue

        days.add(d)
        tss = analysis.get("metrics", {}).get("training_stress_score")
        if tss is not None:
            weekly_tss += tss
            tss_seen = True

        bands = _classify_session(analysis, depth_frac)
        if bands is None:
            unclassified_min += (analysis.get("session", {}).get("duration_sec") or 0) / 60
            continue
        low += bands["low"]
        mod += bands["moderate"]
        high += bands["high"]
        eff_gray += bands["effective_gray_zone"]
        session_count += 1

    classified = low + mod + high
    result: dict[str, Any] = {
        "window": f"{start_iso}..{end_iso}",
        "status": "insufficient_data",
        "description": "",
        "low_pct": None,
        "moderate_pct": None,
        "high_pct": None,
        "effective_gray_zone_min": round(eff_gray, 1),
        "classified_minutes": round(classified, 1),
        "unclassified_minutes": round(unclassified_min, 1),
        "days_with_training": len(days),
        "session_count": session_count,
        "weekly_tss": round(weekly_tss, 1) if tss_seen else None,
        "data_sufficiency": "insufficient",
        "today": today_bands,
        "day_over_day_note": _day_over_day_note(today_bands, dropped_bands),
    }

    if (
        classified < settings.polarized_min_classified_minutes
        or len(days) < settings.polarized_min_training_days
    ):
        result["description"] = (
            "Not enough classified endurance training this week to judge polarization."
        )
        return result

    low_pct = round(low / classified * 100, 1)
    mod_pct = round(mod / classified * 100, 1)
    high_pct = round(high / classified * 100, 1)
    result.update(low_pct=low_pct, moderate_pct=mod_pct, high_pct=high_pct)
    result["data_sufficiency"] = "sparse" if session_count < 3 else "ok"

    # 80/20 cares about easy volume: moderate is only a problem when it eats into easy.
    # Judge on rounded shares so the verdict matches the percentages shown to the athlete.
    low_i = round(low_pct)
    mod_i = round(mod_pct)
    target = settings.polarized_low_target_pct

    if low_i >= target or mod_i <= settings.polarized_min_moderate_pct:
        result["status"] = "polarized"
        if low_i >= target:
            result["description"] = (
                f"Easy volume is on target at {low_pct:.0f}% this week "
                f"({high_pct:.0f}% hard, {mod_pct:.0f}% moderate) — well polarized."
            )
        else:
            result["description"] = (
                f"Easy volume is a little below target at {low_pct:.0f}% this week, but not "
                f"from gray-zone drift (only {mod_pct:.0f}% moderate)."
            )
    elif low_i >= settings.polarized_gray_zone_low_pct:
        result["status"] = "mild_drift"
        result["description"] = (
            f"Some gray-zone drift: easy has slipped to {low_pct:.0f}% (target "
            f"≥{target:.0f}%), with {mod_pct:.0f}% of the week in the moderate range."
        )
    else:
        result["status"] = "gray_zone_week"
        result["description"] = (
            f"Gray-zone drift is eating into easy volume: only {low_pct:.0f}% easy this week "
            f"(target ≥{target:.0f}%), with {mod_pct:.0f}% in the moderate range."
        )
    return result


def _athlete_settings_summary(athlete: Athlete) -> dict[str, Any]:
    s = athlete.settings
    result: dict[str, Any] = {}
    if s.heart_rate:
        result["heart_rate"] = {
            "max_hr": s.heart_rate.max,
            "lactate_threshold_hr": s.heart_rate.lt,
            "zones": [{"name": z.name, "min": z.min, "max": z.max} for z in s.heart_rate.hr_zones],
        }
    if s.cycling:
        result["cycling"] = {
            "ftp_watts": s.cycling.ftp,
            "zones": [{"name": z.name, "min": z.min, "max": z.max} for z in s.cycling.power_zones],
        }
    if s.running:
        result["running"] = {
            "ftp_watts": s.running.ftp,
            "zones": [{"name": z.name, "min": z.min, "max": z.max} for z in s.running.power_zones],
        }
    result["timezone"] = s.timezone
    return result


def _build_timeline(
    start_date: str,
    end_date: str,
    daily_entries: list[DailyEntry],
    workout_analyses: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge restitution entries and workout analyses into a per-day timeline."""
    restitution_by_date: dict[str, dict[str, Any]] = {}
    for entry in daily_entries:
        if entry.restitution:
            r = entry.restitution
            day = {
                "sleep_hours": r.sleep_hours,
                "sleep_quality": r.sleep_quality,
                "hrv": r.hrv,
                "resting_hr": r.resting_hr,
                "readiness": r.readiness,
            }
            # Surface today's free-text comment as a one-off readiness signal;
            # historical comments are intentionally omitted so they are not
            # folded into multi-day baselines/trends.
            if entry.date == end_date and r.comment:
                day["comment"] = r.comment
            restitution_by_date[entry.date] = day

    training_by_date: dict[str, list[dict[str, Any]]] = {}
    for analysis in workout_analyses:
        session = analysis.get("session", {})
        start_time = session.get("start_time")
        if not start_time:
            continue
        day_str = (
            start_time.date().isoformat() if hasattr(start_time, "date") else str(start_time)[:10]
        )
        training_by_date.setdefault(day_str, []).append(analysis)

    def _aggregate_training(analyses: list[dict[str, Any]]) -> dict[str, Any]:
        tss_values = [
            a.get("metrics", {}).get("training_stress_score")
            for a in analyses
            if a.get("metrics", {}).get("training_stress_score") is not None
        ]
        if_values = [
            a.get("metrics", {}).get("intensity_factor")
            for a in analyses
            if a.get("metrics", {}).get("intensity_factor") is not None
        ]
        total_minutes = sum((a.get("session", {}).get("duration_sec") or 0) / 60 for a in analyses)
        sports = list(
            {
                a.get("session", {}).get("category") or a.get("session", {}).get("sport")
                for a in analyses
                if a.get("session", {}).get("category") or a.get("session", {}).get("sport")
            }
        )

        low_min = moderate_min = high_min = 0.0
        for a in analyses:
            dist = _compute_intensity_distribution(a)
            if dist.get("basis") == "unavailable":
                continue
            duration_min = (a.get("session", {}).get("duration_sec") or 0) / 60
            low_min += duration_min * (dist.get("low_pct") or 0) / 100
            moderate_min += duration_min * (dist.get("moderate_pct") or 0) / 100
            high_min += duration_min * (dist.get("high_pct") or 0) / 100

        return {
            "activity_count": len(analyses),
            "total_tss": round(sum(tss_values), 1) if tss_values else None,
            "avg_if": round(sum(if_values) / len(if_values), 3) if if_values else None,
            "total_minutes": round(total_minutes),
            "sports": sports,
            "intensity_distribution_minutes": {
                "low": round(low_min, 1),
                "moderate": round(moderate_min, 1),
                "high": round(high_min, 1),
            },
        }

    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    timeline = []
    current = start
    while current <= end:
        day_str = current.isoformat()
        training_analyses = training_by_date.get(day_str)
        timeline.append(
            {
                "date": day_str,
                "restitution": restitution_by_date.get(day_str),
                "training": _aggregate_training(training_analyses) if training_analyses else None,
            }
        )
        current += timedelta(days=1)

    # Precompute trailing cumulative TSS so the LLM can quote a 2/3-day load
    # figure directly instead of summing per-day values itself (a frequent
    # source of arithmetic errors and fabricated totals). Rest/no-training
    # days contribute 0. Null if the window extends before the data range.
    daily_tss = [
        entry["training"]["total_tss"] if entry["training"] else None for entry in timeline
    ]
    for i, entry in enumerate(timeline):
        for window in (2, 3):
            if i - window + 1 < 0:
                entry[f"rolling_tss_{window}d"] = None
            else:
                window_values = daily_tss[i - window + 1 : i + 1]
                entry[f"rolling_tss_{window}d"] = round(sum(v or 0 for v in window_values), 1)

    return timeline


class _WorkoutDataTool(BaseTool):
    name: str = "get_workout_data"
    description: str = (
        "Retrieve workout analyses and athlete threshold/zone settings for today. "
        "Returns JSON with 'athlete_settings' and 'workouts' keys, plus "
        "'training_philosophy' and a precomputed 'weekly_philosophy_assessment' "
        "(status/description + weekly intensity distribution) when a philosophy is set. "
        "Call this first."
    )
    _payload: str = ""

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, payload: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_payload", payload)

    def _run(self, **kwargs: Any) -> str:
        return self._payload


class _PlansDataTool(BaseTool):
    name: str = "get_plans_data"
    description: str = (
        "Retrieve today's planned activities and athlete settings. "
        "Returns JSON with 'planned_activities' and 'athlete_settings' keys, plus "
        "'training_philosophy' and a precomputed 'weekly_philosophy_assessment' "
        "(status/description + weekly intensity distribution) when a philosophy is set."
    )
    _payload: str = ""

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, payload: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_payload", payload)

    def _run(self, **kwargs: Any) -> str:
        return self._payload


class _RacesDataTool(BaseTool):
    name: str = "get_upcoming_races"
    description: str = (
        "Retrieve the athlete's season goal race and other upcoming races from "
        "the day being analyzed onward. Returns JSON with 'season_goal' (the "
        "race tagged 'seasongoal', or null if none is planned) and "
        "'upcoming_races' (a list of races tagged 'race', sorted by date, each "
        "with 'date', 'name', 'sport', and 'days_until' — the number of days "
        "from the day being analyzed to the race)."
    )
    _payload: str = ""

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, payload: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_payload", payload)

    def _run(self, **kwargs: Any) -> str:
        return self._payload


class _MemoryContextTool(BaseTool):
    name: str = "get_athlete_memories"
    description: str = (
        "Retrieve the memories most relevant to this athlete today — durable observations "
        "about patterns, habits, risks, and goals built up over previous sessions. "
        "Returns JSON with an 'active_memories' list, each entry containing scope, "
        "category, content, and confidence. The list is already prioritised for today's "
        "situation (most important and situationally relevant first); use it as context to "
        "personalise your coaching and avoid repeating what the athlete already knows. "
        "There are no filter arguments — the full prioritised set is returned."
    )
    _payload: str = ""

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, payload: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_payload", payload)

    def _run(self, **kwargs: Any) -> str:
        return self._payload


class _RecoveryMemoryTool(BaseTool):
    name: str = "get_recovery_memories"
    description: str = (
        "Retrieve durable recovery and risk observations about this athlete — recent "
        "illness, injury, life stress, and known recovery patterns built up over "
        "previous sessions. Returns JSON with an 'active_memories' list, each entry "
        "containing scope, category, content, and confidence. Use these to interpret "
        "anomalies in the recovery metrics (e.g. illness explaining elevated resting "
        "HR); they are context, not metrics to average into trends or baselines."
    )
    _payload: str = ""

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, payload: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_payload", payload)

    def _run(self, **kwargs: Any) -> str:
        return self._payload


class _RestitutionDataTool(BaseTool):
    name: str = "get_restitution_data"
    description: str = (
        "Retrieve the daily recovery and training load timeline for the analysis period. "
        "Returns a JSON array with one entry per calendar day, each containing 'date', "
        "'restitution' (HRV, resting HR, sleep, readiness — null if not recorded), and "
        "'training' (TSS, IF, duration — null if no workouts that day). The most recent "
        "day's 'restitution' may also include a 'comment' field — the athlete's free-text "
        "note about today's readiness/recovery (present only for today). Each entry also "
        "includes precomputed 'rolling_tss_2d' and 'rolling_tss_3d' — the summed total_tss "
        "for that day plus the preceding 1-2 days (null if the window extends before the "
        "data range). Use these directly when citing 2-3 day cumulative load; do not sum "
        "per-day TSS values yourself. Call this first."
    )
    _payload: str = ""

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, payload: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_payload", payload)

    def _run(self, **kwargs: Any) -> str:
        return self._payload


class _MemoryDataTool(BaseTool):
    name: str = "get_memory_data"
    description: str = (
        "Retrieve the current active memories for the athlete and today's coaching output. "
        "Returns JSON with 'active_memories' and 'coaching_output' keys. Call this first."
    )
    _payload: str = ""

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, payload: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_payload", payload)

    def _run(self, **kwargs: Any) -> str:
        return self._payload


_TAPER_WINDOW_DAYS = 10
_HARD_TSS_THRESHOLD = 80


def _build_day_context(
    timeline: list[dict[str, Any]],
    planned_activities: list[PlannedActivity],
    upcoming_races: list[PlannedActivity],
    analysis_date: str,
) -> DayContext:
    """Derive today's situation from data already computed for the crew.

    Buckets are intentionally coarse so the result is stable: the same inputs always
    produce the same context, and small fluctuations do not flip the surfaced memories.
    """
    today = next((d for d in timeline if d.get("date") == analysis_date), None)
    restitution = (today or {}).get("restitution") or {}
    readiness_val = restitution.get("readiness")
    if readiness_val is None:
        readiness = "normal"
    elif readiness_val < 40:
        readiness = "low"
    elif readiness_val > 70:
        readiness = "high"
    else:
        readiness = "normal"

    planned_today = [p for p in planned_activities if p.date == analysis_date]
    if not planned_today:
        demand = "rest"
    else:
        max_tss = max((p.estimated_tss or 0) for p in planned_today)
        labels = {label.lower() for p in planned_today for label in p.labels}
        hard_labels = {"interval", "intervals", "hard", "key", "race"}
        if max_tss >= _HARD_TSS_THRESHOLD or labels & hard_labels:
            demand = "hard"
        else:
            demand = "easy"

    target_day = date.fromisoformat(analysis_date)
    phase = "normal"
    for race in upcoming_races:
        if "race" not in race.labels and "seasongoal" not in race.labels:
            continue
        days_until = (date.fromisoformat(race.date) - target_day).days
        if 0 <= days_until <= _TAPER_WINDOW_DAYS:
            phase = "taper"
            break

    return DayContext(readiness=readiness, demand=demand, phase=phase)


def _format_memories_full(memories: list[Memory]) -> list[dict[str, Any]]:
    """Full active set with ids and importance — for the memory extractor, not the coach."""
    ranked = sorted(memories, key=lambda m: (m.confidence, m.updated_at), reverse=True)
    return [
        {
            "memory_id": m.memory_id,
            "scope": m.scope,
            "category": m.category,
            "content": m.content,
            "confidence": m.confidence,
            "importance": m.importance,
        }
        for m in ranked
    ]


_KNOWN_LLM_PREFIXES = ("anthropic/", "openai/", "azure/", "bedrock/", "vertex_ai/")


def _normalize_llm(model: str) -> str:
    """Add a provider prefix when the model string lacks one.

    CrewAI's router only recognises claude-* models that appear in its
    bundled registry. Newer models (e.g. claude-sonnet-4-6) are absent
    and fall through to the OpenAI provider, causing 404s. An explicit
    prefix bypasses the registry lookup entirely.
    """
    if any(model.startswith(p) for p in _KNOWN_LLM_PREFIXES):
        return model
    if model.startswith("claude-"):
        return f"anthropic/{model}"
    if model.startswith(("gpt-", "o1-", "o3-", "o4-")):
        return f"openai/{model}"
    return model


def _make_agent(agent_def: AgentDoc, tools: list, default_llm: str) -> Agent:
    model = _normalize_llm(agent_def.llm_model or default_llm)
    llm = LLM(model=model, max_tokens=settings.llm_max_tokens)
    return Agent(
        role=agent_def.role,
        goal=agent_def.goal,
        backstory=agent_def.backstory.strip(),
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=tools,
        memory=False,
    )


_JSON_OUTPUT_INSTRUCTION = (
    "Output a single valid JSON object matching the schema exactly — no markdown code "
    "fences, no commentary before or after, all strings properly escaped."
)


def _make_task(
    task_def: dict[str, Any],
    agent: Agent,
    context: list[Task] | None = None,
    async_execution: bool = False,
    output_pydantic: type[BaseModel] | None = None,
) -> Task:
    expected_output = task_def["expected_output"].strip()
    if output_pydantic is not None:
        expected_output = f"{expected_output}\n\n{_JSON_OUTPUT_INSTRUCTION}"
    return Task(
        description=task_def["description"].strip(),
        expected_output=expected_output,
        agent=agent,
        context=context or [],
        async_execution=async_execution,
        output_pydantic=output_pydantic,
    )


def _parse_memory_extraction(raw: str) -> MemoryExtractionOutput | None:
    """Best-effort parse of the memory extractor's raw output.

    Returns None (logged) on any failure — memory extraction is non-essential and must
    never discard the completed daily analysis (issue #54).
    """
    try:
        return MemoryExtractionOutput.model_validate_json(raw)
    except ValidationError:
        start, end = raw.find("{"), raw.rfind("}")
        if 0 <= start < end:
            try:
                return MemoryExtractionOutput.model_validate_json(raw[start : end + 1])
            except ValidationError:
                pass
    logger.warning("memory_extraction_task output not parseable, raw=%r", raw[:200])
    return None


def run_daily_analysis(input: DailyAnalysisInput) -> dict[str, Any]:
    """Build and run the three-agent daily analysis crew (synchronous — use asyncio.to_thread).

    The workout performance analyst and restitution analyst run in parallel; their
    outputs are both passed as context to the daily coach.
    """
    if settings.anthropic_api_key:
        os.environ.setdefault("ANTHROPIC_API_KEY", settings.anthropic_api_key)
    if settings.openai_api_key:
        os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)

    from datetime import datetime

    weekday = datetime.fromisoformat(input.date).strftime("%A")
    athlete_name = (
        f"{input.athlete.firstname or ''} {input.athlete.lastname or ''}".strip() or "athlete"
    )

    athlete_settings = _athlete_settings_summary(input.athlete)

    # Build a lookup of athlete assessments per activity_id from the day's daily entry
    from models.daily_entry import ActivityAssessment as _ActivityAssessment

    assessment_by_id: dict[int, _ActivityAssessment] = {}
    for entry in input.daily_entries:
        if entry.date == input.date:
            for a in entry.activity_assessments:
                assessment_by_id[a.activity_id] = a

    def _enrich_workout(w: dict) -> dict:
        aid = w.get("activity_id")
        assessment = assessment_by_id.get(aid) if aid else None
        user_tags = assessment.tags if assessment else []
        session_tags = w.get("session", {}).get("tags", [])
        all_tags = list(dict.fromkeys(session_tags + user_tags))

        enriched = dict(w)
        if all_tags:
            enriched["tags"] = all_tags
        if assessment:
            enriched["athlete_rpe"] = assessment.rpe
            if assessment.notes:
                enriched["athlete_notes"] = assessment.notes
        enriched["intensity_distribution"] = _compute_intensity_distribution(w)
        return enriched

    philosophy = _philosophy_payload(input.philosophy)

    # Deterministic weekly polarization verdict (polarized philosophy only), computed
    # from the 14-day recent analyses; the LLM presents it rather than recomputing it.
    weekly_assessment = None
    if input.philosophy and input.philosophy.name == _POLARIZED_SLUG:
        weekly_assessment = _weekly_philosophy_assessment(input.recent_workout_analyses, input.date)

    # The analyst must judge polarization from the weekly verdict only, never today's
    # session in isolation, so its copy omits both today and day_over_day_note.
    weekly_assessment_for_analyst = (
        {k: v for k, v in weekly_assessment.items() if k not in ("today", "day_over_day_note")}
        if weekly_assessment
        else None
    )
    # The coach additionally writes philosophy_statement, which is specifically about
    # today's session, so its copy keeps `today` (but not the deterministic
    # day_over_day_note text, so it writes its own sentence rather than paraphrasing it).
    weekly_assessment_for_coach = (
        {k: v for k, v in weekly_assessment.items() if k != "day_over_day_note"}
        if weekly_assessment
        else None
    )

    athlete_tz = input.athlete.settings.timezone
    workout_payload_data: dict[str, Any] = {
        "athlete_settings": athlete_settings,
        "workouts": [_enrich_workout(w) for w in input.workout_analyses],
    }
    if philosophy:
        workout_payload_data["training_philosophy"] = philosophy
    if weekly_assessment_for_analyst:
        workout_payload_data["weekly_philosophy_assessment"] = weekly_assessment_for_analyst
    workout_payload = json.dumps(
        convert_datetimes_in_obj(workout_payload_data, athlete_tz), default=str
    )

    plans_payload_data: dict[str, Any] = {
        "athlete_settings": athlete_settings,
        "planned_activities": [p.model_dump(mode="json") for p in input.planned_activities],
    }
    if philosophy:
        plans_payload_data["training_philosophy"] = philosophy
    if weekly_assessment_for_coach:
        plans_payload_data["weekly_philosophy_assessment"] = weekly_assessment_for_coach
    plans_payload = json.dumps(plans_payload_data, default=str)

    def _race_entry(activity: PlannedActivity) -> dict[str, Any]:
        # days_until is relative to the calendar day being analyzed (input.date),
        # not the date the analysis happens to run on.
        race_date = date.fromisoformat(activity.date)
        calendar_day = date.fromisoformat(input.date)
        days_until = (race_date - calendar_day).days
        return {
            "date": activity.date,
            "name": activity.name,
            "sport": activity.sport,
            "days_until": days_until,
        }

    season_goal = next((a for a in input.upcoming_races if "seasongoal" in a.labels), None)
    races_payload = json.dumps(
        {
            "season_goal": _race_entry(season_goal) if season_goal else None,
            "upcoming_races": [_race_entry(a) for a in input.upcoming_races if "race" in a.labels],
        },
        default=str,
    )

    restitution_start = (
        date.fromisoformat(input.date) - timedelta(days=_RESTITUTION_WINDOW_DAYS - 1)
    ).isoformat()
    timeline = _build_timeline(
        restitution_start,
        input.date,
        input.daily_entries,
        input.recent_workout_analyses,
    )
    restitution_payload = json.dumps(timeline, default=str)

    day_context = _build_day_context(
        timeline, input.planned_activities, input.upcoming_races, input.date
    )
    coach_memory_payload = json.dumps(
        {
            "active_memories": select_relevant_memories(
                input.active_memories, day_context, input.date
            )
        },
        default=str,
    )
    extractor_memory_payload = json.dumps(
        {"active_memories": _format_memories_full(input.active_memories)},
        default=str,
    )
    restitution_memory_payload = json.dumps(
        {"active_memories": select_recovery_memories(input.active_memories, input.date)},
        default=str,
    )

    workout_tool = _WorkoutDataTool(payload=workout_payload)
    plans_tool = _PlansDataTool(payload=plans_payload)
    races_tool = _RacesDataTool(payload=races_payload)
    restitution_tool = _RestitutionDataTool(payload=restitution_payload)
    recovery_memory_tool = _RecoveryMemoryTool(payload=restitution_memory_payload)
    memory_context_tool = _MemoryContextTool(payload=coach_memory_payload)

    llm = settings.llm_model

    analyst = _make_agent(
        require_definition(input.agents, "workout_performance_analyst", "agent"),
        tools=[workout_tool],
        default_llm=llm,
    )
    restitution_analyst = _make_agent(
        require_definition(input.agents, "restitution_analyst", "agent"),
        tools=[restitution_tool, recovery_memory_tool],
        default_llm=llm,
    )
    coach = _make_agent(
        require_definition(input.agents, "daily_coach", "agent"),
        tools=[plans_tool, races_tool, memory_context_tool],
        default_llm=llm,
    )

    shared_inputs = {"date": input.date, "weekday": weekday, "athlete_name": athlete_name}
    restitution_inputs = {
        "athlete_name": athlete_name,
        "start_date": restitution_start,
        "end_date": input.date,
        "days": _RESTITUTION_WINDOW_DAYS,
    }

    workout_task_def = require_definition(input.tasks, "workout_analysis_task", "task")
    restitution_task_def = require_definition(input.tasks, "restitution_analysis_task", "task")
    coaching_task_def = require_definition(input.tasks, "daily_coaching_task", "task")
    extraction_task_def = require_definition(input.tasks, "memory_extraction_task", "task")

    analysis_task = _make_task(
        {
            "description": workout_task_def.description.format(**shared_inputs),
            "expected_output": workout_task_def.expected_output.format(**shared_inputs),
        },
        agent=analyst,
        async_execution=True,
        output_pydantic=WorkoutAnalysisOutput,
    )
    restitution_task = _make_task(
        {
            "description": restitution_task_def.description.format(**restitution_inputs),
            "expected_output": restitution_task_def.expected_output.format(**restitution_inputs),
        },
        agent=restitution_analyst,
        async_execution=True,
        output_pydantic=RestitutionAnalysisOutput,
    )
    coaching_task = _make_task(
        {
            "description": coaching_task_def.description.format(**shared_inputs),
            "expected_output": coaching_task_def.expected_output.format(**shared_inputs),
        },
        agent=coach,
        context=[analysis_task, restitution_task],
        output_pydantic=CoachingOutput,
    )

    memory_tool = _MemoryDataTool(payload=extractor_memory_payload)
    memory_extractor = _make_agent(
        require_definition(input.agents, "memory_extractor", "agent"),
        tools=[memory_tool],
        default_llm=llm,
    )
    memory_task = _make_task(
        {
            "description": extraction_task_def.description.format(**shared_inputs),
            "expected_output": extraction_task_def.expected_output.format(**shared_inputs),
        },
        agent=memory_extractor,
        context=[coaching_task],
        output_pydantic=None,  # parsed tolerantly post-kickoff so truncation can't abort the crew
    )

    crew = Crew(
        agents=[analyst, restitution_analyst, coach, memory_extractor],
        tasks=[analysis_task, restitution_task, coaching_task, memory_task],
        verbose=True,
    )

    logger.info("Starting daily analysis crew for %s on %s", athlete_name, input.date)
    with capture_prompt_log(input.athlete.athlete_id, "daily_analysis", crew) as prompt_log_run_id:
        result = crew.kickoff()
    prompt_log_entries = drain_prompt_log(prompt_log_run_id)
    run_usage = collect_run_usage(
        crew, input.athlete.athlete_id, "daily_analysis", prompt_log_run_id
    )

    workout_output: WorkoutAnalysisOutput | None = None
    restitution_output: RestitutionAnalysisOutput | None = None
    coaching_output: CoachingOutput | None = None
    memory_extraction_output: MemoryExtractionOutput | None = None

    if result.tasks_output and len(result.tasks_output) >= 4:
        t0, t1, t2, t3 = result.tasks_output[:4]
        workout_output = t0.pydantic if isinstance(t0.pydantic, WorkoutAnalysisOutput) else None
        restitution_output = (
            t1.pydantic if isinstance(t1.pydantic, RestitutionAnalysisOutput) else None
        )
        coaching_output = t2.pydantic if isinstance(t2.pydantic, CoachingOutput) else None
        memory_extraction_output = _parse_memory_extraction(t3.raw)
        if workout_output is None:
            logger.warning("workout_analysis_task pydantic output missing, raw=%r", t0.raw[:200])
        if restitution_output is None:
            logger.warning(
                "restitution_analysis_task pydantic output missing, raw=%r", t1.raw[:200]
            )
        if coaching_output is None:
            logger.warning("daily_coaching_task pydantic output missing, raw=%r", t2.raw[:200])
    else:
        logger.warning("Unexpected tasks_output length: %d", len(result.tasks_output or []))

    return {
        "workout_analysis": workout_output,
        "restitution_analysis": restitution_output,
        "coaching_feedback": coaching_output,
        "weekly_philosophy_assessment": weekly_assessment,
        "memory_extraction": memory_extraction_output,
        "prompt_log_entries": prompt_log_entries,
        "run_usage": run_usage,
    }
