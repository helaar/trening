"""Deterministic weekly polarized-training assessment (philosophy-neutral aggregation).

Extracted from crew/daily_analysis.py so it can be computed directly from synced
workout analyses, independent of the daily LLM coaching crew.
"""

from datetime import date, timedelta
from typing import Any

from config import settings

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


def _session_start_date(analysis: dict[str, Any]) -> str | None:
    """YYYY-MM-DD of a session's start_time (datetime or string), or None."""
    start_time = analysis.get("session", {}).get("start_time")
    if not start_time:
        return None
    return start_time.date().isoformat() if hasattr(start_time, "date") else str(start_time)[:10]


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


def _today_label(today: dict[str, Any]) -> str:
    """One-word dominant-zone label for a single day, no window comparison at all."""
    if not today["trained"]:
        return "Rest day"
    if today["classified_minutes"] <= 0:
        return "Unclassified"
    pct = {"Easy": today["low_pct"], "Moderate": today["moderate_pct"], "Hard": today["high_pct"]}
    dominant = max(pct, key=lambda k: pct[k] or 0)
    return f"{dominant} day"


def compute_weekly_philosophy_assessment(
    recent_analyses: list[dict[str, Any]], end_date: str
) -> dict[str, Any]:
    """Deterministic polarized verdict over the trailing 7 days ending on end_date.

    Aggregates per-session bands (with graduated gray-zone tolerance) into a weekly
    distribution and a status + plain-language description the coach can paraphrase. All
    thresholds are config tunables so the verdict can be adjusted without code changes.

    Also reports today's own contribution (its own low/moderate/high split and a
    dominant-zone label), with no comparison to the rest of the window.
    """
    end = date.fromisoformat(end_date)
    start = end - timedelta(days=6)
    start_iso, end_iso = start.isoformat(), end.isoformat()
    depth_frac = settings.polarized_gray_zone_depth_frac

    today_bands = _day_bands(recent_analyses, end_iso, depth_frac)
    today_bands["label"] = _today_label(today_bands)

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
