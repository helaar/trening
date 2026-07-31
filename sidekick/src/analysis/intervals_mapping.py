"""Map Intervals.icu activity data onto sidekick's existing analysis shapes.

Whole-workout NP/IF/TSS/variability-index and zone-distribution are sourced
from Intervals.icu instead of computed locally (per the athlete's direction:
don't reinvent what Intervals.icu already computes). There is no local
fallback — when Intervals.icu data isn't available yet for an activity, these
fields are simply None (see intervals_sync_status on WorkoutAnalysis for why).

Lap analysis, ERG detection, the power histogram, and whole-workout HR drift
are untouched by this module — they either depend on raw stream data Intervals
doesn't expose in an equivalent shape (per-lap power stdev), or aren't offered
by Intervals.icu at all in the same shape (the two-half aerobic-decoupling
split), so they stay fully local.

Field names confirmed against a real Intervals.icu API response
(scripts/spike_intervals_icu.py, run 2026-07-30). Zone boundaries adopt
Intervals.icu's own zone configuration (icu_power_zones/icu_hr_zones, set up
in the Intervals.icu UI) rather than sidekick's AthleteSettings zone
definitions — those remain in effect only for the local lap zone-name lookup.
"""

from typing import Any, NamedTuple

from analysis.models import StatsSummary, ZoneAnalysis, ZoneDistribution, ZoneInfo
from models.intervals_activity import IntervalsActivityRaw

# icu_zone_times includes a "SS" (sweet spot) bucket that overlaps other zones —
# excluded here so it doesn't corrupt the positional Zone-3-at-index-2 convention
# that weekly_intensity.py and _compute_intensity_distribution rely on.
_POWER_ZONE_IDS = ("Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "Z7")


class PowerMetricsResult(NamedTuple):
    power_stats: StatsSummary | None
    normalized_power: float | None
    variability_index: float | None
    intensity_factor: float | None
    training_stress_score: float | None


def map_power_metrics(intervals_activity: IntervalsActivityRaw | None) -> PowerMetricsResult:
    """Whole-workout power metrics from Intervals.icu, or all-None if unavailable."""
    if intervals_activity is None:
        return PowerMetricsResult(None, None, None, None, None)

    raw = intervals_activity.raw_data
    avg_watts = raw.get("icu_average_watts")
    np_value = raw.get("icu_weighted_avg_watts")
    vi_value = raw.get("icu_variability_index")
    tss_value = raw.get("icu_training_load")
    ftp = raw.get("icu_ftp")

    power_stats = StatsSummary(mean=avg_watts) if avg_watts is not None else None
    if_value = (np_value / ftp) if np_value and ftp else None

    return PowerMetricsResult(
        power_stats=power_stats,
        normalized_power=np_value,
        variability_index=vi_value,
        intensity_factor=if_value,
        training_stress_score=tss_value,
    )


def map_rpe(intervals_activity: IntervalsActivityRaw | None) -> int | None:
    """Athlete-logged RPE (1-10) for this activity, sourced from Intervals.icu.

    Field name (icu_rpe) is a best documented guess, pending confirmation via
    scripts/spike_intervals_icu.py's RPE diagnostics (candidates also captured
    there: feel, session_rpe, perceived_exertion).
    """
    if intervals_activity is None:
        return None
    value = intervals_activity.raw_data.get("icu_rpe")
    if isinstance(value, int) and 1 <= value <= 10:
        return value
    return None


def map_athlete_ftp(intervals_activity: IntervalsActivityRaw | None) -> float | None:
    """FTP Intervals.icu used for this activity's own IF/TSS calculation.

    Sourced from Intervals.icu rather than sidekick's locally-configured FTP so
    the displayed FTP is always internally consistent with the IF/TSS shown
    alongside it.
    """
    if intervals_activity is None:
        return None
    return intervals_activity.raw_data.get("icu_ftp")


def _build_zone_distribution(
    zone_times: list[dict[str, Any]] | list[float] | None,
    breakpoints: list[float] | None,
    ftp_or_absolute_scale: float | None,
    zone_ids: tuple[str, ...] | None,
) -> ZoneDistribution | None:
    """Shared builder for power/HR zone distributions.

    zone_times: for power, a list of {"id": "Z1".., "secs": N} dicts (icu_zone_times);
        for HR, a plain positional list of seconds per zone (icu_hr_zone_times).
    breakpoints: cumulative upper bounds per zone (power: % of FTP; HR: absolute bpm).
    ftp_or_absolute_scale: multiply breakpoints by this/100 for power (percent->watts);
        pass None for HR, where breakpoints are already absolute.
    zone_ids: when set, filter zone_times to only these ids, in this order (power).
        When None, zone_times is assumed already positional (HR).
    """
    if not zone_times or not breakpoints:
        return None

    if zone_ids is not None:
        by_id = {z.get("id"): z.get("secs", 0.0) for z in zone_times}
        seconds_per_zone = [float(by_id.get(zid, 0.0)) for zid in zone_ids]
    else:
        seconds_per_zone = [float(s) for s in zone_times]

    n = min(len(seconds_per_zone), len(breakpoints))
    if n == 0:
        return None

    total_seconds = sum(seconds_per_zone[:n])
    zones: list[ZoneInfo] = []
    lower = 0.0
    for i in range(n):
        is_last = i == n - 1
        raw_upper = breakpoints[i]
        upper = None if is_last else (
            raw_upper * ftp_or_absolute_scale / 100.0 if ftp_or_absolute_scale else raw_upper
        )
        scaled_lower = lower * ftp_or_absolute_scale / 100.0 if ftp_or_absolute_scale else lower
        seconds = seconds_per_zone[i]
        zones.append(
            ZoneInfo(
                name=f"Z{i + 1}",
                lower=scaled_lower,
                upper=upper,
                seconds=seconds,
                percent=(seconds / total_seconds * 100.0) if total_seconds else 0.0,
            )
        )
        lower = raw_upper

    return ZoneDistribution(total_seconds=total_seconds, sample_interval=1.0, zones=zones)


def map_zone_analysis(intervals_activity: IntervalsActivityRaw | None) -> ZoneAnalysis | None:
    """Power/HR time-in-zone from Intervals.icu's own zone configuration."""
    if intervals_activity is None:
        return None

    raw = intervals_activity.raw_data

    power_dist = _build_zone_distribution(
        zone_times=raw.get("icu_zone_times"),
        breakpoints=raw.get("icu_power_zones"),
        ftp_or_absolute_scale=raw.get("icu_ftp"),
        zone_ids=_POWER_ZONE_IDS,
    )
    hr_dist = _build_zone_distribution(
        zone_times=raw.get("icu_hr_zone_times"),
        breakpoints=raw.get("icu_hr_zones"),
        ftp_or_absolute_scale=None,
        zone_ids=None,
    )

    if power_dist is None and hr_dist is None:
        return None
    return ZoneAnalysis(power_zones=power_dist, heart_rate_zones=hr_dist)
