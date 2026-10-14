from analysis.models import (
    HeartRateDrift,
    LapAnalysis,
    PowerHistogram,
    SessionInfo,
    WorkoutAnalysis,
    WorkoutMetrics,
    ZoneAnalysis,
    ZoneDistribution,
    ZoneInfo,
)
from crew.daily_analysis import _compact_workout_details


def _zone(name: str, lower: float, upper: float | None, seconds: float) -> ZoneInfo:
    return ZoneInfo(name=name, lower=lower, upper=upper, seconds=seconds, percent=seconds / 36)


def _analysis() -> dict:
    return WorkoutAnalysis(
        analysis_type="endurance",
        session=SessionInfo(
            name="Threshold intervals",
            sport="cycling",
            sub_sport=None,
            category="cycling",
            start_time=None,
            distance_km=30.0,
            duration_sec=3600,
            data_points=3600,
            sample_interval=1.0,
        ),
        metrics=WorkoutMetrics(),
        laps=[
            LapAnalysis(
                lap_number=1,
                start_time_sec=0,
                duration_sec=600,
                avg_power=180.04,
                avg_heart_rate=132.27,
                intensity_type="warmup",
            ),
            LapAnalysis(
                lap_number=2,
                start_time_sec=600,
                duration_sec=480,
                avg_power=281.66,
                max_power=402.0,
                avg_heart_rate=164.0,
                hr_drift_pct=3.21,
                is_erg_mode=True,
                description="free text",
            ),
        ],
        zones=ZoneAnalysis(
            power_zones=ZoneDistribution(
                total_seconds=3600,
                zones=[_zone("Z1", 0, 150, 900), _zone("Z4", 250, 300, 2700)],
            ),
            heart_rate_zones=ZoneDistribution(
                total_seconds=3600, zones=[_zone("Z2", 120, 150, 3600)]
            ),
        ),
        heart_rate_drift=HeartRateDrift(
            duration_sec=3000,
            avg_hr_p1=140.12,
            avg_hr_p2=146.48,
            avg_power_p1=220.0,
            avg_power_p2=219.0,
            hr_per_watt_p1=0.6369,
            hr_per_watt_p2=0.6688,
            drift_pct=5.01,
        ),
        power_histogram=PowerHistogram(bucket_width=5.0, seconds=[1.0] * 80),
    ).model_dump(mode="json")


def test_payload_carries_laps_zones_and_hr_drift():
    details = _compact_workout_details(_analysis())

    assert [lap["lap_number"] for lap in details["laps"]] == [1, 2]
    assert details["laps"][0] == {
        "lap_number": 1,
        "duration_sec": 600.0,
        "avg_power": 180.0,
        "avg_heart_rate": 132.3,
        "intensity_type": "warmup",
    }
    assert details["laps"][1]["hr_drift_pct"] == 3.2
    assert details["laps"][1]["is_erg_mode"] is True
    assert "description" not in details["laps"][1]

    assert [z["name"] for z in details["zones"]["power_zones"]] == ["Z1", "Z4"]
    assert details["zones"]["power_zones"][1] == {
        "name": "Z4",
        "lower": 250.0,
        "upper": 300.0,
        "seconds": 2700,
        "percent": 75.0,
    }
    assert details["zones"]["heart_rate_zones"][0]["seconds"] == 3600

    assert details["heart_rate_drift"]["drift_pct"] == 5.01
    assert details["heart_rate_drift"]["hr_per_watt_p2"] == 0.669

    assert "power_histogram" not in details


def test_payload_omits_details_the_workout_lacks():
    analysis = _analysis()
    analysis["laps"] = []
    analysis["zones"] = None
    analysis["heart_rate_drift"] = None

    assert _compact_workout_details(analysis) == {}
