#!/usr/bin/env python3
"""
Kalkulerer NP, IF, TSS, VI, sonefordeling, pulsdrift og lap-detaljer (inkl. autolap)
fra en FIT-fil. All output logges både til terminal og til <fitfil>-analyse.txt.
"""
import argparse
import re
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from fitparse import FitFile


# ---------------------------------------------------------------------------
# Dataklasser og hjelpefunksjoner
# ---------------------------------------------------------------------------
INTENSITY_NAMES = {
    0: "active",
    1: "rest",
    2: "warmup",
    3: "cooldown",
    4: "recovery",
    5: "interval",
}

@dataclass
class Zone:
    name: str
    low: Optional[float]
    high: Optional[float]


def _safe_float(value: Optional[float]) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def read_timeseries_and_laps(fit_path: str) -> Tuple[pd.DataFrame, List[Dict[str, pd.Timestamp]]]:
    """
    Leser kraft/puls/kadens (resamplet til 1 Hz) og lap-informasjon fra FIT-filen.
    Returnerer en DataFrame og en liste med lap-intervaller {"start": ts, "end": ts}.
    """
    fitfile = FitFile(fit_path)

    record_rows: List[Dict[str, object]] = []
    laps: List[Dict[str, pd.Timestamp]] = []

    for message in fitfile.get_messages():
        if message.name == "record":
            data = {d.name: d.value for d in message}
            timestamp = data.get("timestamp")
            if timestamp is None:
                continue
            record_rows.append(
                {
                    "timestamp": pd.to_datetime(timestamp),
                    "power": _safe_float(data.get("power")),
                    "heart_rate": _safe_float(data.get("heart_rate")),
                    "cadence": _safe_float(data.get("cadence")),
                }
            )
        elif message.name == "lap":
            data = {d.name: d.value for d in message}
            start = data.get("start_time")
            duration = data.get("total_elapsed_time")
            end = data.get("timestamp") or data.get("end_time")
            if not start:
                continue
            start_ts = pd.to_datetime(start)
            if duration is not None:
                end_ts = start_ts + pd.to_timedelta(duration, unit="s")
            elif end is not None:
                end_ts = pd.to_datetime(end)
            else:
                continue
            lap_name = data.get("wkt_step_name") or data.get("lap_name") or data.get("swim_name")
            intensity = data.get("intensity")

            laps.append(
                {
                    "start": start_ts,
                    "end": end_ts,
                    "label": lap_name,
                    "intensity": intensity,
                }
            )


    if not record_rows:
        raise ValueError("Fant ingen record-data i FIT-filen.")

    df = pd.DataFrame(record_rows).set_index("timestamp").sort_index()

    # Resample til 1 Hz ved behov
    deltas = df.index.to_series().diff().dropna().dt.total_seconds()
    if not deltas.empty and not np.allclose(deltas, 1.0, atol=0.5):
        df = (
            df.resample("1S")
            .mean()
            .interpolate(method="time", limit_direction="both")
        )

    laps.sort(key=lambda lap: lap["start"])
    return df, laps


def normalized_power(power: pd.Series, window: int = 30) -> float:
    valid_power = power.dropna()
    if valid_power.empty:
        raise ValueError("Ingen power-data å beregne NP fra.")

    rolling_mean = valid_power.rolling(window=window, min_periods=window).mean().dropna()
    if rolling_mean.empty:
        raise ValueError("For kort segment til å beregne Normalized Power.")

    np_value = (rolling_mean.pow(4).mean()) ** 0.25
    return float(np_value)


def intensity_factor(np_value: float, ftp: float) -> float:
    if ftp <= 0:
        raise ValueError("FTP må være > 0.")
    return np_value / ftp


def training_stress_score(duration_sec: float, np_value: float, if_value: float, ftp: float) -> float:
    return (duration_sec * np_value * if_value) / (ftp * 3600) * 100


def series_stats(series: pd.Series) -> Dict[str, Optional[float]]:
    valid = series.dropna()
    if valid.empty:
        return {"min": None, "max": None, "mean": None, "std": None}

    return {
        "min": float(valid.min()),
        "max": float(valid.max()),
        "mean": float(valid.mean()),
        "std": float(valid.std(ddof=0)),
    }


def load_settings(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data


def parse_zone_definitions(raw: Optional[List[Dict[str, str]]]) -> List[Zone]:
    if not raw:
        return []

    zones: List[Zone] = []
    for entry in raw:
        if not isinstance(entry, dict) or len(entry) != 1:
            raise ValueError(f"Ugyldig sone-definisjon: {entry!r}")
        name, range_spec = next(iter(entry.items()))
        low, high = parse_range(range_spec)
        zones.append(Zone(name=name, low=low, high=high))
    return zones


def parse_range(range_spec: str) -> Tuple[Optional[float], Optional[float]]:
    parts = str(range_spec).split("-", 1)
    if len(parts) != 2:
        raise ValueError(f"Kunne ikke tolke intervall: {range_spec!r}")

    low_str, high_str = parts[0].strip(), parts[1].strip()
    low = float(low_str) if low_str else None
    high = float(high_str) if high_str else None
    return low, high


def parse_iso8601_duration(value: str) -> float:
    """
    Tolker en ISO 8601 varighet (f.eks. PT10M) til sekunder.
    """
    pattern = (
        r"^P(?:(?P<days>\d+)D)?"
        r"(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+(?:\.\d+)?)S)?)?$"
    )
    match = re.fullmatch(pattern, value)
    if not match:
        raise ValueError(f"Ugyldig ISO 8601 varighet: {value!r}")

    days = float(match.group("days") or 0)
    hours = float(match.group("hours") or 0)
    minutes = float(match.group("minutes") or 0)
    seconds = float(match.group("seconds") or 0)
    return ((days * 24 + hours) * 60 + minutes) * 60 + seconds


def infer_sample_interval(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 0.0
    deltas = index.to_series().diff().dropna().dt.total_seconds()
    if deltas.empty:
        return 0.0
    return float(deltas.median())


def parse_hms(value: str) -> float:
    """
    Tolker HH:MM:SS, MM:SS eller SS til sekunder.
    """
    parts = value.strip().split(":")
    if not parts or len(parts) > 3:
        raise ValueError(f"Ugyldig tidsformat: {value!r}")

    parts = [int(p) for p in parts]
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, m, s = 0, parts[0], parts[1]
    else:
        h, m, s = 0, 0, parts[0]
    return float(h * 3600 + m * 60 + s)


def compute_zone_durations(
    series: pd.Series,
    zones: List[Zone],
    sample_interval: Optional[float] = None,
) -> Dict[str, object]:
    if not zones:
        return {"total_seconds": 0.0, "sample_interval": 0.0, "zones": []}

    if sample_interval is None or sample_interval <= 0:
        sample_interval = infer_sample_interval(series.index)
        if sample_interval <= 0:
            sample_interval = 1.0  # Fallback

    valid_mask = series.notna()
    total_samples = int(valid_mask.sum())
    total_seconds = total_samples * sample_interval

    results = []
    for i, zone in enumerate(zones):
        low = zone.low if zone.low is not None else float("-inf")
        high = zone.high if zone.high is not None else float("inf")

        if zone.high is None or i == len(zones) - 1:
            mask = (series >= low) & (series <= high)
        else:
            mask = (series >= low) & (series < high)

        mask = mask & valid_mask
        zone_seconds = float(mask.sum() * sample_interval)
        percent = (zone_seconds / total_seconds * 100.0) if total_seconds else 0.0

        results.append(
            {
                "name": zone.name,
                "lower": zone.low,
                "upper": zone.high,
                "seconds": zone_seconds,
                "percent": percent,
            }
        )

    return {
        "total_seconds": total_seconds,
        "sample_interval": sample_interval,
        "zones": results,
    }


def compute_heart_rate_drift(
    df: pd.DataFrame,
    start_offset: Optional[float] = None,
    duration: Optional[float] = None,
) -> Optional[Dict[str, object]]:
    if "heart_rate" not in df or "power" not in df:
        return None

    start_ts = df.index[0]
    end_ts = df.index[-1]

    if start_offset:
        start_ts = df.index[0] + pd.to_timedelta(start_offset, unit="s")
    if duration:
        end_ts = start_ts + pd.to_timedelta(duration, unit="s")

    start_ts = max(start_ts, df.index[0])
    end_ts = min(end_ts, df.index[-1])

    if end_ts <= start_ts:
        return None

    segment = df.loc[start_ts:end_ts, ["heart_rate", "power"]].dropna()
    if len(segment) < 4:
        return None

    midpoint = len(segment) // 2
    if midpoint == 0 or midpoint == len(segment):
        return None

    p1 = segment.iloc[:midpoint]
    p2 = segment.iloc[midpoint:]

    avg_power_p1 = p1["power"].mean()
    avg_power_p2 = p2["power"].mean()

    if avg_power_p1 <= 0 or avg_power_p2 <= 0:
        return None

    avg_hr_p1 = p1["heart_rate"].mean()
    avg_hr_p2 = p2["heart_rate"].mean()

    hr_per_watt_p1 = avg_hr_p1 / avg_power_p1
    hr_per_watt_p2 = avg_hr_p2 / avg_power_p2

    drift_pct = ((hr_per_watt_p2 - hr_per_watt_p1) / hr_per_watt_p1) * 100.0

    return {
        "start_ts": segment.index[0],
        "end_ts": segment.index[-1],
        "duration": (segment.index[-1] - segment.index[0]).total_seconds(),
        "samples": len(segment),
        "avg_hr_p1": float(avg_hr_p1),
        "avg_hr_p2": float(avg_hr_p2),
        "avg_power_p1": float(avg_power_p1),
        "avg_power_p2": float(avg_power_p2),
        "hr_per_watt_p1": float(hr_per_watt_p1),
        "hr_per_watt_p2": float(hr_per_watt_p2),
        "drift_pct": float(drift_pct),
    }


def compute_segment_stats(segment: pd.DataFrame, ftp: float, window: int = 30) -> Dict[str, Optional[float]]:
    stats: Dict[str, Optional[float]] = {}
    if len(segment) < 2:
        stats["duration_sec"] = 0.0
        stats["avg_power"] = None
        stats["np"] = None
        stats["vi"] = None
        stats["if"] = None
        stats["avg_hr"] = None
        stats["avg_cad"] = None
        stats["max_power"] = None
        stats["max_hr"] = None
        stats["drift_pct"] = None
        return stats

    duration_sec = (segment.index[-1] - segment.index[0]).total_seconds()
    stats["duration_sec"] = duration_sec

    avg_power = segment["power"].dropna().mean()
    stats["avg_power"] = float(avg_power) if pd.notna(avg_power) else None

    try:
        if segment["power"].dropna().empty:
            np_value = None
        else:
            np_value = normalized_power(segment["power"], window=window)
    except Exception:
        np_value = None
    stats["np"] = float(np_value) if np_value is not None else None

    stats["vi"] = (stats["np"] / stats["avg_power"]) if stats["np"] and stats["avg_power"] else None
    stats["if"] = (stats["np"] / ftp) if stats["np"] and ftp > 0 else None

    avg_hr = segment["heart_rate"].dropna().mean()
    stats["avg_hr"] = float(avg_hr) if pd.notna(avg_hr) else None

    avg_cad = segment["cadence"].dropna().mean()
    stats["avg_cad"] = float(avg_cad) if pd.notna(avg_cad) else None

    stats["max_power"] = float(segment["power"].dropna().max()) if not segment["power"].dropna().empty else None
    stats["max_hr"] = float(segment["heart_rate"].dropna().max()) if not segment["heart_rate"].dropna().empty else None

    drift = compute_heart_rate_drift(segment)
    stats["drift_pct"] = drift["drift_pct"] if drift else None

    return stats


def split_into_autolaps(df: pd.DataFrame, autolap_seconds: float) -> List[Dict[str, pd.Timestamp]]:
    """
    Deler økten inn i like segmenter (autolapper) på autolap_seconds.
    """
    if len(df) < 2 or autolap_seconds <= 0:
        return []

    autolaps: List[Dict[str, pd.Timestamp]] = []
    start_ts = df.index[0]
    end_ts = df.index[-1]
    current_start = start_ts

    while current_start < end_ts:
        current_end = current_start + pd.to_timedelta(autolap_seconds, unit="s")
        if current_end > end_ts:
            current_end = end_ts
        autolaps.append({"start": current_start, "end": current_end})
        current_start = current_end

    return autolaps


def format_range(low: Optional[float], high: Optional[float]) -> str:
    def fmt(value: Optional[float]) -> str:
        if value is None:
            return ""
        return f"{int(value)}" if float(value).is_integer() else f"{value:g}"

    if low is None and high is None:
        return "—"
    if low is None:
        return f"≤{fmt(high)}"
    if high is None:
        return f"{fmt(low)}+"
    return f"{fmt(low)}–{fmt(high)}"


def seconds_to_hms(seconds: float) -> str:
    return str(timedelta(seconds=int(round(seconds))))


def _print_stats(log, stats: Dict[str, Optional[float]]) -> None:
    def fmt(value: Optional[float]) -> str:
        return f"{value:.1f}" if value is not None else "—"

    log(f"- Min : {fmt(stats['min'])}")
    log(f"- Maks: {fmt(stats['max'])}")
    log(f"- Snitt: {fmt(stats['mean'])}")
    log(f"- Std : {fmt(stats['std'])}")


def _print_zone_summary(log, summary: Dict[str, object], label: str) -> None:
    zones = summary["zones"]
    total = summary["total_seconds"]

    if not zones or total == 0:
        log(f"Ingen {label}-data til soneberegning.")
        return

    log(f"Totalt gyldig {label}-tid: {seconds_to_hms(total)}")
    for zone in zones:
        range_str = format_range(zone["lower"], zone["upper"])
        log(
            f"- {zone['name']:<20} {range_str:<10} "
            f"{seconds_to_hms(zone['seconds']):>8} ({zone['percent']:5.1f}%)"
        )


def _print_lap_table(log, lap_rows: List[Dict[str, object]]) -> None:
    if not lap_rows:
        log("Ingen lap-data.")
        return

    headers = [
        ("Lap", "lap"),
        ("Start", "start_str"),
        ("Varighet", "duration_str"),
        ("NP", "np"),
        ("Snitt W", "avg_power"),
        ("Snitt HR", "avg_hr"),
        ("Snitt kad", "avg_cad"),
        ("Pulsdrift %", "drift_pct"),
        ("Maks W", "max_power"),
        ("Maks HR", "max_hr"),
        ("Beskrivelse", "description")
    ]

    def fmt_float(value: Optional[float], decimals: int = 1) -> str:
        return f"{value:.{decimals}f}" if value is not None else "—"

    col_widths = []
    for header, key in headers:
        max_len = len(header)
        for row in lap_rows:
            if key.endswith("_str"):
                val = str(row.get(key, ""))
            else:
                val = fmt_float(row.get(key)) if isinstance(row.get(key), (float, int)) else str(row.get(key, ""))
            max_len = max(max_len, len(val))
        col_widths.append(max_len)

    header_line = "  ".join(header.ljust(width) for (header, _), width in zip(headers, col_widths))
    log(header_line)
    log("-" * len(header_line))

    for row in lap_rows:
        values = []
        for (header, key), width in zip(headers, col_widths):
            if key.endswith("_str"):
                val = str(row.get(key, ""))
            else:
                if key in ("np", "avg_power", "avg_hr", "avg_cad", "max_power", "max_hr"):
                    val = fmt_float(row.get(key))
                elif key == "drift_pct":
                    val = fmt_float(row.get(key), decimals=2)
                else:
                    val = str(row.get(key, ""))
            values.append(val.ljust(width))
        log("  ".join(values))


# ---------------------------------------------------------------------------
# Hovedprogram
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Beregn NP, IF, TSS, VI, sonefordeling, pulsdrift og lap-detaljer fra en FIT-fil."
    )
    parser.add_argument("--fitfile", required=True, help="Path til FIT-fil.")
    parser.add_argument("--settings", help="Path til settings.yaml.")
    parser.add_argument("--ftp", type=float, help="Overstyr FTP (watt).")
    parser.add_argument("--window", type=int, default=30, help="Vinduslengde (s) for NP. Default 30.")
    parser.add_argument("--drift-start", help="Startpunkt for pulsdrift (HH:MM:SS, MM:SS eller SS).")
    parser.add_argument("--drift-duration", help="Varighet for pulsdrift (HH:MM:SS, MM:SS eller SS).")
    args = parser.parse_args()

    fit_path = Path(args.fitfile).expanduser().resolve()
    log_lines: List[str] = []

    def log(msg: str = "") -> None:
        log_lines.append(msg)
        print(msg)

    try:
        settings: Dict[str, object] = {}
        if args.settings:
            settings = load_settings(args.settings)

        ftp = args.ftp if args.ftp is not None else settings.get("ftp")
        if ftp is None:
            raise ValueError("FTP må oppgis via --ftp eller i settings.yaml.")

        max_hr = settings.get("max-hr")
        power_zones = parse_zone_definitions(settings.get("power-zones"))
        hr_zones = parse_zone_definitions(settings.get("hr-zones"))
        autolap = settings.get("autolap")  # f.eks. "PT10M"

        drift_start = parse_hms(args.drift_start) if args.drift_start else None
        drift_duration = parse_hms(args.drift_duration) if args.drift_duration else None

        df, laps = read_timeseries_and_laps(str(fit_path))
        sample_interval = infer_sample_interval(df.index)
        if sample_interval <= 0:
            sample_interval = 1.0

        duration_sec = sample_interval * len(df)

        power_stats = series_stats(df["power"])
        hr_stats = series_stats(df["heart_rate"])
        cad_stats = series_stats(df["cadence"])

        np_value = normalized_power(df["power"], window=args.window)
        if_value = intensity_factor(np_value, ftp)
        tss_value = training_stress_score(duration_sec, np_value, if_value, ftp)

        avg_power = df["power"].dropna().mean()
        vi_value = (np_value / avg_power) if avg_power and avg_power > 0 else None

        log("=== Øktinformasjon ===")
        log(f"Varighet: {seconds_to_hms(duration_sec)}")
        log(f"Datapunkter: {len(df)}")
        log(f"Estimert samplingsintervall: {sample_interval:.2f} s")
        log(f"FTP: {ftp:.0f} W")
        if max_hr:
            log(f"Maks puls (fra settings): {max_hr} bpm")

        log("\n=== Power (W) ===")
        _print_stats(log, power_stats)
        log(f"Normalized Power (NP): {np_value:.1f} W")
        if avg_power and avg_power > 0:
            log(f"Gjennomsnittlig watt: {avg_power:.1f} W")
            log(f"Variabilitetsindeks (VI): {vi_value:.3f}")
        else:
            log("Gjennomsnittlig watt: —")
            log("Variabilitetsindeks (VI): —")
        log(f"Intensity Factor (IF): {if_value:.3f}")
        log(f"Training Stress Score (TSS): {tss_value:.1f}")

        log("\n=== Puls (bpm) ===")
        _print_stats(log, hr_stats)

        log("\n=== Kadens (rpm) ===")
        _print_stats(log, cad_stats)

        if power_zones:
            log("\n=== Tid i power-zoner ===")
            power_summary = compute_zone_durations(df["power"], power_zones, sample_interval)
            _print_zone_summary(log, power_summary, "power")
        if hr_zones:
            log("\n=== Tid i puls-zoner ===")
            hr_summary = compute_zone_durations(df["heart_rate"], hr_zones, sample_interval)
            _print_zone_summary(log, hr_summary, "puls")

        drift_result = compute_heart_rate_drift(df, drift_start, drift_duration)
        log("\n=== Pulsdrift ===")
        if drift_result:
            rel_start = (drift_result["start_ts"] - df.index[0]).total_seconds()
            rel_end = (drift_result["end_ts"] - df.index[0]).total_seconds()
            log(
                f"Segment: {seconds_to_hms(rel_start)} → {seconds_to_hms(rel_end)} "
                f"({seconds_to_hms(drift_result['duration'])})"
            )
            log(f"- Snitt HR (P1): {drift_result['avg_hr_p1']:.1f} bpm")
            log(f"- Snitt HR (P2): {drift_result['avg_hr_p2']:.1f} bpm")
            log(f"- Snitt watt (P1): {drift_result['avg_power_p1']:.1f} W")
            log(f"- Snitt watt (P2): {drift_result['avg_power_p2']:.1f} W")
            log(f"- HR/W (P1): {drift_result['hr_per_watt_p1']:.4f}")
            log(f"- HR/W (P2): {drift_result['hr_per_watt_p2']:.4f}")
            log(f"- Pulsdrift: {drift_result['drift_pct']:.2f} %")
        else:
            log("Ingen gyldig data til å beregne pulsdrift for valgt segment.")

        # ------------------------------------------------------------------
        # Lap-detaljer
        # ------------------------------------------------------------------
        log("\n=== Lap-detaljer ===")

        autolap_seconds = None
        if autolap:
            autolap_seconds = parse_iso8601_duration(str(autolap))

        # Hvis ingen (eller bare én) lap i datafilen og autolap er definert:
        if autolap_seconds and (len(laps) <= 1):
            autolaps = split_into_autolaps(df, autolap_seconds)
            if autolaps:
                log(
                    f"Autolap aktivert ({seconds_to_hms(autolap_seconds)}). "
                    f"Genererer {len(autolaps)} autolapper."
                )
                laps = autolaps

        if not laps:
            log("Ingen lap (heller ingen autolap) funnet i økten.")
        else:
            lap_rows = []
            for idx, lap in enumerate(laps, start=1):
                start_ts = lap["start"]
                end_ts = lap["end"]
                lap_segment = df.loc[start_ts:end_ts].dropna(how="all")
                if lap_segment.empty:
                    continue

                intensity = lap.get("intensity")
                intensity_str = INTENSITY_NAMES.get(intensity, str(intensity) if intensity is not None else "")
                label = lap.get("label")
                description = " / ".join([part for part in (label, intensity_str) if part])
                stats = compute_segment_stats(lap_segment, ftp=ftp, window=args.window)
                lap_rows.append(
                    {
                        "lap": idx,
                        "start_str": seconds_to_hms((start_ts - df.index[0]).total_seconds()),
                        "duration_str": seconds_to_hms(stats["duration_sec"]),
                        "np": stats["np"],
                        "avg_power": stats["avg_power"],
                        "avg_hr": stats["avg_hr"],
                        "avg_cad": stats["avg_cad"],
                        "drift_pct": stats["drift_pct"],
                        "max_power": stats["max_power"],
                        "max_hr": stats["max_hr"],
                        "description": description or "-"
                    }
                )

            if lap_rows:
                _print_lap_table(log, lap_rows)
            else:
                log("Fant laps, men ingen gyldige data i disse segmentene.")

        # ------------------------------------------------------------------
        # Skriv til fil
        # ------------------------------------------------------------------
        analysis_text = "\n".join(log_lines)
        analysis_path = fit_path.with_name(f"{fit_path.stem}-analyse.txt")
        analysis_path.write_text(analysis_text, encoding="utf-8")

        print(f"\nAnalyse lagret til: {analysis_path}")

    except Exception as exc:
        error_msg = f"\n[FEIL] {exc}"
        log(error_msg)

        if log_lines:
            analysis_text = "\n".join(log_lines)
            analysis_path = fit_path.with_name(f"{fit_path.stem}-analyse.txt")
            analysis_path.write_text(analysis_text, encoding="utf-8")
            print(f"Foreløpig logg lagret i: {analysis_path}")

        raise


if __name__ == "__main__":
    main()