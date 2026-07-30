"""Phase 0 spike: inspect the real Intervals.icu API response shapes.

Throwaway exploration tool, not shipped as product code — safe to delete once
its findings have been used to confirm/correct the field-name assumptions in
clients/intervals_icu/client.py and services/intervals_matching.py. Standalone
(no sidekick imports) so it can run without PYTHONPATH or a database.

First pass (2026-07-30) confirmed auth, all endpoint paths, and most field
names — see clients/intervals_icu/client.py and services/intervals_matching.py
for what's already settled. Three things are still open, and this pass adds
targeted diagnostics for each instead of just dumping the first item:

  1. Whether `strava_id` is populated for activities that reached Intervals.icu
     via Garmin/Zwift *without* passing through Strava — section 2 below now
     lists id/device_name/source/strava_id/external_id for every fetched
     activity, not just the first, so this is visible across your real mix of
     upload paths.
  2. The race-priority category value(s) on calendar events (expected
     RACE_A/RACE_B/RACE_C alongside plain WORKOUT) — section 5 lists
     id/category/name for every event in the window. Widen --event-days-back
     if no race falls inside it.
  3. Which of the four RPE-shaped fields (icu_rpe, feel, session_rpe,
     perceived_exertion) is the one Intervals.icu actually populates when you
     log perceived exertion — section 2's table surfaces all four per
     activity, so log an RPE value on a recent activity in Intervals.icu
     before running this if none of your existing activities have one set.

Run:
  cd sidekick
  uv run python scripts/spike_intervals_icu.py --api-key YOUR_KEY --athlete-id YOUR_ID \\
      --days 90 --event-days-back 365 --event-days-forward 180
"""

import argparse
import json
from datetime import date, timedelta

import httpx

BASE_URL = "https://intervals.icu/api/v1"

_ACTIVITY_DIAGNOSTIC_FIELDS = (
    "id",
    "name",
    "type",
    "start_date",
    "device_name",
    "source",
    "strava_id",
    "external_id",
    "icu_rpe",
    "feel",
    "session_rpe",
    "perceived_exertion",
)

_EVENT_DIAGNOSTIC_FIELDS = ("id", "category", "type", "name", "start_date_local")


def _print_section(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def _print_json(label: str, data) -> None:
    print(f"--- {label} ---")
    print(json.dumps(data, indent=2, default=str)[:4000])


def _print_table(items: list[dict], fields: tuple[str, ...]) -> None:
    header = " | ".join(fields)
    print(header)
    print("-" * len(header))
    for item in items:
        print(" | ".join(str(item.get(f)) for f in fields))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-key", required=True, help="Intervals.icu API key (Settings -> Developer)")
    parser.add_argument("--athlete-id", required=True, help="Intervals.icu athlete id")
    parser.add_argument("--days", type=int, default=30, help="Activity/wellness lookback window in days")
    parser.add_argument("--event-days-back", type=int, default=180, help="Events lookback window in days")
    parser.add_argument("--event-days-forward", type=int, default=90, help="Events lookahead window in days")
    args = parser.parse_args()

    newest = date.today()
    oldest = newest - timedelta(days=args.days)
    athlete_path = f"/athlete/{args.athlete_id}"

    with httpx.Client(base_url=BASE_URL, auth=("API_KEY", args.api_key), timeout=15) as client:
        _print_section("1. Auth + activities list")
        response = client.get(
            f"{athlete_path}/activities",
            params={"oldest": oldest.isoformat(), "newest": newest.isoformat()},
        )
        print(f"status={response.status_code}")
        print("rate-limit headers:", {k: v for k, v in response.headers.items() if "ratelimit" in k.lower() or "retry" in k.lower()})
        response.raise_for_status()
        activities = response.json()
        print(f"Fetched {len(activities)} activities")

        if activities:
            _print_section("2. Strava-id / RPE fields across all fetched activities")
            _print_table(activities, _ACTIVITY_DIAGNOSTIC_FIELDS)
            print(
                "\nCheck: is strava_id populated for entries whose device_name/source "
                "looks like Garmin/Zwift rather than Strava? Which of icu_rpe/feel/"
                "session_rpe/perceived_exertion is non-null on any activity where you "
                "logged an RPE in Intervals.icu?"
            )

            first = activities[0]
            activity_id = first.get("id")
            if activity_id:
                _print_section("3. Single activity detail (NP/IF/TSS/decoupling/eFTP field names)")
                detail = client.get(f"/activity/{activity_id}")
                print(f"status={detail.status_code}")
                if detail.status_code == 200:
                    _print_json("activity detail", detail.json())

                _print_section("3b. Per-interval/lap breakdown")
                intervals_resp = client.get(f"/activity/{activity_id}/intervals")
                print(f"status={intervals_resp.status_code}")
                if intervals_resp.status_code == 200:
                    _print_json("intervals", intervals_resp.json())

        _print_section("4. Wellness")
        wellness = client.get(
            f"{athlete_path}/wellness.json",
            params={"oldest": oldest.isoformat(), "newest": newest.isoformat()},
        )
        print(f"status={wellness.status_code}")
        if wellness.status_code == 200:
            data = wellness.json()
            print(f"Fetched {len(data)} wellness days")
            if data:
                _print_json("most recent wellness day", data[-1])

        _print_section("5. Calendar / events — category values across the full window")
        event_oldest = newest - timedelta(days=args.event_days_back)
        event_newest = newest + timedelta(days=args.event_days_forward)
        events = client.get(
            f"{athlete_path}/events",
            params={"oldest": event_oldest.isoformat(), "newest": event_newest.isoformat()},
        )
        print(f"status={events.status_code}")
        if events.status_code == 200:
            data = events.json()
            print(f"Fetched {len(data)} events for {event_oldest}..{event_newest}")
            if data:
                _print_table(data, _EVENT_DIAGNOSTIC_FIELDS)
                categories = sorted({e.get("category") for e in data if e.get("category")})
                print(f"\nDistinct category values seen: {categories}")
                print(
                    "If no RACE_A/RACE_B/RACE_C shows up, widen --event-days-back/"
                    "--event-days-forward to cover a known race date."
                )

    print("\nDone. Compare findings against the assumptions documented in "
          "clients/intervals_icu/client.py and services/intervals_matching.py.")


if __name__ == "__main__":
    main()
