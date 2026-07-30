"""Phase 0 spike: inspect the real Intervals.icu API response shapes.

Throwaway exploration tool, not shipped as product code — safe to delete once
its findings have been used to confirm/correct the field-name assumptions in
clients/intervals_icu/client.py and services/intervals_matching.py. Standalone
(no sidekick imports) so it can run without PYTHONPATH or a database.

Confirms, per the approved plan:
  1. Auth (HTTP Basic, username "API_KEY") and the athlete-id path format.
  2. Whether GET /athlete/{id}/activities ever carries a Strava-id
     back-reference — test this against activities that reached Intervals.icu
     via Garmin/Zwift *without* passing through Strava, not just
     Strava-sourced ones, since the real device setup fans out to both
     independently.
  3. Field names for NP/IF/TSS-equivalent, decoupling, eFTP, and the
     per-interval/lap breakdown.
  4. Wellness endpoint field names (HRV, resting HR, sleep, weight, CTL/ATL).
  5. Calendar/events endpoint shape, including the race-priority category
     field and any activity-level RPE field.
  6. Rate-limit headers and Strava-upload-to-Intervals-visibility lag.

Run:
  cd sidekick
  uv run python scripts/spike_intervals_icu.py --api-key YOUR_KEY --athlete-id YOUR_ID
"""

import argparse
import json
from datetime import date, timedelta

import httpx

BASE_URL = "https://intervals.icu/api/v1"


def _print_section(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def _print_json(label: str, data) -> None:
    print(f"--- {label} ---")
    print(json.dumps(data, indent=2, default=str)[:4000])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-key", required=True, help="Intervals.icu API key (Settings -> Developer)")
    parser.add_argument("--athlete-id", required=True, help="Intervals.icu athlete id")
    parser.add_argument("--days", type=int, default=30, help="Lookback window in days")
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
            first = activities[0]
            _print_json("first activity — inspect for a Strava-id field (2)", first)
            print("\nTop-level keys present:", sorted(first.keys()))
            print(
                "\nCheck manually: does this activity carry a strava_id / external_id "
                "field, especially for entries whose source looks like Garmin/Zwift "
                "rather than Strava? Compare across multiple activities, not just this one."
            )

            activity_id = first.get("id")
            if activity_id:
                _print_section("3. Single activity detail (NP/IF/TSS/decoupling/eFTP field names)")
                detail = client.get(f"/activity/{activity_id}")
                print(f"status={detail.status_code}")
                if detail.status_code == 200:
                    _print_json("activity detail", detail.json())
                else:
                    print(f"Path /activity/{{id}} may be wrong — try alternates, e.g. "
                          f"{athlete_path}/activities/{activity_id}")

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
                print("\nTop-level keys present:", sorted(data[-1].keys()))

        _print_section("5. Calendar / events (planned workouts, race-priority tagging, RPE)")
        events = client.get(
            f"{athlete_path}/events",
            params={"oldest": oldest.isoformat(), "newest": (newest + timedelta(days=60)).isoformat()},
        )
        print(f"status={events.status_code}")
        if events.status_code == 200:
            data = events.json()
            print(f"Fetched {len(data)} events")
            if data:
                _print_json("first event", data[0])
                print("\nLook for: category/type field (race priority: RACE_A/RACE_B/RACE_C?), "
                      "and any per-activity RPE/feel field on completed activities above.")

    print("\nDone. Compare findings against the assumptions documented in "
          "clients/intervals_icu/client.py and services/intervals_matching.py.")


if __name__ == "__main__":
    main()
