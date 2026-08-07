"""Intervals.icu API client.

Auth is a per-athlete API key (HTTP Basic, username "API_KEY"), configured via
AthleteSettings.intervals_icu — no OAuth dance, matching the convention already
used for the TrainingPeaks calendar-sync URL. Strava remains the sole login
mechanism; this client is a second data source, not a second auth path.

All five endpoint paths below were confirmed (HTTP 200) against a real
Intervals.icu account via scripts/spike_intervals_icu.py on 2026-07-30.
Response-field mapping (NP/TSS/zones/eFTP/wellness/etc.) is deferred to
Phase 2 — this client only returns raw JSON.
"""

from datetime import date

import httpx

from models.athlete import AthleteSettings

BASE_URL = "https://intervals.icu/api/v1"


class IntervalsIcuError(Exception):
    """Raised for any non-2xx response from the Intervals.icu API.

    Carries the HTTP status and the decoded response body (Intervals.icu
    returns JSON error bodies) so callers get an actionable message instead of
    a bare httpx traceback.
    """

    def __init__(self, status_code: int, body: str):
        self.status_code = status_code
        self.body = body
        super().__init__(f"Intervals.icu API error {status_code}: {body}")


class IntervalsIcuNotConfigured(Exception):
    """Raised when an athlete has no Intervals.icu API key/athlete id configured."""


class IntervalsIcuRateLimited(IntervalsIcuError):
    """Raised on HTTP 429. retry_after_seconds is None if the response has no header."""

    def __init__(self, status_code: int, body: str, retry_after_seconds: float | None):
        self.retry_after_seconds = retry_after_seconds
        super().__init__(status_code, body)


class IntervalsIcuClient:
    """Thin async wrapper over the Intervals.icu REST API for a single athlete.

    No caching or retry logic here — that's the caller's/repository's job, the
    same split of responsibility as StravaClient (thin API wrapper) vs.
    WorkoutRepository (storage/staleness semantics).
    """

    def __init__(self, api_key: str, intervals_athlete_id: str):
        if not api_key or not intervals_athlete_id:
            raise IntervalsIcuNotConfigured(
                "Intervals.icu API key and athlete id must both be set"
            )
        self._api_key = api_key
        self._athlete_path = f"/athlete/{intervals_athlete_id}"

    @classmethod
    def from_athlete_settings(cls, settings: AthleteSettings) -> "IntervalsIcuClient":
        """Build a client from AthleteSettings.intervals_icu, or raise if unconfigured."""
        icu_settings = settings.intervals_icu
        if icu_settings is None or not icu_settings.api_key or not icu_settings.intervals_athlete_id:
            raise IntervalsIcuNotConfigured(
                "Athlete has not configured Intervals.icu integration"
            )
        return cls(icu_settings.api_key, icu_settings.intervals_athlete_id)

    async def _get(self, path: str, params: dict | None = None) -> httpx.Response:
        async with httpx.AsyncClient(
            base_url=BASE_URL,
            auth=("API_KEY", self._api_key),
            timeout=15,
        ) as client:
            response = await client.get(path, params=params)

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            raise IntervalsIcuRateLimited(
                response.status_code,
                response.text,
                float(retry_after) if retry_after else None,
            )
        if response.is_error:
            raise IntervalsIcuError(response.status_code, response.text)
        return response

    async def get_activities(self, oldest: date, newest: date) -> list[dict]:
        """Activities in [oldest, newest], each activity's raw JSON as returned by the API.

        Used by the identity-matching service to find a Strava-id back-reference
        (the `strava_id` field, confirmed — see services/intervals_matching.py) or,
        failing that, to fuzzy-match on start time/duration/distance.
        """
        response = await self._get(
            f"{self._athlete_path}/activities",
            params={"oldest": oldest.isoformat(), "newest": newest.isoformat()},
        )
        return response.json()

    async def get_activity(self, intervals_activity_id: str) -> dict:
        """Full detail for a single Intervals.icu activity."""
        response = await self._get(f"/activity/{intervals_activity_id}")
        return response.json()

    async def get_activity_intervals(self, intervals_activity_id: str) -> dict:
        """Per-interval/lap breakdown for a single activity."""
        response = await self._get(f"/activity/{intervals_activity_id}/intervals")
        return response.json()

    async def get_wellness(self, oldest: date, newest: date) -> list[dict]:
        """Daily wellness entries (HRV, resting HR, sleep, CTL/ATL/ramp rate, etc.)."""
        response = await self._get(
            f"{self._athlete_path}/wellness.json",
            params={"oldest": oldest.isoformat(), "newest": newest.isoformat()},
        )
        return response.json()

    async def get_events(self, oldest: date, newest: date) -> list[dict]:
        """Calendar events (planned workouts) in [oldest, newest]."""
        response = await self._get(
            f"{self._athlete_path}/events",
            params={"oldest": oldest.isoformat(), "newest": newest.isoformat()},
        )
        return response.json()
