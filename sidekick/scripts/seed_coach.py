"""Seed a coach for the `coaches` collection.

Idempotent: upserts the given coach_id with the given roster, so re-running with the
same args is safe. For the current single-Strava-user DB, seed the owner as a coach
with their own athlete_id as a self-roster, then append more athlete IDs later:

  cd sidekick
  PYTHONPATH=src uv run python scripts/seed_coach.py <coach_id> --display-name "Name" --roster <coach_id>
  PYTHONPATH=src uv run python scripts/seed_coach.py <coach_id> --roster <coach_id> <athlete_id_2> <athlete_id_3>
"""
import argparse
import asyncio
import logging

from database.coach_repository import CoachRepository
from database.mongodb import db_manager
from models.coach import Coach

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


async def _seed(coach_id: int, display_name: str | None, roster: list[int]) -> None:
    await db_manager.connect()
    try:
        repo = CoachRepository(db_manager.db)
        existing = await repo.get_coach(coach_id)
        kwargs = {
            "coach_id": coach_id,
            "display_name": display_name or (existing.display_name if existing else None),
            "athlete_ids": roster,
        }
        if existing:
            kwargs["created_at"] = existing.created_at
        coach = Coach(**kwargs)
        await repo.upsert(coach)
        logger.info(f"Seeded coach {coach_id} with roster {roster}")
    finally:
        await db_manager.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coach_id", type=int, help="Strava athlete ID of the coach")
    parser.add_argument("--display-name", default=None, help="Display name for the coach")
    parser.add_argument(
        "--roster",
        type=int,
        nargs="+",
        required=True,
        help="Athlete IDs the coach may read (include coach_id for self-roster)",
    )
    args = parser.parse_args()
    asyncio.run(_seed(args.coach_id, args.display_name, args.roster))


if __name__ == "__main__":
    main()
