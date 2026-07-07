"""Dump the `coach_memories` collection for local inspection.

Read-only diagnostic — writes nothing back to MongoDB. Includes inactive and
expired memories (not just what `MemoryRepository.get_active` would return)
so you can see the full history, e.g. to spot memories created during testing
that are still active and shaping live coaching output.

Never commit the output file — it is per-athlete personal training data.

Run:
  cd sidekick
  PYTHONPATH=src uv run python scripts/export_memories.py [--athlete-id ID] [--output PATH]

With no --athlete-id, dumps every athlete's memories (grouped). With no
--output, prints a summary table only (no full content) to stdout.
"""
import argparse
import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from database.mongodb import db_manager

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--athlete-id", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None, help="JSON file to write full dump to")
    return parser.parse_args()


async def _fetch(athlete_id: int | None) -> list[dict]:
    query: dict = {} if athlete_id is None else {"athlete_id": athlete_id}
    cursor = db_manager.db["coach_memories"].find(query)
    docs = []
    async for doc in cursor:
        doc.pop("_id", None)
        docs.append(doc)
    return docs


def _annotate(doc: dict, now: datetime) -> dict:
    expires_at = doc.get("expires_at")
    doc["expired"] = bool(expires_at and expires_at < now)
    return doc


def _print_summary(docs: list[dict]) -> None:
    by_athlete: dict[int, list[dict]] = {}
    for d in docs:
        by_athlete.setdefault(d["athlete_id"], []).append(d)

    for athlete_id, athlete_docs in sorted(by_athlete.items()):
        print(f"\nathlete_id={athlete_id}  ({len(athlete_docs)} memories)")
        athlete_docs.sort(
            key=lambda d: (d["category"], not d["active"], d["expired"], -d["importance"])
        )
        for d in athlete_docs:
            flags = []
            if not d["active"]:
                flags.append("INACTIVE")
            if d["expired"]:
                flags.append("EXPIRED")
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            print(
                f"  {d['scope']:10s} {d['category']:10s} "
                f"conf={d['confidence']:.2f} imp={d['importance']:.2f}{flag_str}"
            )
            print(f"      {d['content']}")


async def export() -> None:
    args = _parse_args()
    await db_manager.connect()
    try:
        docs = await _fetch(args.athlete_id)
        now = datetime.now(timezone.utc)
        docs = [_annotate(d, now) for d in docs]

        if not docs:
            logger.info("No memories found.")
            return

        _print_summary(docs)

        if args.output:
            args.output.write_text(json.dumps(docs, indent=2, default=str))
            logger.info("\nWrote %d memories -> %s", len(docs), args.output)
    finally:
        await db_manager.disconnect()


asyncio.run(export())
