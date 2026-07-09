"""Deactivate specific coach memories by ID — targeted cleanup for duplicates or
oversized memories that automated consolidation missed.

Dry-run by default: shows what would change without writing anything. Pass --apply
to actually commit. Find memory_ids with scripts/export_memories.py (each printed
line now shows `id=...`).

Run:
  cd sidekick
  PYTHONPATH=src uv run python scripts/deactivate_memories.py --memory-id ID1 --memory-id ID2 [--apply]
"""
import argparse
import asyncio
import logging

from database.memory_repository import MemoryRepository
from database.mongodb import db_manager

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--memory-id", action="append", required=True, dest="memory_ids",
        help="Memory ID to deactivate; repeat for multiple",
    )
    parser.add_argument("--apply", action="store_true", help="Write the change (default: dry-run)")
    return parser.parse_args()


async def deactivate() -> None:
    args = _parse_args()
    await db_manager.connect()
    try:
        memory_repo = MemoryRepository(db_manager.db)
        for memory_id in args.memory_ids:
            doc = await memory_repo.collection.find_one({"memory_id": memory_id})
            if not doc:
                logger.warning("No memory found with id=%s — skipping", memory_id)
                continue
            if not doc.get("active", True):
                logger.info("%s: already inactive — skipping", memory_id)
                continue
            preview = doc.get("content", "")[:80]
            if args.apply:
                await memory_repo.deactivate(memory_id)
                logger.info("%s: deactivated — %s", memory_id, preview)
            else:
                logger.info("%s: would deactivate (dry-run) — %s", memory_id, preview)

        if not args.apply:
            logger.info("\nDry-run only — no changes written. Re-run with --apply to commit.")
    finally:
        await db_manager.disconnect()


asyncio.run(deactivate())
