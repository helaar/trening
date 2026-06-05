"""One-time migration: convert flat philosophy.* keys to philosophy.{slug}.* format.

Before:
  philosophy.name                   → "Polarized (80/20)"
  philosophy.intensity_targets      → "..."
  philosophy.coach_guidance         → "..."
  philosophy.analyst_guidance       → "..."
  philosophy.{athlete_id}.name      → "Polarized (80/20)"  (athlete override or selection)

After:
  philosophy.{slug}.name            → "Polarized (80/20)"
  philosophy.{slug}.intensity_targets
  philosophy.{slug}.coach_guidance
  philosophy.{slug}.analyst_guidance
  philosophy.{athlete_id}.selected  → "{slug}"

Run once:
  cd sidekick
  PYTHONPATH=src uv run python scripts/migrate_philosophy_keys.py
"""
import asyncio
import logging
import re
import sys

from database.mongodb import db_manager
from database.prompt_repository import PromptRepository

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_SUB_KEYS = ("name", "intensity_targets", "coach_guidance", "analyst_guidance")


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


async def migrate() -> None:
    await db_manager.connect()
    try:
        repo = PromptRepository(db_manager.db)
        all_prompts = await repo.get_all()
        by_key = {p.key: p.value for p in all_prompts}

        # ── 1. identify the flat global philosophy ────────────────────────────
        flat_name = by_key.get("philosophy.name", "").strip()
        if not flat_name:
            logger.info("No legacy flat philosophy.name found — nothing to migrate.")
            return

        slug = _slugify(flat_name)
        if not slug:
            logger.error(f"Could not derive a slug from name {flat_name!r}. Aborting.")
            sys.exit(1)

        logger.info(f"Migrating philosophy {flat_name!r} → slug {slug!r}")

        # ── 2. write new philosophy.{slug}.* keys ─────────────────────────────
        new_items = [
            {"key": f"philosophy.{slug}.{sub}", "value": by_key.get(f"philosophy.{sub}", "")}
            for sub in _SUB_KEYS
        ]
        await repo.upsert_many(new_items)
        logger.info(f"  Wrote {len(new_items)} keys under philosophy.{slug}.*")

        # ── 3. migrate athlete selection keys ─────────────────────────────────
        athlete_old_keys = [
            k for k in by_key
            if k.startswith("philosophy.") and k.endswith(".name")
            and k != "philosophy.name"
            and k.split(".")[1].isdigit()
        ]
        selection_items = [
            {"key": f"philosophy.{k.split('.')[1]}.selected", "value": slug}
            for k in athlete_old_keys
        ]
        if selection_items:
            await repo.upsert_many(selection_items)
            logger.info(f"  Converted {len(selection_items)} athlete selection key(s)")

        # ── 4. delete old flat and athlete-override keys ──────────────────────
        keys_to_delete = [f"philosophy.{sub}" for sub in _SUB_KEYS] + athlete_old_keys
        existing_to_delete = [k for k in keys_to_delete if k in by_key]
        if existing_to_delete:
            await repo.delete_many(existing_to_delete)
            logger.info(f"  Deleted {len(existing_to_delete)} legacy key(s): {existing_to_delete}")

        logger.info("Migration complete.")
    finally:
        await db_manager.disconnect()


asyncio.run(migrate())
