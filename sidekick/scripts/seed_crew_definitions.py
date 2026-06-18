"""Seed/sync the `crew_definitions` collection from the git-tracked YAML snapshot.

Reads scripts/crew_defaults/{agents,tasks,philosophies}.yaml as typed documents.
The snapshot is produced from the live DB by scripts/export_crew_definitions.py;
MongoDB stays the source of truth. (philosophies.yaml only exists once that export
has been run, so it is optional here.)

Default (insert-if-absent): existing (type, name) docs are never overwritten, so
re-running is safe and never clobbers admin-edited values — rebuild a lost DB by
seeding into an empty database:
  cd sidekick
  PYTHONPATH=src uv run python scripts/seed_crew_definitions.py

Reverse sync (--overwrite): apply a snapshot edited in git back to the live DB,
upserting changed docs. This REPLACES live content with the snapshot, so export
first if the DB may hold newer admin edits:
  PYTHONPATH=src uv run python scripts/seed_crew_definitions.py --overwrite
"""
import argparse
import asyncio
import logging
from pathlib import Path

import yaml

from database.crew_definition_repository import CrewDefinitionRepository
from database.mongodb import db_manager
from models.crew_definition import AgentDoc, CrewDefinition, PhilosophyDoc, TaskDoc

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_DEFAULTS_DIR = Path(__file__).parent / "crew_defaults"


def _load_yaml(filename: str) -> dict:
    """Load a seed-default YAML snapshot from scripts/crew_defaults.

    Returns an empty dict if the file is absent (philosophies.yaml is optional).
    """
    path = _DEFAULTS_DIR / filename
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build_definitions() -> list[CrewDefinition]:
    defs: list[CrewDefinition] = []

    for name, cfg in _load_yaml("agents.yaml").items():
        defs.append(
            AgentDoc(
                name=name,
                role=cfg["role"],
                goal=cfg["goal"],
                backstory=cfg["backstory"],
                llm_model=cfg["llm_model"],
            )
        )

    for name, cfg in _load_yaml("tasks.yaml").items():
        defs.append(
            TaskDoc(
                name=name,
                description=cfg["description"],
                expected_output=cfg["expected_output"],
            )
        )

    for name, cfg in _load_yaml("philosophies.yaml").items():
        defs.append(
            PhilosophyDoc(
                name=name,
                display_name=cfg.get("display_name", ""),
                intensity_targets=cfg.get("intensity_targets", ""),
                coach_guidance=cfg.get("coach_guidance", ""),
                analyst_guidance=cfg.get("analyst_guidance", ""),
            )
        )

    return defs


async def seed(overwrite: bool) -> None:
    await db_manager.connect()
    try:
        repo = CrewDefinitionRepository(db_manager.db)
        await repo.ensure_indexes()

        if not overwrite:
            inserted = skipped = 0
            for doc in _build_definitions():
                if await repo.insert_if_absent(doc):
                    inserted += 1
                    logger.info("  inserted %s '%s'", doc.type, doc.name)
                else:
                    skipped += 1
                    logger.info("  skipped  %s '%s' (already present)", doc.type, doc.name)
            logger.info("Seed complete: %d inserted, %d skipped.", inserted, skipped)
            return

        existing = {(d.type, d.name): d for d in await repo.get_all()}
        inserted = updated = unchanged = 0
        for doc in _build_definitions():
            current = existing.get((doc.type, doc.name))
            if current is None:
                await repo.upsert(doc)
                inserted += 1
                logger.info("  inserted  %s '%s'", doc.type, doc.name)
            elif doc.model_dump(exclude={"updated_at"}) != current.model_dump(
                exclude={"updated_at"}
            ):
                await repo.upsert(doc)
                updated += 1
                logger.info("  updated   %s '%s'", doc.type, doc.name)
            else:
                unchanged += 1
                logger.info("  unchanged %s '%s'", doc.type, doc.name)
        logger.info(
            "Sync complete: %d inserted, %d updated, %d unchanged.",
            inserted,
            updated,
            unchanged,
        )
    finally:
        await db_manager.disconnect()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Apply the git snapshot back to the DB, replacing changed docs.",
    )
    return parser.parse_args()


asyncio.run(seed(_parse_args().overwrite))
