"""Bootstrap the `crew_definitions` collection from the legacy YAML.

Reads the agent/task definitions that historically lived in
src/crew/agents.yaml and src/crew/tasks.yaml and inserts them as typed
documents. Insert-if-absent: existing (type, name) docs are never overwritten,
so re-running is safe and never clobbers admin-edited values.

Philosophy documents are NOT seeded here — they are migrated from the legacy
prompt_configs collection by scripts/migrate_philosophy_to_crew_definitions.py.

Run once (or any time to fill in missing definitions):
  cd sidekick
  PYTHONPATH=src uv run python scripts/seed_crew_definitions.py
"""
import asyncio
import logging
from pathlib import Path

import yaml

from database.crew_definition_repository import CrewDefinitionRepository
from database.mongodb import db_manager
from models.crew_definition import AgentDoc, CrewDefinition, TaskDoc

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_DEFAULTS_DIR = Path(__file__).parent / "crew_defaults"


def _load_yaml(filename: str) -> dict:
    """Load a seed-default YAML snapshot from scripts/crew_defaults."""
    path = _DEFAULTS_DIR / filename
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

    return defs


async def seed() -> None:
    await db_manager.connect()
    try:
        repo = CrewDefinitionRepository(db_manager.db)
        await repo.ensure_indexes()

        inserted = skipped = 0
        for doc in _build_definitions():
            if await repo.insert_if_absent(doc):
                inserted += 1
                logger.info("  inserted %s '%s'", doc.type, doc.name)
            else:
                skipped += 1
                logger.info("  skipped  %s '%s' (already present)", doc.type, doc.name)

        logger.info("Seed complete: %d inserted, %d skipped.", inserted, skipped)
    finally:
        await db_manager.disconnect()


asyncio.run(seed())
