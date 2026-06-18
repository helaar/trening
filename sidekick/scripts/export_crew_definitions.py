"""Export the live `crew_definitions` collection to the git-tracked YAML snapshot.

MongoDB is the source of truth for agents, tasks, and philosophies — but it only
lives on one machine. This script dumps the collection to
scripts/crew_defaults/{agents,tasks,philosophies}.yaml, which is the durable,
diffable, review-able backup. Commit the result after editing prompts in the
admin UI.

Rebuild a lost/fresh database from the snapshot with scripts/seed_crew_definitions.py.

`updated_at` is intentionally omitted so diffs stay content-only; multi-line
prompt text is written as `|` block scalars so git diffs are readable.

Run:
  cd sidekick
  PYTHONPATH=src uv run python scripts/export_crew_definitions.py
"""
import asyncio
import logging
from pathlib import Path

import yaml

from database.crew_definition_repository import CrewDefinitionRepository
from database.mongodb import db_manager

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_DEFAULTS_DIR = Path(__file__).parent / "crew_defaults"

# Field order per type — keeps the snapshot stable and matches the seed's reader.
_FIELDS: dict[str, tuple[str, ...]] = {
    "agent": ("role", "goal", "llm_model", "backstory"),
    "task": ("description", "expected_output"),
    "philosophy": ("display_name", "intensity_targets", "coach_guidance", "analyst_guidance"),
}

_FILES = {"agent": "agents.yaml", "task": "tasks.yaml", "philosophy": "philosophies.yaml"}


def _str_repr(dumper: yaml.Dumper, data: str):
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


yaml.add_representer(str, _str_repr)


def _dump(type_: str, docs: list) -> None:
    fields = _FIELDS[type_]
    by_name = {
        doc.name: {f: getattr(doc, f) for f in fields}
        for doc in sorted(docs, key=lambda d: d.name)
    }
    path = _DEFAULTS_DIR / _FILES[type_]
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(
            by_name,
            f,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )
    logger.info("  wrote %d %s(s) -> %s", len(by_name), type_, path.name)


async def export() -> None:
    await db_manager.connect()
    try:
        repo = CrewDefinitionRepository(db_manager.db)
        all_defs = await repo.get_all()
        by_type: dict[str, list] = {"agent": [], "task": [], "philosophy": []}
        for doc in all_defs:
            by_type[doc.type].append(doc)

        _DEFAULTS_DIR.mkdir(parents=True, exist_ok=True)
        for type_ in ("agent", "task", "philosophy"):
            _dump(type_, by_type[type_])

        logger.info("Export complete.")
    finally:
        await db_manager.disconnect()


asyncio.run(export())
