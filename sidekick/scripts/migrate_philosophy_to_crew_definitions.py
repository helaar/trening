"""One-time migration: move agent/task/philosophy data out of `prompt_configs`.

The legacy flat `prompt_configs` collection held two kinds of data as
dot-notation keys:
  - Admin overrides of agent/task fields, e.g. agents.daily_coach.backstory or
    tasks.workout_analysis_task.description.
  - Training philosophies and per-athlete selections, e.g.
    philosophy.{slug}.name / .intensity_targets / .coach_guidance /
    .analyst_guidance, and philosophy.{athlete_id}.selected.

Run scripts/seed_crew_definitions.py FIRST to bootstrap agent/task docs from the
YAML defaults; this migration then:
  - Overlays any agents.*/tasks.* admin overrides onto those seeded docs so
    edits made through the old admin UI are preserved (override wins over the
    YAML default).
  - Writes each philosophy as a typed PhilosophyDoc in `crew_definitions`.
  - Moves each athlete's selection onto AthleteSettings.training_philosophy.
  - Deletes the now-obsolete prompt_configs keys.

Idempotent: docs are upserted; selection patches are naturally idempotent; key
deletion is a no-op on a second run.

Run once (after seeding crew_definitions):
  cd sidekick
  PYTHONPATH=src uv run python scripts/migrate_philosophy_to_crew_definitions.py
"""
import asyncio
import logging

from database.athlete_repository import AthleteRepository
from database.crew_definition_repository import CrewDefinitionRepository
from database.mongodb import db_manager
from database.prompt_repository import PromptRepository
from models.crew_definition import AgentDoc, PhilosophyDoc, TaskDoc

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_SUB_KEYS = ("name", "intensity_targets", "coach_guidance", "analyst_guidance")


async def migrate() -> None:
    await db_manager.connect()
    try:
        prompts = PromptRepository(db_manager.db)
        crew_defs = CrewDefinitionRepository(db_manager.db)
        athletes = AthleteRepository(db_manager.db)
        await crew_defs.ensure_indexes()

        by_key = {p.key: p.value for p in await prompts.get_all()}

        # ── 1. group keys by kind ─────────────────────────────────────────────
        agent_overrides: dict[str, dict[str, str]] = {}
        task_overrides: dict[str, dict[str, str]] = {}
        slugs: dict[str, dict[str, str]] = {}
        selections: dict[int, str] = {}
        for key, value in by_key.items():
            parts = key.split(".")
            if len(parts) < 3:
                continue
            prefix, mid, field = parts[0], parts[1], ".".join(parts[2:])
            if prefix == "agents":
                agent_overrides.setdefault(mid, {})[field] = value
            elif prefix == "tasks":
                task_overrides.setdefault(mid, {})[field] = value
            elif prefix == "philosophy":
                if mid.isdigit():
                    if field == "selected" and value:
                        selections[int(mid)] = value
                elif field in _SUB_KEYS:
                    slugs.setdefault(mid, {})[field] = value

        # ── 2. overlay agent/task overrides onto seeded docs ──────────────────
        for name, fields in agent_overrides.items():
            doc = await crew_defs.get_agent(name)
            if not doc:
                logger.warning("  agent '%s' not seeded — override skipped (run seed first)", name)
                continue
            await crew_defs.upsert(AgentDoc(**{**doc.model_dump(), **fields}))
            logger.info("  applied %d override(s) to agent '%s'", len(fields), name)

        for name, fields in task_overrides.items():
            doc = await crew_defs.get_task(name)
            if not doc:
                logger.warning("  task '%s' not seeded — override skipped (run seed first)", name)
                continue
            await crew_defs.upsert(TaskDoc(**{**doc.model_dump(), **fields}))
            logger.info("  applied %d override(s) to task '%s'", len(fields), name)

        # ── 3. write philosophy documents ─────────────────────────────────────
        for slug, fields in slugs.items():
            doc = PhilosophyDoc(
                name=slug,
                display_name=fields.get("name", ""),
                intensity_targets=fields.get("intensity_targets", ""),
                coach_guidance=fields.get("coach_guidance", ""),
                analyst_guidance=fields.get("analyst_guidance", ""),
            )
            await crew_defs.upsert(doc)
            logger.info("  upserted philosophy '%s' (%s)", slug, doc.display_name)

        # ── 4. move athlete selections onto settings ──────────────────────────
        for athlete_id, slug in selections.items():
            updated = await athletes.patch_athlete_settings(
                athlete_id, {"training_philosophy": slug}
            )
            if updated:
                logger.info("  set athlete %d training_philosophy = '%s'", athlete_id, slug)
            else:
                logger.warning("  athlete %d not found — selection '%s' skipped", athlete_id, slug)

        # ── 5. delete obsolete prompt_configs keys ────────────────────────────
        obsolete = [
            k for k in by_key
            if k.startswith(("philosophy.", "agents.", "tasks."))
        ]
        if obsolete:
            await prompts.delete_many(obsolete)
            logger.info("  deleted %d obsolete prompt_configs key(s)", len(obsolete))

        logger.info(
            "Migration complete: %d agent override(s), %d task override(s), "
            "%d philosophy doc(s), %d athlete selection(s).",
            len(agent_overrides), len(task_overrides), len(slugs), len(selections),
        )
    finally:
        await db_manager.disconnect()


asyncio.run(migrate())
