# LLM Agentic endurance coaching service 

## Project overview

Monorepo: Python backend services(sidekick) and script prototypes (analyzer, coaches).
Non-functional priorities: correctness > security > maintainability > performance.

`sidekick` is the production service. `analyzer` and `coaches` are prototypes — do
not modify them unless explicitly asked to.

## Crew definitions & prompt persistence (sidekick)

Agents, tasks, and philosophies live in the MongoDB `crew_definitions` collection,
which is the source of truth and is edited in place via the admin interface. The DB
currently runs only locally, so this content is only as safe as the last export.

- **Backup workflow:** after editing prompts in the admin UI, run
  `scripts/export_crew_definitions.py` and commit the updated
  `scripts/crew_defaults/*.yaml` snapshot (the durable, diffable backup-of-record).
  To rebuild or restore a database, run `scripts/seed_crew_definitions.py` against an
  empty database — it is insert-if-absent and will not overwrite a populated DB.
- **Reverse sync (snapshot → DB):** to apply a snapshot edited in git (e.g. by an agent)
  back to the live DB, run `scripts/seed_crew_definitions.py --overwrite` — it upserts
  changed docs and reports inserted/updated/unchanged. This replaces live content with
  the snapshot, so **export first** if the DB may hold newer admin edits, to avoid
  discarding them.
- **Guardrail (do this first):** before proposing or making any change that touches
  crew definitions — model/schema changes, adding or removing agents/tasks, or editing
  prompt content — ask the user to run the export (sync the git snapshot) or confirm it
  is already current, so live, admin-edited prompts are not lost.

## Coding standards

- Clarity over cleverness; no implicit behavior or "magic"
- Small, cohesive modules; clear naming; avoid deep nesting
- Fail fast with helpful messages; never swallow exceptions
- Composition over inheritance; keep public APIs minimal
- Comments explain "why", not "what" — delete verbose LLM-generated comments

## Python

### Package Management

- Use `uv` for dependency management (not Poetry)
- Install dependencies: `uv sync`
- Run commands: `uv run <command>`
- Each module has its own `uv.lock` file for independent dependency management
- Running scripts outside the module's src directory will need additional pythonpath: `PYTHONPATH=src uv run python scripts/<script>.py`

### Style

- Python 3.13; use modern features (pattern matching, dataclasses, etc.)
- Type hints required on public functions; prefer built-in generics (`list`, `dict`, `set`)
- Ruff formatter: line-length=88, LF line endings
- Keep imports sorted; remove unused imports

### Datetime Policy

- All `datetime` fields in Pydantic models must use `pydantic.AwareDatetime` as the field type
- Default factory for datetime fields: `default_factory=lambda: datetime.now(timezone.utc)`
- Never use `datetime.utcnow()` — deprecated in Python 3.12+, returns a naive datetime
- Never strip timezone info for MongoDB queries (no `.replace(tzinfo=None)`)
- Datetimes read back from MongoDB are naive UTC; normalise with a `field_validator(..., mode='before')` using `ensure_utc` from `utils.datetime_utils`
- Datetimes embedded in **LLM prompts** must be converted to the athlete's local timezone using `convert_datetimes_in_obj()` / `to_athlete_tz()` from `utils.datetime_utils` — the LLM must reason about local time, not UTC
- Athlete timezone is stored as an IANA string (`timezone` field on `AthleteSettings`); default `"UTC"`
- Dates presented to the user are converted to local timezone in the frontend (browser `Intl` API)
- Date-only calendar keys use `str` in `YYYY-MM-DD` format — timezone-neutral by convention

### Testing

- Do not write tests unless explicitly asked to

### Architecture

- Separate domain logic from I/O (DB, network, filesystem)
- Validate inputs at boundaries; sanitize user-controlled data
- Async code must be cancellation-safe
- Prefer stdlib; justify heavy third-party deps

### Security

- Least privilege; no secrets in logs, errors, or client bundles
- Parameterized queries; no string concatenation for queries
- Validate/encode all untrusted input (injection, SSRF, path traversal)
- Centralize token handling; check authorization at every server entrypoint

### Observability

- Structured logging with context; no PII unless masked
- Correlation IDs across services

## Development Workflow

### Making Changes

For any non-trivial change, follow this process strictly:

1. **Plan first** — Enter plan mode and design the full solution before touching any code
2. **Get approval** — Wait for explicit user approval of the plan; refine if asked
3. **Branch** — Create a new branch: `git checkout -b <type>/<short-description>` (e.g. `feat/add-pagination`, `fix/auth-token-expiry`)
4. **Implement** — Make exactly the changes approved in the plan; no scope creep
5. **Commit** — Use a descriptive commit message following the existing style
6. **PR** — Open a pull request against `main` using `gh pr create`; include the approved plan summary in the PR body

### Rules

- Never push directly to `main`
- Never implement before the plan is approved
- Never create a PR without a branch
- Keep PRs focused — one logical change per PR
- PR title: short and imperative (e.g. "Add workout history pagination")
- PR body: summarize what was planned, what changed, and why
