# LLM Agentic endurance coaching service 

## Project overview

Monorepo: Python backend services(sidekick) and script prototypes (analyzer, coaches).
Non-functional priorities: correctness > security > maintainability > performance.

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
