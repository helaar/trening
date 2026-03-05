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
