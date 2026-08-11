# Sidekick

Joint solution integrating functionality from analyzer and coaches projects.

## Prerequisites

- Docker and Docker Compose (dev and prod both run fully containerized)
- `uv` (only needed if working on sidekick outside Docker, e.g. quick script runs)

## Quick Start (local development)

Compose files live at the repo root, not here, since the stack spans
`sidekick`, `frontend`, and `mongodb` together.

```bash
# From the repo root
cp .env.example .env   # fill in real values (Strava keys, JWT secret, etc.)

# Start the full stack with live-reload (bare `up` auto-merges
# docker-compose.override.yml)
docker compose up -d

# Optionally start Mongo Express too
docker compose --profile tools up -d
```

- **API**: http://localhost:5175
- **Docs**: http://localhost:5175/docs
- **Health**: http://localhost:5175/health
- **Frontend**: http://localhost:5173
- **MongoDB**: localhost:27010
- **Mongo Express** (with `--profile tools`): http://localhost:8001

Editing files under `src/` reloads the running container automatically.

## Configuration

See `.env.example` at the repo root for every setting sidekick reads
(`config.py`), including:
- `MONGODB_URL` / `MONGODB_DATABASE` — MongoDB connection
- `ENVIRONMENT`, `LOG_LEVEL`
- `STRAVA_CLIENT_ID`, `STRAVA_CLIENT_SECRET` — Strava OAuth
- `JWT_SECRET_KEY` — session token signing

See [`docs/STRAVA_AUTH.md`](docs/STRAVA_AUTH.md) for Strava OAuth setup.

## Maintenance scripts

Scripts under `scripts/` (crew definition export/seed, memory tools, etc.)
import from `src/`, so they need `src` on `PYTHONPATH`. Run them inside the
running container:

```bash
docker compose exec sidekick sh -c "PYTHONPATH=src python scripts/export_crew_definitions.py"
```

## Docker Commands (from the repo root)

```bash
docker compose up -d          # Start the full stack (dev mode, live-reload)
docker compose down           # Stop services
docker compose logs -f        # View logs
docker compose down -v        # Remove volumes (clears data)

# Production (never bare `up` — that merges docker-compose.override.yml):
docker compose -f docker-compose.yml up -d --build
```
