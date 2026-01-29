# Sidekick

Joint solution integrating functionality from analyzer and coaches projects.

## Prerequisites

- Python 3.10+
- Docker and Docker Compose
- Poetry

## Quick Start

```bash
# Install dependencies
poetry install

# Start MongoDB
docker-compose up -d

# MongoDB will be available at localhost:27010
# Mongo Express UI at http://localhost:8001
```

## Configuration

Environment variables in [`.env`](.env):
- `MONGODB_URL` - MongoDB connection string
- `MONGODB_DATABASE` - Database name
- `ENVIRONMENT` - Application environment
- `LOG_LEVEL` - Logging level

## Docker Commands

```bash
docker-compose up -d      # Start services
docker-compose down       # Stop services
docker-compose logs -f    # View logs
docker-compose down -v    # Remove volumes (clears data)
```
