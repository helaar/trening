import logging
import logging.config
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from api.routes import router
from api.auth_routes import router as auth_router
from config import settings
from crew.prompt_logging import register_prompt_log_listener
from database.crew_definition_repository import CrewDefinitionRepository
from database.mongodb import db_manager
from database.prompt_log_repository import PromptLogRepository
from services.task_processor import TaskProcessor

# Configure logging from logging.yaml (per-area levels, consistent format).
# Falls back to a basic stdout config using log_level if the file is missing.
_LOGGING_CONFIG_PATH = Path(__file__).resolve().parent.parent / "logging.yaml"


def _configure_logging() -> None:
    try:
        with open(_LOGGING_CONFIG_PATH) as f:
            logging.config.dictConfig(yaml.safe_load(f))
    except FileNotFoundError:
        logging.basicConfig(
            level=settings.log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
        logging.getLogger(__name__).warning(
            "logging.yaml not found at %s; using basic stdout config", _LOGGING_CONFIG_PATH
        )


_configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown events."""
    # Startup
    await db_manager.connect()
    await PromptLogRepository(db_manager.db).ensure_indexes()
    await CrewDefinitionRepository(db_manager.db).ensure_indexes()
    register_prompt_log_listener()
    logging.getLogger(__name__).info(
        "Startup complete: LLM token-usage capture active (prompt_logs doc_type=usage)"
    )
    yield
    # Shutdown
    await TaskProcessor.shutdown()
    await db_manager.disconnect()


app = FastAPI(
    title="Sidekick API",
    description="Joint solution integrating functionality from analyzer and coaches",
    version="0.1.0",
    lifespan=lifespan
)

# Configure CORS - environment-based settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_methods_list,
    allow_headers=settings.cors_headers_list,
)

# Add GZip middleware for response compression
app.add_middleware(GZipMiddleware, minimum_size=2048)

# Include API routes
app.include_router(router)
app.include_router(auth_router)


@app.get("/")
async def root():
    """Root endpoint returning API status."""
    return {
        "name": "Sidekick API",
        "version": "0.1.0",
        "environment": settings.environment,
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    db_healthy = await db_manager.ping()
    return {
        "status": "healthy" if db_healthy else "degraded",
        "database": "connected" if db_healthy else "disconnected"
    }
