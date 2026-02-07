import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from api.routes import router
from api.auth_routes import router as auth_router
from config import settings
from database.mongodb import db_manager

# Configure logging
logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Configure uvicorn loggers to use the same format
for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error"]:
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(handler)
    logger.propagate = False

# Suppress MongoDB debug logging (always set to INFO or higher)
logging.getLogger("pymongo").setLevel(logging.INFO)
logging.getLogger("motor").setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown events."""
    # Startup
    await db_manager.connect()
    yield
    # Shutdown
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
