import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from pymongo import AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase

from config import settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages MongoDB database connections using async PyMongo."""

    def __init__(self):
        self.client: AsyncMongoClient | None = None
        self.db: AsyncDatabase | None = None

    async def connect(self):
        """Connect to MongoDB."""
        try:
            logger.info(f"Connecting to MongoDB at {settings.mongodb_url}")
            self.client = AsyncMongoClient(settings.mongodb_url)
            self.db = self.client[settings.mongodb_database]
            
            # Test the connection
            await self.client.admin.command("ping")
            logger.info("Successfully connected to MongoDB")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise

    async def disconnect(self):
        """Disconnect from MongoDB."""
        if self.client:
            await self.client.close()
            logger.info("Disconnected from MongoDB")

    async def ping(self) -> bool:
        """
        Check if database connection is alive.

        Returns:
            True if connection is alive, False otherwise
        """
        try:
            await self.client.admin.command("ping")
            return True
        except Exception as e:
            logger.error(f"Database ping failed: {e}")
            return False


# Global database manager instance
db_manager = DatabaseManager()


@asynccontextmanager
async def get_database() -> AsyncGenerator[AsyncDatabase, None]:
    """
    Async context manager for database connections.

    Usage:
        async with get_database() as db:
            # perform database operations
            collection = db['my_collection']
    """
    if db_manager.db is None:
        raise RuntimeError("Database not connected. Call db_manager.connect() first.")
    try:
        yield db_manager.db
    finally:
        pass


def get_db() -> AsyncDatabase:
    """
    Dependency function to get database instance for FastAPI.
    
    Returns the database directly without context manager.
    """
    if db_manager.db is None:
        raise RuntimeError("Database not connected. Call db_manager.connect() first.")
    return db_manager.db
