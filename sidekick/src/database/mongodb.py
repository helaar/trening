import logging
from contextlib import contextmanager
from typing import Generator

from pymongo import MongoClient
from pymongo.database import Database

from ..config import settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages MongoDB database connections using PyMongo."""

    def __init__(self):
        self.client: MongoClient | None = None
        self.db: Database | None = None

    def connect(self):
        """Connect to MongoDB."""
        try:
            logger.info(f"Connecting to MongoDB at {settings.mongodb_url}")
            self.client = MongoClient(settings.mongodb_url)
            self.db = self.client[settings.mongodb_database]
            
            # Test the connection
            self.client.admin.command("ping")
            logger.info("Successfully connected to MongoDB")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise

    def disconnect(self):
        """Disconnect from MongoDB."""
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB")

    def ping(self) -> bool:
        """
        Check if database connection is alive.

        Returns:
            True if connection is alive, False otherwise
        """
        try:
            self.client.admin.command("ping")
            return True
        except Exception as e:
            logger.error(f"Database ping failed: {e}")
            return False


# Global database manager instance
db_manager = DatabaseManager()


@contextmanager
def get_database() -> Generator[Database, None, None]:
    """
    Context manager for database connections.

    Usage:
        with get_database() as db:
            # perform database operations
            collection = db['my_collection']
    """
    try:
        yield db_manager.db
    finally:
        pass
