import logging
from datetime import datetime, timezone
from typing import Any

from pymongo.asynchronous.database import AsyncDatabase

from models.task import Task, TaskStatus

logger = logging.getLogger(__name__)


class TaskRepository:
    """Repository for task database operations."""
    
    def __init__(self, db: AsyncDatabase):
        self.db = db
        self.collection = db["tasks"]
    
    async def create_task(self, task: Task) -> Task:
        """Create a new task in the database."""
        task_dict = task.model_dump(mode="json")
        await self.collection.insert_one(task_dict)
        logger.info(f"Created task {task.task_id} for athlete {task.athlete_id}")
        return task
    
    async def get_task(self, task_id: str) -> Task | None:
        """Get a task by ID."""
        task_dict = await self.collection.find_one({"task_id": task_id})
        if task_dict:
            task_dict.pop("_id", None)
            return Task(**task_dict)
        return None
    
    async def get_tasks_by_athlete(
        self,
        athlete_id: int,
        limit: int = 50,
        status: TaskStatus | None = None
    ) -> list[Task]:
        """Get tasks for a specific athlete."""
        query: dict[str, Any] = {"athlete_id": athlete_id}
        if status:
            query["status"] = status.value
        
        cursor = self.collection.find(query).sort("created_at", -1).limit(limit)
        tasks = []
        async for task_dict in cursor:
            task_dict.pop("_id", None)
            tasks.append(Task(**task_dict))
        return tasks
    
    async def update_task_status(
        self, 
        task_id: str, 
        status: TaskStatus,
        error: str | None = None
    ) -> bool:
        """Update task status."""
        update_data: dict[str, Any] = {
            "status": status.value,
            "updated_at": datetime.now(timezone.utc)
        }
        
        if status == TaskStatus.RUNNING and not error:
            update_data["started_at"] = datetime.now(timezone.utc)
        elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            update_data["completed_at"] = datetime.now(timezone.utc)
        
        if error:
            update_data["error"] = error
        
        result = await self.collection.update_one(
            {"task_id": task_id},
            {"$set": update_data}
        )
        return result.modified_count > 0
    
    async def update_task_progress(self, task_id: str, progress: float) -> bool:
        """Update task progress."""
        result = await self.collection.update_one(
            {"task_id": task_id},
            {"$set": {"progress": progress}}
        )
        return result.modified_count > 0
    
    async def update_task_result(self, task_id: str, result: dict[str, Any]) -> bool:
        """Update task result."""
        update_result = await self.collection.update_one(
            {"task_id": task_id},
            {"$set": {"result": result}}
        )
        return update_result.modified_count > 0
    
    async def delete_task(self, task_id: str) -> bool:
        """Delete a task."""
        result = await self.collection.delete_one({"task_id": task_id})
        return result.deleted_count > 0
