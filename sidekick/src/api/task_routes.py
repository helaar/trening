import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pymongo.asynchronous.database import AsyncDatabase

from auth.dependencies import get_current_athlete_id
from database.mongodb import get_db
from database.task_repository import TaskRepository
from models.task import TaskCreateRequest, TaskResponse, TaskStatus
from services.task_processor import TaskProcessor, create_task

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tasks", tags=["tasks"])


def get_task_repository(db: Annotated[AsyncDatabase, Depends(get_db)]) -> TaskRepository:
    """Dependency to get task repository."""
    return TaskRepository(db)


def get_task_processor(
    task_repo: Annotated[TaskRepository, Depends(get_task_repository)]
) -> TaskProcessor:
    """Dependency to get task processor."""
    return TaskProcessor(task_repo)


@router.post("", response_model=TaskResponse, status_code=status.HTTP_201_CREATED)
async def create_new_task(
    request: TaskCreateRequest,
    athlete_id: Annotated[int, Depends(get_current_athlete_id)],
    task_repo: Annotated[TaskRepository, Depends(get_task_repository)],
    task_processor: Annotated[TaskProcessor, Depends(get_task_processor)]
):
    """
    Create a new asynchronous task.
    
    This endpoint creates a task and starts processing it in the background.
    The client receives a task_id that can be used to query the task status.
    """
    try:
        task = await create_task(
            task_repo=task_repo,
            athlete_id=athlete_id,
            task_type=request.task_type,
            parameters=request.parameters
        )
        
        task_processor.start_task_in_background(task.task_id)
        
        return TaskResponse(
            task_id=task.task_id,
            status=task.status,
            progress=task.progress,
            result=task.result,
            error=task.error,
            created_at=task.created_at,
            started_at=task.started_at,
            completed_at=task.completed_at,
            duration_seconds=task.duration_seconds
        )
    except Exception as e:
        logger.error(f"Failed to create task: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create task: {str(e)}"
        )


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task_status(
    task_id: str,
    athlete_id: Annotated[int, Depends(get_current_athlete_id)],
    task_repo: Annotated[TaskRepository, Depends(get_task_repository)]
):
    """
    Get the status and result of a specific task.
    
    Returns current task status, progress, and result if completed.
    """
    task = await task_repo.get_task(task_id)
    
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )
    
    if task.athlete_id != athlete_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this task"
        )
    
    return TaskResponse(
        task_id=task.task_id,
        status=task.status,
        progress=task.progress,
        result=task.result,
        error=task.error,
        created_at=task.created_at,
        started_at=task.started_at,
        completed_at=task.completed_at,
        duration_seconds=task.duration_seconds
    )


@router.get("", response_model=list[TaskResponse])
async def list_tasks(
    athlete_id: Annotated[int, Depends(get_current_athlete_id)],
    task_repo: Annotated[TaskRepository, Depends(get_task_repository)],
    limit: int = 50,
    status_filter: TaskStatus | None = None
):
    """
    List all tasks for the authenticated athlete.
    
    Optionally filter by status and limit the number of results.
    """
    tasks = await task_repo.get_tasks_by_athlete(
        athlete_id=athlete_id,
        limit=limit,
        status=status_filter
    )
    
    return [
        TaskResponse(
            task_id=task.task_id,
            status=task.status,
            progress=task.progress,
            result=task.result,
            error=task.error,
            created_at=task.created_at,
            started_at=task.started_at,
            completed_at=task.completed_at,
            duration_seconds=task.duration_seconds
        )
        for task in tasks
    ]


@router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_task(
    task_id: str,
    athlete_id: Annotated[int, Depends(get_current_athlete_id)],
    task_repo: Annotated[TaskRepository, Depends(get_task_repository)]
):
    """
    Delete a task.
    
    Only the task owner can delete their tasks.
    """
    task = await task_repo.get_task(task_id)
    
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )
    
    if task.athlete_id != athlete_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this task"
        )
    
    success = await task_repo.delete_task(task_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete task"
        )
