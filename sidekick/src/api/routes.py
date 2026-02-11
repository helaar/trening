from fastapi import APIRouter
from api.workout_routes import router as workout_router
from api.task_routes import router as task_router

router = APIRouter()

# Include workout routes
router.include_router(workout_router, prefix="/api/v1")

# Include task routes
router.include_router(task_router, prefix="/api/v1")


@router.get("/api/v1/status")
async def api_status():
    """API status endpoint."""
    return {"status": "operational", "version": "v1"}
