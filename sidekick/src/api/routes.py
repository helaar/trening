from fastapi import APIRouter
from api.athlete_routes import router as athlete_router
from api.daily_entry_routes import router as daily_entry_router
from api.plan_routes import router as plan_router
from api.workout_routes import router as workout_router
from api.task_routes import router as task_router

router = APIRouter()

# Include athlete routes
router.include_router(athlete_router, prefix="/api/v1")

# Include workout routes
router.include_router(workout_router, prefix="/api/v1")

# Include task routes
router.include_router(task_router, prefix="/api/v1")

# Include daily entry routes
router.include_router(daily_entry_router, prefix="/api/v1")

# Include plan routes
router.include_router(plan_router, prefix="/api/v1")


@router.get("/api/v1/status")
async def api_status():
    """API status endpoint."""
    return {"status": "operational", "version": "v1"}
