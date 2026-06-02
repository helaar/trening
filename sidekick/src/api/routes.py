from fastapi import APIRouter
from api.athlete_routes import router as athlete_router
from api.daily_analysis_routes import router as daily_analysis_router
from api.daily_entry_routes import router as daily_entry_router
from api.feed_routes import router as feed_router
from api.plan_routes import router as plan_router
from api.prompt_routes import router as prompt_router
from api.workout_routes import router as workout_router
from api.task_routes import router as task_router

router = APIRouter()

router.include_router(athlete_router, prefix="/api/v1")
router.include_router(workout_router, prefix="/api/v1")
router.include_router(task_router, prefix="/api/v1")
router.include_router(daily_entry_router, prefix="/api/v1")
router.include_router(daily_analysis_router, prefix="/api/v1")
router.include_router(plan_router, prefix="/api/v1")
router.include_router(feed_router, prefix="/api/v1")
router.include_router(prompt_router, prefix="/api/v1")


@router.get("/api/v1/status")
async def api_status():
    """API status endpoint."""
    return {"status": "operational", "version": "v1"}
