from fastapi import APIRouter
from api.workout_routes import router as workout_router

router = APIRouter()

# Include workout routes
router.include_router(workout_router, prefix="/api/v1")


@router.get("/api/v1/status")
async def api_status():
    """API status endpoint."""
    return {"status": "operational", "version": "v1"}
