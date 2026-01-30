from fastapi import APIRouter

router = APIRouter()


@router.get("/api/v1/status")
async def api_status():
    """API status endpoint."""
    return {"status": "operational", "version": "v1"}
