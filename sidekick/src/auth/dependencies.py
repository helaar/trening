from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pymongo.asynchronous.database import AsyncDatabase

from auth.oauth import StravaOAuthService
from config import settings
from database.athlete_repository import AthleteRepository
from database.mongodb import get_db
from models.athlete import Athlete
from clients.strava.client import StravaClient


security = HTTPBearer(auto_error=False)


async def get_athlete_repository(db: AsyncDatabase = Depends(get_db)) -> AthleteRepository:
    """Dependency to get athlete repository."""
    return AthleteRepository(db)


async def get_oauth_service(
    athlete_repo: AthleteRepository = Depends(get_athlete_repository)
) -> StravaOAuthService:
    """Dependency to get OAuth service."""
    return StravaOAuthService(athlete_repo)


async def get_current_athlete_id(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
    oauth_service: StravaOAuthService = Depends(get_oauth_service)
) -> int:
    """
    Dependency to get current authenticated athlete ID from JWT token.
    
    In dev mode (when DEV_MODE=true and DEV_ATHLETE_ID is set),
    bypasses authentication and returns the configured athlete ID.
    """
    # Dev mode bypass
    if settings.dev_mode:
        if settings.dev_athlete_id is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="DEV_MODE is enabled but DEV_ATHLETE_ID is not configured"
            )
        return settings.dev_athlete_id
    
    # Production authentication
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = credentials.credentials
    
    try:
        athlete_id = oauth_service.verify_session_token(token)
        return athlete_id
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_athlete(
    athlete_id: int = Depends(get_current_athlete_id),
    athlete_repo: AthleteRepository = Depends(get_athlete_repository)
) -> Athlete:
    """Dependency to get current authenticated athlete."""
    athlete = await athlete_repo.get_athlete(athlete_id)
    
    if not athlete:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Athlete not found"
        )
    
    return athlete


async def get_strava_client(
    athlete_id: int = Depends(get_current_athlete_id),
    athlete_repo: AthleteRepository = Depends(get_athlete_repository)
) -> StravaClient:
    """
    Dependency to get an authenticated Strava client for the current athlete.
    
    Automatically handles token refresh if needed.
    """
    try:
        return await StravaClient.from_athlete_id(athlete_id, athlete_repo)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Failed to authenticate with Strava: {str(e)}"
        )
