from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import RedirectResponse, JSONResponse

from auth.oauth import StravaOAuthService
from auth.dependencies import get_oauth_service, get_current_athlete
from database.athlete_repository import AthleteRepository
from models.athlete import Athlete


router = APIRouter(prefix="/auth", tags=["authentication"])


@router.get("/strava/authorize")
async def authorize_strava():
    """
    Redirect user to Strava authorization page.
    
    This initiates the OAuth flow by redirecting to Strava's authorization page.
    After authorization, Strava will redirect back to /auth/strava/callback.
    """
    # Create a temporary OAuth service just for generating the URL (no DB needed)
    oauth_service = StravaOAuthService(athlete_repo=None)  # type: ignore
    authorization_url = oauth_service.get_authorization_url()
    return RedirectResponse(url=authorization_url)


@router.get("/strava/callback")
async def strava_callback(
    code: str = Query(..., description="Authorization code from Strava"),
    scope: str | None = Query(None, description="Granted scopes"),
    oauth_service: StravaOAuthService = Depends(get_oauth_service)
):
    """
    Handle OAuth callback from Strava.
    
    Exchanges the authorization code for access and refresh tokens,
    stores athlete information, and returns a session JWT token.
    """
    try:
        athlete, tokens = await oauth_service.exchange_code_for_tokens(code)
        session_token = oauth_service.create_session_token(athlete.athlete_id)
        
        return JSONResponse(
            content={
                "message": "Successfully authenticated with Strava",
                "athlete": {
                    "athlete_id": athlete.athlete_id,
                    "username": athlete.username,
                    "firstname": athlete.firstname,
                    "lastname": athlete.lastname,
                    "profile_picture": athlete.profile_picture
                },
                "access_token": session_token,
                "token_type": "bearer"
            },
            status_code=status.HTTP_200_OK
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to exchange authorization code: {str(e)}"
        )


@router.get("/me")
async def get_current_user(
    athlete: Athlete = Depends(get_current_athlete)
):
    """
    Get current authenticated athlete information.
    
    Requires a valid JWT token in the Authorization header.
    """
    return {
        "athlete_id": athlete.athlete_id,
        "username": athlete.username,
        "firstname": athlete.firstname,
        "lastname": athlete.lastname,
        "profile_picture": athlete.profile_picture,
        "created_at": athlete.created_at,
        "updated_at": athlete.updated_at
    }


@router.post("/strava/disconnect")
async def disconnect_strava(
    athlete: Athlete = Depends(get_current_athlete),
    oauth_service: StravaOAuthService = Depends(get_oauth_service)
):
    """
    Disconnect Strava account by deleting stored tokens.
    
    Requires authentication. This does not revoke the token on Strava's side,
    but removes it from our database.
    """
    success = await oauth_service.athlete_repo.delete_tokens(athlete.athlete_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No Strava connection found"
        )
    
    return {"message": "Successfully disconnected from Strava"}


@router.get("/strava/status")
async def check_strava_connection(
    athlete: Athlete = Depends(get_current_athlete),
    oauth_service: StravaOAuthService = Depends(get_oauth_service)
):
    """
    Check if current athlete has a valid Strava connection.
    
    Returns connection status and token expiration information.
    """
    tokens = await oauth_service.athlete_repo.get_tokens(athlete.athlete_id)
    
    if not tokens:
        return {
            "connected": False,
            "message": "No Strava connection found"
        }
    
    import time
    current_time = int(time.time())
    is_expired = tokens.expires_at <= current_time
    
    return {
        "connected": True,
        "expires_at": tokens.expires_at,
        "is_expired": is_expired,
        "message": "Strava connection active" if not is_expired else "Token expired but can be refreshed"
    }
