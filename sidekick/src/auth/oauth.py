import time
from datetime import datetime, timezone
from urllib.parse import urlencode
import requests
from jose import jwt, JWTError

from config import settings
from models.athlete import Athlete, StravaTokens
from database.athlete_repository import AthleteRepository


class StravaOAuthService:
    """Service for handling Strava OAuth flow and token management."""
    
    AUTHORIZE_URL = "https://www.strava.com/oauth/authorize"
    TOKEN_URL = "https://www.strava.com/oauth/token"
    
    def __init__(self, athlete_repo: AthleteRepository):
        self.athlete_repo = athlete_repo
    
    def get_authorization_url(self, state: str | None = None) -> str:
        """Generate Strava authorization URL."""
        params = {
            "client_id": settings.strava_client_id,
            "redirect_uri": settings.strava_redirect_uri,
            "response_type": "code",
            "scope": "read,activity:read_all,profile:read_all",
            "approval_prompt": "auto"
        }
        
        if state:
            params["state"] = state
        
        return f"{self.AUTHORIZE_URL}?{urlencode(params)}"
    
    async def exchange_code_for_tokens(self, code: str) -> tuple[Athlete, StravaTokens]:
        """Exchange authorization code for access and refresh tokens."""
        data = {
            "client_id": settings.strava_client_id,
            "client_secret": settings.strava_client_secret,
            "code": code,
            "grant_type": "authorization_code"
        }
        
        response = requests.post(self.TOKEN_URL, data=data)
        
        # If error, log the response details
        if not response.ok:
            try:
                error_detail = response.json()
                raise ValueError(f"Strava API error: {error_detail}")
            except Exception:
                response.raise_for_status()
        
        token_data = response.json()
        
        # Extract athlete info from response
        athlete_data = token_data.get("athlete", {})
        athlete = Athlete(
            athlete_id=athlete_data["id"],
            username=athlete_data.get("username"),
            firstname=athlete_data.get("firstname"),
            lastname=athlete_data.get("lastname"),
            profile_picture=athlete_data.get("profile")
        )
        
        # Create tokens object
        tokens = StravaTokens(
            athlete_id=athlete.athlete_id,
            access_token=token_data["access_token"],
            refresh_token=token_data["refresh_token"],
            expires_at=token_data["expires_at"],
            token_type=token_data.get("token_type", "Bearer")
        )
        
        # Save to database
        await self.athlete_repo.create_or_update_athlete(athlete)
        await self.athlete_repo.save_tokens(tokens)
        
        return athlete, tokens
    
    async def refresh_access_token(self, athlete_id: int) -> StravaTokens:
        """Refresh the access token using the refresh token."""
        tokens = await self.athlete_repo.get_tokens(athlete_id)
        if not tokens:
            raise ValueError(f"No tokens found for athlete {athlete_id}")
        
        data = {
            "client_id": settings.strava_client_id,
            "client_secret": settings.strava_client_secret,
            "refresh_token": tokens.refresh_token,
            "grant_type": "refresh_token"
        }
        
        response = requests.post(self.TOKEN_URL, data=data)
        response.raise_for_status()
        token_data = response.json()
        
        # Update tokens
        tokens.access_token = token_data["access_token"]
        tokens.refresh_token = token_data["refresh_token"]
        tokens.expires_at = token_data["expires_at"]
        
        # Save updated tokens
        await self.athlete_repo.save_tokens(tokens)
        
        return tokens
    
    async def get_valid_tokens(self, athlete_id: int) -> StravaTokens:
        """Get valid tokens, refreshing if necessary."""
        tokens = await self.athlete_repo.get_tokens(athlete_id)
        if not tokens:
            raise ValueError(f"No tokens found for athlete {athlete_id}")
        
        # Check if token is expired or will expire in the next 5 minutes
        current_time = int(time.time())
        if tokens.expires_at <= current_time + 300:
            tokens = await self.refresh_access_token(athlete_id)
        
        return tokens
    
    def create_session_token(self, athlete_id: int) -> str:
        """Create a JWT session token for the athlete."""
        payload = {
            "athlete_id": athlete_id,
            "exp": datetime.now(timezone.utc).timestamp() + (settings.jwt_access_token_expire_minutes * 60),
            "iat": datetime.now(timezone.utc).timestamp()
        }
        
        token = jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
        return token
    
    def verify_session_token(self, token: str) -> int:
        """Verify JWT session token and return athlete_id."""
        try:
            payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
            athlete_id = payload.get("athlete_id")
            if athlete_id is None:
                raise ValueError("Invalid token payload")
            return athlete_id
        except JWTError:
            raise ValueError("Invalid or expired session token")
