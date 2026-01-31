# Strava OAuth Authentication

## Quick Start

1. Get Strava credentials from https://www.strava.com/settings/api
2. Add to `.env`:
   ```bash
   STRAVA_CLIENT_ID=your_client_id
   STRAVA_CLIENT_SECRET=your_client_secret
   JWT_SECRET_KEY=generate_with_secrets.token_urlsafe(32)
   ```

3. Authorize your athlete:
   ```bash
   # Visit in browser
   http://localhost:8000/auth/strava/authorize
   
   # Save the JWT token from response
   export TOKEN="eyJ0eXAiOiJKV1Qi..."
   ```

4. Use the token for API requests:
   ```bash
   curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/auth/me
   ```

## Architecture

- **Models**: [`Athlete`](../src/models/athlete.py) and [`StravaTokens`](../src/models/athlete.py) stored separately in MongoDB
- **OAuth Service**: [`StravaOAuthService`](../src/auth/oauth.py) handles token exchange and refresh
- **Auth Routes**: [`/auth/strava/*`](../src/api/auth_routes.py) endpoints for OAuth flow
- **Dependencies**: [`get_strava_client()`](../src/auth/dependencies.py) provides authenticated client

## OAuth Flow

```
GET /auth/strava/authorize 
  → Redirects to Strava
  → User approves
  → Callback with code
  → Exchange for tokens
  → Store in MongoDB
  → Return JWT session token
```

## Token Management

- Access tokens auto-refresh when expired (6 hour lifetime)
- JWT session tokens expire after 30 days (configurable)
- Tokens stored separately from athlete profiles for security

## Usage in Routes

```python
from fastapi import Depends
from auth.dependencies import get_strava_client
from clients.strava.client import StravaClient

@router.get("/my-activities")
async def get_activities(client: StravaClient = Depends(get_strava_client)):
    return client.get_activities_for_date(date.today())
```

## Endpoints

- `GET /auth/strava/authorize` - Start OAuth flow
- `GET /auth/strava/callback` - OAuth callback (automatic)
- `GET /auth/me` - Current athlete info
- `GET /auth/strava/status` - Check connection
- `POST /auth/strava/disconnect` - Remove tokens
