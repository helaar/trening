"""Strava OAuth token management and refresh."""

import os
import requests
from datetime import datetime, timedelta
from pathlib import Path


class StravaTokenManager:
    """Manages Strava OAuth tokens with automatic refresh capability."""

    STRAVA_TOKEN_URL = "https://www.strava.com/oauth/token"

    def __init__(self, env_file: Path | None = None):
        """Initialize token manager with values from environment variables.
        
        Args:
            env_file: Optional path to .env file to update. If None, uses ../analyzer/.env
        """
        self.env_file = env_file or Path(__file__).parent.parent / ".env"
        self.client_id = os.getenv("STRAVA_CLIENT_ID")
        self.client_secret = os.getenv("STRAVA_CLIENT_SECRET")
        self.refresh_token = os.getenv("STRAVA_REFRESH_TOKEN")
        self.access_token = os.getenv("STRAVA_ACCESS_TOKEN")
        self.access_token_expiry = self._parse_expiry("STRAVA_ACCESS_TOKEN_EXPIRY")
        self.refresh_token_expiry = self._parse_expiry("STRAVA_REFRESH_TOKEN_EXPIRY")

    def _parse_expiry(self, env_var: str) -> datetime | None:
        """Parse token expiry from environment variable."""
        expiry_str = os.getenv(env_var)
        if expiry_str:
            try:
                return datetime.fromisoformat(expiry_str)
            except ValueError:
                return None
        return None

    def is_access_token_expired(self) -> bool:
        """Check if access token has expired or will expire soon (within 5 minutes)."""
        if not self.access_token_expiry:
            # If no expiry set, assume expired to trigger refresh
            return True
        # Consider expired if less than 5 minutes remaining
        return datetime.utcnow() >= (self.access_token_expiry - timedelta(minutes=5))

    def is_refresh_token_expired(self) -> bool:
        """Check if refresh token has expired."""
        if not self.refresh_token_expiry:
            # If no expiry set, assume not expired (Strava refresh tokens last ~6 months)
            return False
        return datetime.utcnow() >= self.refresh_token_expiry

    def refresh_access_token(self) -> bool:
        """
        Refresh the access token using the refresh token.

        Returns:
            bool: True if refresh was successful, False otherwise.
        """
        if not all([self.client_id, self.client_secret, self.refresh_token]):
            print("Missing required credentials for token refresh")
            return False

        if self.is_refresh_token_expired():
            print("Refresh token has expired - manual re-authentication required")
            return False

        try:
            response = requests.post(
                self.STRAVA_TOKEN_URL,
                data={
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "grant_type": "refresh_token",
                    "refresh_token": self.refresh_token,
                },
            )
            response.raise_for_status()

            data = response.json()
            self.access_token = data.get("access_token")
            # Strava may return a new refresh token
            new_refresh_token = data.get("refresh_token")
            if new_refresh_token:
                self.refresh_token = new_refresh_token

            # Calculate expiry times
            expires_in = data.get("expires_in", 21600)  # Default 6 hours
            self.access_token_expiry = datetime.utcnow() + timedelta(seconds=expires_in)
            
            # Refresh tokens typically expire after 6 months if not used
            if new_refresh_token:
                self.refresh_token_expiry = datetime.utcnow() + timedelta(days=180)

            # Update environment file
            self._update_env_file()

            print(f"Token refreshed successfully. New expiry: {self.access_token_expiry.isoformat()}")
            return True

        except requests.exceptions.RequestException as e:
            print(f"Failed to refresh token: {e}")
            return False

    def get_valid_access_token(self) -> str | None:
        """
        Get a valid access token, refreshing if necessary.

        Returns:
            str | None: Valid access token or None if refresh fails.
        """
        if self.is_access_token_expired():
            print("Access token expired or expiring soon, refreshing...")
            if not self.refresh_access_token():
                return None

        return self.access_token

    def _update_env_file(self) -> None:
        """Update .env file with new token values and expiry times."""
        if not self.env_file.exists():
            print(f"Warning: .env file not found at {self.env_file}")
            return

        content = self.env_file.read_text()
        lines = content.splitlines(keepends=True)

        updates = {
            "STRAVA_ACCESS_TOKEN": self.access_token,
            "STRAVA_REFRESH_TOKEN": self.refresh_token,
            "STRAVA_ACCESS_TOKEN_EXPIRY": self.access_token_expiry.isoformat() if self.access_token_expiry else "",
            "STRAVA_REFRESH_TOKEN_EXPIRY": self.refresh_token_expiry.isoformat() if self.refresh_token_expiry else "",
        }

        new_lines = []
        updated_keys = set()

        for line in lines:
            key_found = False
            for key, value in updates.items():
                if line.startswith(f"{key}="):
                    new_lines.append(f"{key}={value}\n")
                    updated_keys.add(key)
                    key_found = True
                    break

            if not key_found:
                new_lines.append(line)

        # Add any missing keys at the end
        for key, value in updates.items():
            if key not in updated_keys and value:
                new_lines.append(f"{key}={value}\n")

        self.env_file.write_text("".join(new_lines))
        print(f"Updated token values in {self.env_file}")

    @staticmethod
    def ensure_valid_token(env_file: Path | None = None) -> bool:
        """
        Convenience method to ensure a valid Strava access token.
        Checks expiry and refreshes if needed.
        
        Args:
            env_file: Optional path to .env file. If None, uses default location.
            
        Returns:
            bool: True if valid token is available, False otherwise.
        """
        try:
            manager = StravaTokenManager(env_file=env_file)
            if manager.is_access_token_expired():
                print("Strava access token expired, attempting refresh...")
                return manager.refresh_access_token()
            print("Strava access token is valid")
            return True
        except Exception as e:
            print(f"Error checking/refreshing token: {e}")
            return False