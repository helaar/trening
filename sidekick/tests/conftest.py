import os

# config.Settings is instantiated at import time and requires these; tests never
# reach Strava or issue tokens, so placeholders are enough.
os.environ.setdefault("STRAVA_CLIENT_ID", "test")
os.environ.setdefault("STRAVA_CLIENT_SECRET", "test")
os.environ.setdefault("JWT_SECRET_KEY", "test")
