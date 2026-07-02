from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    mongodb_url: str = "mongodb://admin:admin_password@localhost:27010"
    mongodb_database: str = "sidekick"
    qdrant_url: str = "http://localhost:6333"
    environment: str = "development"
    log_level: str = "INFO"
    dev_mode: bool = False  # Skip authentication when True
    dev_athlete_id: int | None = None  # Athlete ID to use in dev mode

    # Strava OAuth settings
    strava_client_id: str
    strava_client_secret: str
    strava_redirect_uri: str = "http://localhost:5175/auth/strava/callback"

    # JWT settings for session management
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 43200  # 30 days

    # CORS settings
    cors_origins: str = "*"  # Comma-separated list of allowed origins
    cors_allow_credentials: bool = True
    cors_allow_methods: str = "*"  # Comma-separated list or "*"
    cors_allow_headers: str = "*"  # Comma-separated list or "*"

    # LLM settings for CrewAI
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    llm_model: str = "anthropic/claude-haiku-4-5-20251001"
    llm_max_tokens: int = 8192  # Output cap; prevents truncated structured (JSON) outputs

    # Memory consolidation — ISO 8601 duration; set PT0S to disable the guard
    consolidation_min_age: str = "P1W"

    # Per-session power histogram bucket width (watts). FTP-agnostic; the weekly
    # polarized assessment derives intensity bands from these buckets.
    power_histogram_bucket_watts: float = 5.0

    # Polarized (80/20) weekly assessment tunables — deterministic verdict thresholds.
    # Within the gray zone (Z3), only time in the upper part of the zone counts as drift;
    # time in the lower `depth_frac` of the zone is treated as effectively easy
    # (0.25 = tolerate the lower quarter). The weekly cutoffs below decide the verdict, so
    # small amounts of drift never trip it and no minimum-duration knob is needed.
    polarized_gray_zone_depth_frac: float = 0.25
    # Weekly status cutoffs. The verdict keys on whether easy volume holds its target:
    # gray-zone time only matters when it eats into easy. Easy at/above target (or with
    # negligible moderate) reads as polarized; below target with moderate present is drift,
    # graded by how far easy has slipped.
    polarized_low_target_pct: float = 80.0  # easy >= this -> polarized
    polarized_gray_zone_low_pct: float = 75.0  # easy below this (w/ moderate) -> gray_zone_week
    polarized_min_moderate_pct: float = 5.0  # moderate <= this -> easy dip not blamed on gray zone
    # Sufficiency floor: below either and the week is "insufficient_data".
    polarized_min_classified_minutes: float = 60.0
    polarized_min_training_days: int = 2

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins into a list."""
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def cors_methods_list(self) -> list[str]:
        """Parse CORS methods into a list."""
        if self.cors_allow_methods == "*":
            return ["*"]
        return [method.strip() for method in self.cors_allow_methods.split(",") if method.strip()]

    @property
    def cors_headers_list(self) -> list[str]:
        """Parse CORS headers into a list."""
        if self.cors_allow_headers == "*":
            return ["*"]
        return [header.strip() for header in self.cors_allow_headers.split(",") if header.strip()]


settings = Settings()
