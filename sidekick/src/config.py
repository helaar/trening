from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    mongodb_url: str = "mongodb://admin:admin_password@localhost:27010"
    mongodb_database: str = "sidekick"
    qdrant_url: str = "http://localhost:6333"
    environment: str = "development"
    log_level: str = "INFO"
    
    # CORS settings
    cors_origins: str = "*"  # Comma-separated list of allowed origins
    cors_allow_credentials: bool = True
    cors_allow_methods: str = "*"  # Comma-separated list or "*"
    cors_allow_headers: str = "*"  # Comma-separated list or "*"

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
