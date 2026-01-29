from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    mongodb_url: str = "mongodb://admin:admin_password@localhost:27010"
    mongodb_database: str = "sidekick"
    qdrant_url: str = "http://localhost:6333"
    environment: str = "development"
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


settings = Settings()
