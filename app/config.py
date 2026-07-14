from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime configuration, overridable via environment variables or a .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="LANGID_",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    app_name: str = "Language Identification API"
    api_v1_prefix: str = "/api/v1"

    model_path: Path = PROJECT_ROOT / "models" / "language_classifier.h5"
    model_input_size: int = 192

    max_upload_size_mb: int = 10
    allowed_audio_extensions: tuple[str, ...] = (".wav", ".mp3", ".flac", ".ogg", ".m4a")

    cors_allow_origins: tuple[str, ...] = ("*",)

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
