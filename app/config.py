"""Application configuration loaded from environment variables / `.env`."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- General -----------------------------------------------------------------
    app_name: str = "Intelligent Document Processing System"
    app_version: str = "1.0.0"
    environment: str = "development"
    log_level: str = "INFO"

    # --- Storage -----------------------------------------------------------------
    data_dir: Path = Path("data")
    database_url: str | None = None
    max_upload_mb: int = 25

    # --- LLM (Gemini) ------------------------------------------------------------
    gemini_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY")
    )
    gemini_model: str = "gemini-3.8-flash"
    llm_enabled: bool = True
    llm_temperature: float = 0.0
    llm_max_retries: int = 3
    llm_timeout_s: int = 120
    llm_send_files: bool = True  # attach the original PDF/image to the prompt (multimodal)
    llm_max_text_chars: int = 60_000

    # --- OCR ---------------------------------------------------------------------
    ocr_enabled: bool = True
    tesseract_cmd: str | None = None
    ocr_languages: str = "eng"
    ocr_dpi: int = 300
    min_text_chars_per_page: int = 40  # below this a PDF page is treated as scanned
    max_pages: int = 50
    max_preview_pages: int = 10

    # --- Confidence / review -----------------------------------------------------
    confidence_threshold: float = 0.80  # document-level auto-approve threshold
    field_confidence_threshold: float = 0.60  # any required field below => review
    classification_threshold: float = 0.60

    # --- Processing --------------------------------------------------------------
    batch_concurrency: int = 4

    # --- API ---------------------------------------------------------------------
    api_key: str | None = None  # if set, clients must send `X-API-Key`
    cors_origins: list[str] = ["*"]

    @field_validator("gemini_api_key", "api_key", "tesseract_cmd", "database_url", mode="before")
    @classmethod
    def _empty_to_none(cls, v):
        return v or None

    @property
    def llm_available(self) -> bool:
        return bool(self.llm_enabled and self.gemini_api_key)

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def preview_dir(self) -> Path:
        return self.data_dir / "previews"

    @property
    def resolved_database_url(self) -> str:
        return self.database_url or f"sqlite:///{(self.data_dir / 'idp.db').as_posix()}"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.upload_dir, self.preview_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
