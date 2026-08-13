"""Centralized application configuration.

This module defines a single, immutable source of truth for all runtime
configuration values used across the ActumAI Extract platform. Configuration
is sourced from environment variables (and an optional ``.env`` file) via
``pydantic-settings``, giving us type-safe, validated configuration objects
instead of scattered ``os.environ.get(...)`` calls.

The settings object is exposed as a cached singleton (``get_settings``) so
that it is parsed exactly once per process, which matters both for
performance and for guaranteeing configuration consistency across the
lifetime of a running worker.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly typed application settings.

    Attributes:
        app_name: Human readable service name, surfaced in logs and the
            OpenAPI schema.
        environment: Deployment environment, used to toggle behavior such as
            debug tracebacks and log verbosity.
        api_v1_prefix: URL prefix under which all v1 routes are mounted.

        openai_api_key: Secret API key used to authenticate against the LLM
            provider. Never logged, never serialized in ``repr``.
        openai_base_url: Base URL of the OpenAI-compatible endpoint. This is
            intentionally overridable so the same codebase can target Azure
            OpenAI, self-hosted vLLM/TGI gateways, or OpenRouter without code
            changes.
        extraction_model: Model identifier used for structured extraction
            calls.
        extraction_temperature: Sampling temperature for extraction calls.
            Kept low by default because deterministic, literal extraction is
            preferred over creative paraphrasing.
        extraction_max_tokens: Hard ceiling on completion tokens per
            extraction call, used as a cost and latency guardrail.
        extraction_max_retries: Maximum number of self-healing retry attempts
            the extraction engine will perform when the LLM returns a
            response that fails Pydantic validation.
        llm_request_timeout_seconds: Per-request timeout applied to the
            underlying HTTP client used by the OpenAI SDK.

        max_upload_size_bytes: Maximum accepted size, in bytes, for uploaded
            PDF documents. Requests exceeding this are rejected with 413
            before any parsing work is attempted.
        max_pdf_pages: Maximum number of pages the parser will process from a
            single PDF, protecting the service from pathological documents.

        log_level: Root logging level for the structured JSON logger.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Service metadata ---
    app_name: str = Field(default="ActumAI Extract")
    environment: Literal["local", "staging", "production"] = Field(default="local")
    api_v1_prefix: str = Field(default="/api/v1")

    # --- LLM provider configuration ---
    openai_api_key: SecretStr = Field(
        default=SecretStr("sk-placeholder-override-in-env"),
        description="API key for the OpenAI-compatible chat completions endpoint.",
    )
    openai_base_url: str | None = Field(
        default=None,
        description="Optional base URL override for OpenAI-compatible providers.",
    )
    extraction_model: str = Field(default="gpt-4o-2024-08-06")
    extraction_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    extraction_max_tokens: int = Field(default=4096, gt=0, le=32_000)
    extraction_max_retries: int = Field(default=3, ge=0, le=10)
    llm_request_timeout_seconds: float = Field(default=60.0, gt=0.0, le=600.0)

    # --- Document ingestion limits ---
    max_upload_size_bytes: int = Field(default=15 * 1024 * 1024)  # 15 MB
    max_pdf_pages: int = Field(default=100, gt=0, le=2000)
    max_text_input_chars: int = Field(default=200_000, gt=0)

    # --- Observability ---
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")

    @field_validator("openai_base_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str | None) -> str | None:
        """Normalizes the base URL by removing a trailing slash if present.

        Args:
            value: The raw base URL supplied via environment configuration.

        Returns:
            The normalized base URL, or ``None`` if not configured.
        """
        if value is None:
            return None
        return value.rstrip("/")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Returns the process-wide cached ``Settings`` singleton.

    Using ``lru_cache`` ensures the ``.env`` file and environment are parsed
    exactly once, and that every module importing this function shares the
    same configuration instance — critical for consistent behavior under
    concurrent request handling in an async ASGI worker.

    Returns:
        The cached, validated application settings instance.
    """
    return Settings()
