"""
Application configuration via environment variables and .env files.

Uses pydantic-settings so every value can be overridden at runtime:
    export AIEAR_WHISPER_MODEL=medium
    export AIEAR_API_PORT=9090
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global configuration for the AI Ear system."""

    model_config = SettingsConfigDict(
        env_prefix="AIEAR_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ API
    api_host: str = Field(default="0.0.0.0", description="API server bind host")
    api_port: int = Field(default=8080, ge=1, le=65535, description="API server bind port")
    api_cors_origins: list[str] = Field(
        default=["*"], description="Allowed CORS origins"
    )

    # ---------------------------------------------------------------- Audio
    audio_sample_rate: int = Field(
        default=16000, ge=8000, le=96000, description="Capture sample rate (Hz)"
    )
    audio_chunk_duration_s: float = Field(
        default=2.0, gt=0.0, le=30.0, description="Processing window size (seconds)"
    )
    audio_channels: int = Field(default=1, ge=1, le=8, description="Number of input channels")
    audio_device_index: int | None = Field(
        default=None, description="sounddevice device index (None = system default)"
    )

    # --------------------------------------------------------------- Speech
    speech_enabled: bool = Field(default=True, description="Enable speech recognition")
    whisper_model: str = Field(
        default="base",
        description="Whisper model size: tiny, base, small, medium, large",
    )
    whisper_language: str | None = Field(
        default=None, description="Force language (None = auto-detect)"
    )
    whisper_device: str = Field(
        default="cpu", description="Compute device for Whisper: cpu, cuda, mps"
    )

    # -------------------------------------------------------------- Emotion
    emotion_model: str = Field(
        default="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        description="HuggingFace model ID for speech emotion recognition",
    )
    emotion_enabled: bool = Field(default=True)

    # --------------------------------------------------------- Environment
    environment_enabled: bool = Field(default=True)
    environment_noise_gate_db: float = Field(
        default=-50.0, description="Below this level treat as silence (dBFS)"
    )

    # ---------------------------------------------------------------- Music
    music_enabled: bool = Field(default=True)

    # ---------------------------------------------------------------- Memory
    memory_max_events: int = Field(
        default=1000, ge=10, description="Maximum aural events to retain in memory"
    )
    memory_max_results: int = Field(
        default=500, ge=10, description="Maximum analysis results to retain in memory"
    )
    memory_context_window_s: float = Field(
        default=60.0, gt=0.0, description="Rolling context window size (seconds)"
    )

    # --------------------------------------------------------------- Logging
    log_level: str = Field(default="INFO", description="Log level: DEBUG, INFO, WARNING, ERROR")
    log_json: bool = Field(default=False, description="Emit structured JSON logs")
