"""
Shared Pydantic data models for the AI Ear pipeline.

All data flowing through the system is strongly typed using these models,
making cross-component communication explicit and serialisable.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Low-level audio chunk
# ---------------------------------------------------------------------------

class AudioChunk(BaseModel):
    """A fixed-size window of raw PCM audio samples."""

    model_config = {"arbitrary_types_allowed": True}

    samples: np.ndarray = Field(..., description="PCM float32 samples, shape (N,) or (N, C)")
    sample_rate: int = Field(..., ge=8000, le=192000, description="Samples per second")
    timestamp: float = Field(default_factory=time.time, description="Wall-clock capture time (UTC epoch)")
    source_id: str = Field(default="default", description="Logical source identifier")
    duration_s: float = Field(default=0.0, description="Duration in seconds (computed)")

    @model_validator(mode="after")
    def _compute_duration(self) -> "AudioChunk":
        n = self.samples.shape[0]
        object.__setattr__(self, "duration_s", n / self.sample_rate)
        return self

    @field_validator("samples", mode="before")
    @classmethod
    def _coerce_samples(cls, v: Any) -> np.ndarray:
        arr = np.asarray(v, dtype=np.float32)
        if arr.ndim not in (1, 2):
            raise ValueError("samples must be 1-D (mono) or 2-D (multi-channel)")
        return arr


# ---------------------------------------------------------------------------
# Analyser result models
# ---------------------------------------------------------------------------

class SpeechSegment(BaseModel):
    """A transcribed speech segment with metadata."""

    text: str = Field(..., description="Transcribed text")
    language: str = Field(default="en", description="Detected language code")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    start_s: float = Field(default=0.0, ge=0.0, description="Segment start relative to chunk")
    end_s: float = Field(default=0.0, ge=0.0, description="Segment end relative to chunk")
    words: list[dict[str, Any]] = Field(
        default_factory=list, description="Word-level timestamps from Whisper"
    )


class EmotionLabel(str, Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    DISGUSTED = "disgusted"
    SURPRISED = "surprised"
    CALM = "calm"


class EmotionProfile(BaseModel):
    """Emotional valence inferred from vocal features."""

    dominant: EmotionLabel = Field(default=EmotionLabel.NEUTRAL)
    scores: dict[str, float] = Field(
        default_factory=dict, description="Per-label probability scores"
    )
    arousal: float = Field(default=0.5, ge=0.0, le=1.0, description="Arousal (0=calm, 1=excited)")
    valence: float = Field(default=0.5, ge=0.0, le=1.0, description="Valence (0=negative, 1=positive)")


class EnvironmentLabel(str, Enum):
    SPEECH = "speech"
    MUSIC = "music"
    SILENCE = "silence"
    CROWD = "crowd"
    NATURE = "nature"
    TRAFFIC = "traffic"
    OFFICE = "office"
    ALARM = "alarm"
    UNKNOWN = "unknown"


class EnvironmentSnapshot(BaseModel):
    """Classification of the acoustic environment."""

    dominant: EnvironmentLabel = Field(default=EnvironmentLabel.UNKNOWN)
    scores: dict[str, float] = Field(default_factory=dict)
    noise_floor_db: float = Field(default=-60.0, description="Estimated noise floor in dBFS")
    snr_db: float = Field(default=0.0, description="Estimated signal-to-noise ratio in dB")


class MusicProfile(BaseModel):
    """Characterisation of detected music content."""

    is_music: bool = Field(default=False)
    tempo_bpm: float | None = Field(default=None, ge=0.0, description="Estimated BPM")
    key: str | None = Field(default=None, description="Detected musical key (e.g. 'C major')")
    energy: float = Field(default=0.0, ge=0.0, le=1.0, description="Normalised spectral energy")
    spectral_centroid_hz: float | None = Field(default=None, description="Timbral brightness")
    genre_hints: list[str] = Field(default_factory=list, description="Top genre probability hints")


# ---------------------------------------------------------------------------
# Composite analysis result
# ---------------------------------------------------------------------------

class AnalysisResult(BaseModel):
    """The fully-analysed output for a single AudioChunk."""

    chunk_id: str = Field(..., description="Unique chunk identifier")
    source_id: str = Field(default="default")
    timestamp: float = Field(default_factory=time.time)
    duration_s: float = Field(default=0.0)

    speech: SpeechSegment | None = Field(default=None)
    emotion: EmotionProfile | None = Field(default=None)
    environment: EnvironmentSnapshot | None = Field(default=None)
    music: MusicProfile | None = Field(default=None)

    # Fused semantic tags generated by the pipeline
    semantic_tags: list[str] = Field(
        default_factory=list,
        description="High-level semantic descriptors derived by multi-modal fusion",
    )
    # Confidence in the overall analysis
    overall_confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    model_config = {"arbitrary_types_allowed": True, "serialize_by_alias": True}


# ---------------------------------------------------------------------------
# Aural event (higher-level memory unit)
# ---------------------------------------------------------------------------

class AuralEventType(str, Enum):
    SPEECH_STARTED = "speech_started"
    SPEECH_ENDED = "speech_ended"
    KEYWORD_DETECTED = "keyword_detected"
    EMOTION_SHIFT = "emotion_shift"
    ENVIRONMENT_CHANGE = "environment_change"
    MUSIC_STARTED = "music_started"
    MUSIC_ENDED = "music_ended"
    ALARM_DETECTED = "alarm_detected"
    SILENCE_STARTED = "silence_started"
    SILENCE_ENDED = "silence_ended"
    ANOMALY = "anomaly"


class AuralEvent(BaseModel):
    """A discrete, semantically meaningful event surfaced by the pipeline."""

    event_type: AuralEventType
    timestamp: float = Field(default_factory=time.time)
    source_id: str = Field(default="default")
    description: str = Field(default="")
    payload: dict[str, Any] = Field(default_factory=dict)
    severity: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="0 = informational, 1 = critical"
    )
