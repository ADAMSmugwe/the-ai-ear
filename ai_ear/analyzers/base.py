"""
Abstract base class for all AI Ear analysers.

Every analyser follows the same lifecycle contract:
1. ``load()``   – initialise models / download weights (called once at startup)
2. ``analyse()`` – process a single :class:`AudioChunk`, return a typed result
3. ``unload()`` – release GPU/CPU memory (called once at shutdown)

Concrete result types carry the ``confidence`` attribute so the pipeline
can compute an overall confidence score for each fused :class:`AnalysisResult`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from ai_ear.core.models import (
    AudioChunk,
    EmotionProfile,
    EnvironmentSnapshot,
    MusicProfile,
    SpeechSegment,
)

# ---------------------------------------------------------------------------
# Typed partial results
# ---------------------------------------------------------------------------

class SpeechResult(BaseModel):
    segment: SpeechSegment
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class EmotionResult(BaseModel):
    profile: EmotionProfile
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class EnvironmentResult(BaseModel):
    snapshot: EnvironmentSnapshot
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class MusicResult(BaseModel):
    profile: MusicProfile
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Base analyser
# ---------------------------------------------------------------------------

class BaseAnalyzer(ABC):
    """
    Abstract base for a single-modality audio analyser.

    Sub-classes must implement :meth:`load` and :meth:`analyse`.
    """

    #: Human-readable name used in logs and API responses.
    name: str = "unnamed"

    async def load(self) -> None:
        """Load model weights / resources.  Override as needed."""

    async def unload(self) -> None:
        """Release resources.  Override as needed."""

    @abstractmethod
    async def analyse(
        self, chunk: AudioChunk
    ) -> SpeechResult | EmotionResult | EnvironmentResult | MusicResult:
        """
        Analyse a single audio chunk and return a typed partial result.

        Implementations MUST NOT modify the chunk in-place.
        """
