"""
The AI Ear — Enterprise-grade multi-modal AI audio listening system.

Hear beyond words: speech, emotion, environment, and music understanding
woven together into a continuous, memory-backed semantic stream.
"""

from ai_ear.core.listener import AudioListener
from ai_ear.core.memory import AuralMemory
from ai_ear.core.models import (
    AnalysisResult,
    AudioChunk,
    AuralEvent,
    AuralEventType,
    EmotionProfile,
    EnvironmentSnapshot,
    MusicProfile,
    SpeechSegment,
)
from ai_ear.core.pipeline import AudioPipeline

__version__ = "0.1.0"
__all__ = [
    "AudioListener",
    "AudioPipeline",
    "AuralMemory",
    "AudioChunk",
    "AnalysisResult",
    "SpeechSegment",
    "EmotionProfile",
    "EnvironmentSnapshot",
    "MusicProfile",
    "AuralEvent",
    "AuralEventType",
]
