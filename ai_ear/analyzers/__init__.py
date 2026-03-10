"""Analyzers subpackage: base class + concrete implementations."""

from ai_ear.analyzers.base import (
    BaseAnalyzer,
    EmotionResult,
    EnvironmentResult,
    MusicResult,
    SpeechResult,
)
from ai_ear.analyzers.emotion import EmotionAnalyzer
from ai_ear.analyzers.environment import EnvironmentAnalyzer
from ai_ear.analyzers.music import MusicAnalyzer
from ai_ear.analyzers.speech import SpeechAnalyzer

__all__ = [
    "BaseAnalyzer",
    "SpeechResult",
    "EmotionResult",
    "EnvironmentResult",
    "MusicResult",
    "SpeechAnalyzer",
    "EmotionAnalyzer",
    "EnvironmentAnalyzer",
    "MusicAnalyzer",
]
