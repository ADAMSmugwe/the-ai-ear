"""
Tests for ai_ear.core.models — data model validation and behaviour.
"""

import time

import numpy as np
import pytest

from ai_ear.core.models import (
    AnalysisResult,
    AudioChunk,
    AuralEvent,
    AuralEventType,
    EmotionLabel,
    EmotionProfile,
    EnvironmentLabel,
    EnvironmentSnapshot,
    MusicProfile,
    SpeechSegment,
)

SR = 16_000


class TestAudioChunk:
    def test_basic_creation(self):
        samples = np.zeros(SR, dtype=np.float32)
        chunk = AudioChunk(samples=samples, sample_rate=SR)
        assert chunk.sample_rate == SR
        assert chunk.duration_s == pytest.approx(1.0)

    def test_duration_computed_correctly(self):
        samples = np.zeros(SR // 2, dtype=np.float32)
        chunk = AudioChunk(samples=samples, sample_rate=SR)
        assert chunk.duration_s == pytest.approx(0.5)

    def test_2d_samples_accepted(self):
        # Stereo input
        samples = np.zeros((SR, 2), dtype=np.float32)
        chunk = AudioChunk(samples=samples, sample_rate=SR)
        assert chunk.samples.ndim == 2

    def test_invalid_ndim_rejected(self):
        with pytest.raises(Exception):
            AudioChunk(samples=np.zeros((SR, 2, 3), dtype=np.float32), sample_rate=SR)

    def test_source_id_default(self):
        chunk = AudioChunk(samples=np.zeros(100, dtype=np.float32), sample_rate=SR)
        assert chunk.source_id == "default"

    def test_timestamp_is_recent(self):
        before = time.time()
        chunk = AudioChunk(samples=np.zeros(100, dtype=np.float32), sample_rate=SR)
        after = time.time()
        assert before <= chunk.timestamp <= after

    def test_sample_rate_bounds(self):
        with pytest.raises(Exception):
            AudioChunk(samples=np.zeros(100, dtype=np.float32), sample_rate=100)
        with pytest.raises(Exception):
            AudioChunk(samples=np.zeros(100, dtype=np.float32), sample_rate=300_000)


class TestSpeechSegment:
    def test_defaults(self):
        seg = SpeechSegment(text="hello")
        assert seg.language == "en"
        assert seg.confidence == 1.0
        assert seg.words == []

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            SpeechSegment(text="hi", confidence=1.5)
        with pytest.raises(Exception):
            SpeechSegment(text="hi", confidence=-0.1)


class TestEmotionProfile:
    def test_defaults(self):
        ep = EmotionProfile()
        assert ep.dominant == EmotionLabel.NEUTRAL
        assert 0.0 <= ep.arousal <= 1.0
        assert 0.0 <= ep.valence <= 1.0

    def test_custom_scores(self):
        ep = EmotionProfile(
            dominant=EmotionLabel.HAPPY,
            scores={"happy": 0.8, "neutral": 0.2},
        )
        assert ep.dominant == EmotionLabel.HAPPY


class TestEnvironmentSnapshot:
    def test_defaults(self):
        snap = EnvironmentSnapshot()
        assert snap.dominant == EnvironmentLabel.UNKNOWN


class TestMusicProfile:
    def test_not_music_by_default(self):
        mp = MusicProfile()
        assert not mp.is_music

    def test_tempo_bpm_optional(self):
        mp = MusicProfile(is_music=True, tempo_bpm=120.0)
        assert mp.tempo_bpm == 120.0


class TestAnalysisResult:
    def test_creation_with_id(self):
        result = AnalysisResult(chunk_id="abc123")
        assert result.chunk_id == "abc123"
        assert result.speech is None
        assert result.emotion is None
        assert result.semantic_tags == []

    def test_full_result(self):
        result = AnalysisResult(
            chunk_id="test",
            speech=SpeechSegment(text="hello world"),
            emotion=EmotionProfile(dominant=EmotionLabel.HAPPY),
            environment=EnvironmentSnapshot(dominant=EnvironmentLabel.SPEECH),
            music=MusicProfile(is_music=False),
            semantic_tags=["contains_speech", "emotion:happy"],
        )
        assert result.speech.text == "hello world"
        assert result.emotion.dominant == EmotionLabel.HAPPY

    def test_json_serialisable(self):
        result = AnalysisResult(chunk_id="json_test")
        data = result.model_dump()
        assert "chunk_id" in data
        assert data["chunk_id"] == "json_test"


class TestAuralEvent:
    def test_creation(self):
        event = AuralEvent(
            event_type=AuralEventType.SPEECH_STARTED,
            description="Speech detected",
        )
        assert event.severity == 0.0
        assert event.source_id == "default"

    def test_alarm_severity(self):
        event = AuralEvent(
            event_type=AuralEventType.ALARM_DETECTED,
            severity=0.9,
        )
        assert event.severity == pytest.approx(0.9)

    def test_timestamp_is_recent(self):
        before = time.time()
        event = AuralEvent(event_type=AuralEventType.ANOMALY)
        after = time.time()
        assert before <= event.timestamp <= after
