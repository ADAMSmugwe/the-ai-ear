"""
Tests for ai_ear.core.pipeline — AudioPipeline concurrent analysis engine.

We use mock analysers to keep the tests fast and deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest

from ai_ear.analyzers.base import (
    BaseAnalyzer,
    EnvironmentResult,
    MusicResult,
    SpeechResult,
)
from ai_ear.core.memory import AuralMemory
from ai_ear.core.models import (
    AudioChunk,
    AuralEventType,
    EmotionLabel,
    EmotionProfile,
    EnvironmentLabel,
    EnvironmentSnapshot,
    MusicProfile,
    SpeechSegment,
)
from ai_ear.core.pipeline import AudioPipeline, _derive_tags

SR = 16_000


# ---------------------------------------------------------------------------
# Mock analysers
# ---------------------------------------------------------------------------

class MockSpeechAnalyzer(BaseAnalyzer):
    name = "speech"

    def __init__(self, text: str = "hello world"):
        self._text = text
        self.loaded = False

    async def load(self) -> None:
        self.loaded = True

    async def analyse(self, chunk: AudioChunk) -> SpeechResult:
        return SpeechResult(
            segment=SpeechSegment(text=self._text, language="en", confidence=0.9),
            confidence=0.9,
        )


class MockEnvironmentAnalyzer(BaseAnalyzer):
    name = "environment"

    def __init__(self, env: EnvironmentLabel = EnvironmentLabel.SPEECH):
        self._env = env

    async def analyse(self, chunk: AudioChunk) -> EnvironmentResult:
        return EnvironmentResult(
            snapshot=EnvironmentSnapshot(
                dominant=self._env,
                scores={self._env.value: 1.0},
                snr_db=20.0,
            ),
            confidence=0.95,
        )


class MockMusicAnalyzer(BaseAnalyzer):
    name = "music"

    def __init__(self, is_music: bool = False, tempo: float = 120.0):
        self._is_music = is_music
        self._tempo = tempo

    async def analyse(self, chunk: AudioChunk) -> MusicResult:
        return MusicResult(
            profile=MusicProfile(is_music=self._is_music, tempo_bpm=self._tempo),
            confidence=0.8,
        )


class FailingAnalyzer(BaseAnalyzer):
    name = "failing"

    async def analyse(self, chunk: AudioChunk) -> SpeechResult:
        raise RuntimeError("intentional test failure")


def _make_chunk(text: str = "") -> AudioChunk:
    return AudioChunk(
        samples=np.zeros(SR * 2, dtype=np.float32),
        sample_rate=SR,
        source_id="test",
    )


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------

class TestPipelineLifecycle:
    @pytest.mark.asyncio
    async def test_start_loads_all_analysers(self):
        speech = MockSpeechAnalyzer()
        pipeline = AudioPipeline(analyzers=[speech])
        await pipeline.start()
        assert speech.loaded
        await pipeline.stop()

    @pytest.mark.asyncio
    async def test_add_analyser_at_runtime(self):
        pipeline = AudioPipeline()
        await pipeline.start()
        env = MockEnvironmentAnalyzer()
        pipeline.add_analyzer(env)
        assert len(pipeline._analyzers) == 1
        await pipeline.stop()


class TestPipelineProcessing:
    @pytest.mark.asyncio
    async def test_basic_process(self):
        pipeline = AudioPipeline(
            analyzers=[MockSpeechAnalyzer("test utterance"), MockEnvironmentAnalyzer()]
        )
        await pipeline.start()
        chunk = _make_chunk()
        result = await pipeline.process(chunk)
        assert result.speech is not None
        assert result.speech.text == "test utterance"
        assert result.environment is not None
        assert result.environment.dominant == EnvironmentLabel.SPEECH
        await pipeline.stop()

    @pytest.mark.asyncio
    async def test_confidence_averaged(self):
        pipeline = AudioPipeline(
            analyzers=[
                MockSpeechAnalyzer(),    # confidence 0.9
                MockEnvironmentAnalyzer(),  # confidence 0.95
            ]
        )
        await pipeline.start()
        result = await pipeline.process(_make_chunk())
        # Average of 0.9 and 0.95 = 0.925
        assert 0.85 < result.overall_confidence < 1.0
        await pipeline.stop()

    @pytest.mark.asyncio
    async def test_failing_analyser_does_not_crash_pipeline(self):
        pipeline = AudioPipeline(
            analyzers=[MockSpeechAnalyzer("safe"), FailingAnalyzer()]
        )
        await pipeline.start()
        result = await pipeline.process(_make_chunk())
        # The safe analyser should still contribute
        assert result.speech is not None
        assert result.speech.text == "safe"
        await pipeline.stop()

    @pytest.mark.asyncio
    async def test_result_stored_in_memory(self):
        memory = AuralMemory()
        pipeline = AudioPipeline(analyzers=[MockSpeechAnalyzer()], memory=memory)
        await pipeline.start()
        await pipeline.process(_make_chunk())
        assert memory.result_count == 1
        await pipeline.stop()

    @pytest.mark.asyncio
    async def test_result_callback_invoked(self):
        received = []

        async def callback(result):
            received.append(result)

        pipeline = AudioPipeline(analyzers=[MockSpeechAnalyzer()])
        pipeline.on_result(callback)
        await pipeline.start()
        result = await pipeline.process(_make_chunk())
        assert len(received) == 1
        assert received[0].chunk_id == result.chunk_id
        await pipeline.stop()


class TestPipelineEventDetection:
    @pytest.mark.asyncio
    async def test_environment_change_event(self):
        events = []

        async def on_event(event):
            events.append(event)

        env_analyzer = MockEnvironmentAnalyzer(EnvironmentLabel.SPEECH)
        memory = AuralMemory()
        pipeline = AudioPipeline(analyzers=[env_analyzer], memory=memory)
        pipeline.on_event(on_event)
        await pipeline.start()

        # First chunk — triggers ENVIRONMENT_CHANGE from None to SPEECH
        await pipeline.process(_make_chunk())

        env_events = [e for e in events if e.event_type == AuralEventType.ENVIRONMENT_CHANGE]
        assert len(env_events) == 1
        assert env_events[0].payload["current"] == EnvironmentLabel.SPEECH

        await pipeline.stop()

    @pytest.mark.asyncio
    async def test_speech_started_event(self):
        events = []

        async def on_event(event):
            events.append(event)

        pipeline = AudioPipeline(analyzers=[MockSpeechAnalyzer("hello")])
        pipeline.on_event(on_event)
        await pipeline.start()

        await pipeline.process(_make_chunk())

        speech_started = [e for e in events if e.event_type == AuralEventType.SPEECH_STARTED]
        assert len(speech_started) == 1
        await pipeline.stop()

    @pytest.mark.asyncio
    async def test_music_started_event(self):
        events = []

        async def on_event(event):
            events.append(event)

        pipeline = AudioPipeline(analyzers=[MockMusicAnalyzer(is_music=True, tempo=128.0)])
        pipeline.on_event(on_event)
        await pipeline.start()

        await pipeline.process(_make_chunk())

        music_started = [e for e in events if e.event_type == AuralEventType.MUSIC_STARTED]
        assert len(music_started) == 1
        assert music_started[0].payload["tempo_bpm"] == pytest.approx(128.0)
        await pipeline.stop()


class TestPipelineStats:
    @pytest.mark.asyncio
    async def test_stats_tracked(self):
        pipeline = AudioPipeline(analyzers=[MockSpeechAnalyzer()])
        await pipeline.start()
        await pipeline.process(_make_chunk())
        stats = pipeline.stats
        assert stats["chunks_processed"] == 1.0
        assert stats["avg_latency_s"] >= 0.0
        await pipeline.stop()


# ---------------------------------------------------------------------------
# Tag derivation
# ---------------------------------------------------------------------------

class TestDeriveTags:
    def test_speech_tag(self):
        from ai_ear.core.models import AnalysisResult
        r = AnalysisResult(chunk_id="x", speech=SpeechSegment(text="hello"))
        tags = _derive_tags(r)
        assert "contains_speech" in tags

    def test_empty_speech_no_tag(self):
        from ai_ear.core.models import AnalysisResult
        r = AnalysisResult(chunk_id="x", speech=SpeechSegment(text="  "))
        tags = _derive_tags(r)
        assert "contains_speech" not in tags

    def test_emotion_tag(self):
        from ai_ear.core.models import AnalysisResult
        r = AnalysisResult(
            chunk_id="x",
            emotion=EmotionProfile(dominant=EmotionLabel.HAPPY),
        )
        tags = _derive_tags(r)
        assert "emotion:happy" in tags

    def test_high_arousal_tag(self):
        from ai_ear.core.models import AnalysisResult
        r = AnalysisResult(
            chunk_id="x",
            emotion=EmotionProfile(dominant=EmotionLabel.ANGRY, arousal=0.9),
        )
        tags = _derive_tags(r)
        assert "high_arousal" in tags

    def test_environment_tag(self):
        from ai_ear.core.models import AnalysisResult
        r = AnalysisResult(
            chunk_id="x",
            environment=EnvironmentSnapshot(dominant=EnvironmentLabel.OFFICE),
        )
        tags = _derive_tags(r)
        assert "env:office" in tags

    def test_music_tag_with_tempo(self):
        from ai_ear.core.models import AnalysisResult
        r = AnalysisResult(
            chunk_id="x",
            music=MusicProfile(is_music=True, tempo_bpm=130.0),
        )
        tags = _derive_tags(r)
        assert "contains_music" in tags
        assert "fast_tempo" in tags


# ---------------------------------------------------------------------------
# Per-source-id event isolation
# ---------------------------------------------------------------------------

class TestPerSourceEventIsolation:
    @pytest.mark.asyncio
    async def test_two_sources_have_independent_state(self):
        """Events for source B must not depend on source A's prior state."""
        events = []

        async def on_event(event):
            events.append(event)

        pipeline = AudioPipeline(analyzers=[MockEnvironmentAnalyzer(EnvironmentLabel.SPEECH)])
        pipeline.on_event(on_event)
        await pipeline.start()

        # Process source A — sets its env to SPEECH
        chunk_a = AudioChunk(
            samples=np.zeros(SR * 2, dtype=np.float32),
            sample_rate=SR,
            source_id="source_a",
        )
        await pipeline.process(chunk_a)

        # Process source B — should also get ENVIRONMENT_CHANGE (not suppressed by A)
        chunk_b = AudioChunk(
            samples=np.zeros(SR * 2, dtype=np.float32),
            sample_rate=SR,
            source_id="source_b",
        )
        await pipeline.process(chunk_b)

        env_a = [e for e in events if e.event_type == AuralEventType.ENVIRONMENT_CHANGE
                 and e.source_id == "source_a"]
        env_b = [e for e in events if e.event_type == AuralEventType.ENVIRONMENT_CHANGE
                 and e.source_id == "source_b"]

        assert len(env_a) == 1
        assert len(env_b) == 1  # B must see its own transition, not be silenced by A's state
        await pipeline.stop()


# ---------------------------------------------------------------------------
# Callback unsubscription
# ---------------------------------------------------------------------------

class TestCallbackUnsubscription:
    @pytest.mark.asyncio
    async def test_on_result_returns_unsubscribe(self):
        received = []

        async def callback(result):
            received.append(result)

        pipeline = AudioPipeline(analyzers=[MockSpeechAnalyzer()])
        await pipeline.start()

        unsubscribe = pipeline.on_result(callback)
        await pipeline.process(_make_chunk())
        assert len(received) == 1

        # Unsubscribe and verify no more callbacks
        unsubscribe()
        await pipeline.process(_make_chunk())
        assert len(received) == 1  # still 1 — callback was removed
        await pipeline.stop()

    @pytest.mark.asyncio
    async def test_on_event_returns_unsubscribe(self):
        received = []

        async def callback(event):
            received.append(event)

        pipeline = AudioPipeline(analyzers=[MockEnvironmentAnalyzer(EnvironmentLabel.SPEECH)])
        await pipeline.start()

        unsubscribe = pipeline.on_event(callback)
        await pipeline.process(_make_chunk())
        count_after_first = len(received)

        unsubscribe()
        await pipeline.process(_make_chunk())
        assert len(received) == count_after_first  # no new events after unsubscribe
        await pipeline.stop()


# ---------------------------------------------------------------------------
# Pipeline restart (executor recreation)
# ---------------------------------------------------------------------------

class TestPipelineRestart:
    @pytest.mark.asyncio
    async def test_start_stop_start_works(self):
        """Pipeline should be reusable after stop() / start()."""
        pipeline = AudioPipeline(analyzers=[MockSpeechAnalyzer("first")])
        await pipeline.start()
        result1 = await pipeline.process(_make_chunk())
        assert result1.speech.text == "first"
        await pipeline.stop()

        # Restart
        await pipeline.start()
        result2 = await pipeline.process(_make_chunk())
        assert result2.speech.text == "first"
        await pipeline.stop()
