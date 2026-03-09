"""
Smoke tests for the AI Ear system.

These tests verify that the whole system can be composed and exercised
end-to-end without any heavy ML models loaded (all analysers use lightweight
heuristics or mocks).  They act as a quick integration sanity check.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ai_ear.analyzers.environment import EnvironmentAnalyzer
from ai_ear.analyzers.music import MusicAnalyzer
from ai_ear.api.server import create_app
from ai_ear.core.config import Settings
from ai_ear.core.listener import AudioListener
from ai_ear.core.memory import AuralMemory
from ai_ear.core.models import (
    AnalysisResult,
    AudioChunk,
    AuralEventType,
    EnvironmentLabel,
    MusicProfile,
    SpeechSegment,
)
from ai_ear.core.pipeline import AudioPipeline
from ai_ear.utils.audio import generate_noise, generate_silence, generate_tone

SR = 16_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(samples: np.ndarray, source: str = "smoke") -> AudioChunk:
    return AudioChunk(samples=samples, sample_rate=SR, source_id=source)


def _light_pipeline() -> AudioPipeline:
    """Pipeline with only no-model analysers for fast smoke testing."""
    return AudioPipeline(
        analyzers=[
            EnvironmentAnalyzer(sample_rate=SR),
            MusicAnalyzer(sample_rate=SR),
        ]
    )


def _test_settings() -> Settings:
    return Settings(
        speech_enabled=False,
        emotion_enabled=False,
        music_enabled=True,
        environment_enabled=True,
        audio_sample_rate=SR,
        audio_chunk_duration_s=1.0,
    )


# ---------------------------------------------------------------------------
# Smoke: full pipeline + memory flow
# ---------------------------------------------------------------------------

class TestPipelineMemorySmoke:
    @pytest.mark.asyncio
    async def test_silence_through_pipeline(self):
        """Silence passes through without errors and is classified."""
        pipeline = _light_pipeline()
        memory = AuralMemory()
        pipeline._memory = memory
        await pipeline.start()

        result = await pipeline.process(_chunk(generate_silence(2.0, SR)))

        assert result.chunk_id
        assert result.environment is not None
        assert result.environment.dominant == EnvironmentLabel.SILENCE
        assert memory.result_count == 1
        await pipeline.stop()

    @pytest.mark.asyncio
    async def test_tone_through_pipeline(self):
        """A pure tone is classified and produces a non-silence environment."""
        pipeline = _light_pipeline()
        await pipeline.start()

        result = await pipeline.process(_chunk(generate_tone(440.0, 2.0, SR)))

        assert result.environment is not None
        assert result.environment.dominant != EnvironmentLabel.SILENCE
        await pipeline.stop()

    @pytest.mark.asyncio
    async def test_noise_through_pipeline(self):
        """White noise passes through without errors."""
        rng = np.random.default_rng(42)
        pipeline = _light_pipeline()
        await pipeline.start()

        result = await pipeline.process(_chunk(generate_noise(2.0, SR, rng=rng)))

        assert result.chunk_id
        assert result.environment is not None
        await pipeline.stop()

    @pytest.mark.asyncio
    async def test_context_summary_populated_after_processing(self):
        """context_summary() returns non-empty data after processing several chunks."""
        memory = AuralMemory(context_window_s=300)
        pipeline = _light_pipeline()
        pipeline._memory = memory
        await pipeline.start()

        for _ in range(3):
            await pipeline.process(_chunk(generate_tone(440.0, 2.0, SR)))

        summary = memory.context_summary()
        assert summary["dominant_environments"]  # at least one env entry
        assert memory.result_count == 3
        await pipeline.stop()

    @pytest.mark.asyncio
    async def test_multiple_chunks_stream(self):
        """process_stream() handles an async generator of chunks."""
        pipeline = _light_pipeline()
        await pipeline.start()
        count = 0

        async def callback(result: AnalysisResult) -> None:
            nonlocal count
            count += 1

        pipeline.on_result(callback)

        async def _gen():
            for _ in range(4):
                yield _chunk(generate_silence(1.0, SR))

        await pipeline.process_stream(_gen())
        assert count == 4
        await pipeline.stop()


# ---------------------------------------------------------------------------
# Smoke: event generation end-to-end
# ---------------------------------------------------------------------------

class TestEventGenerationSmoke:
    @pytest.mark.asyncio
    async def test_silence_start_stop_events(self):
        """Alternating silence and non-silence generates environment-change events."""
        memory = AuralMemory()
        pipeline = _light_pipeline()
        pipeline._memory = memory
        await pipeline.start()

        await pipeline.process(_chunk(generate_silence(2.0, SR)))
        await pipeline.process(_chunk(generate_tone(440.0, 2.0, SR)))
        await pipeline.process(_chunk(generate_silence(2.0, SR)))

        stored = memory.recent_events(event_type=AuralEventType.ENVIRONMENT_CHANGE)
        assert len(stored) >= 2  # at least silence->X and X->silence
        await pipeline.stop()

    @pytest.mark.asyncio
    async def test_music_start_event_emitted(self):
        """MUSIC_STARTED event is stored in memory when music begins."""
        memory = AuralMemory()
        pipeline = _light_pipeline()
        pipeline._memory = memory
        await pipeline.start()

        # Inject a result that marks music as active directly
        r = AnalysisResult(
            chunk_id="music_test",
            source_id="smoke",
            music=MusicProfile(is_music=True, tempo_bpm=120.0),
        )
        pipeline._prev_music_active_by_source["smoke"] = False
        events = pipeline._derive_events(r)
        for e in events:
            await pipeline._dispatch_event(e)

        stored = memory.recent_events(event_type=AuralEventType.MUSIC_STARTED)
        assert len(stored) == 1
        assert stored[0].payload["tempo_bpm"] == pytest.approx(120.0)
        await pipeline.stop()


# ---------------------------------------------------------------------------
# Smoke: memory context as LLM prompt material
# ---------------------------------------------------------------------------

class TestMemoryContextSmoke:
    @pytest.mark.asyncio
    async def test_transcript_accumulates(self):
        """Speech added directly to memory appears in context_summary transcript."""
        memory = AuralMemory(context_window_s=300)
        phrases = ["hello world", "the quick brown fox", "enterprise AI"]
        for i, phrase in enumerate(phrases):
            r = AnalysisResult(
                chunk_id=f"r{i}",
                source_id="smoke",
                speech=SpeechSegment(text=phrase),
            )
            await memory.store_result(r)

        summary = memory.context_summary()
        for phrase in phrases:
            assert phrase in summary["transcript"]

    @pytest.mark.asyncio
    async def test_context_summary_structure(self):
        """context_summary() always returns a dict with all required keys."""
        memory = AuralMemory()
        summary = memory.context_summary()
        required = {
            "window_s", "transcript", "dominant_emotions",
            "dominant_environments", "music_detected", "events", "semantic_tags",
        }
        assert required.issubset(summary.keys())


# ---------------------------------------------------------------------------
# Smoke: AudioListener file ingestion
# ---------------------------------------------------------------------------

class TestAudioListenerSmoke:
    @pytest.mark.asyncio
    async def test_file_ingestion_yields_chunks(self, tmp_path):
        """ingest_file() yields at least one AudioChunk from a WAV file."""
        try:
            import soundfile as sf
        except ImportError:
            pytest.skip("soundfile not available")

        import numpy as np

        wav = tmp_path / "test.wav"
        samples = np.zeros(SR * 3, dtype=np.float32)
        sf.write(str(wav), samples, SR)

        listener = AudioListener(sample_rate=SR, chunk_duration_s=1.0)
        chunks = []
        async for chunk in listener.ingest_file(wav):
            chunks.append(chunk)

        assert len(chunks) >= 1
        assert all(c.sample_rate == SR for c in chunks)
        assert all(c.samples.ndim == 1 for c in chunks)


# ---------------------------------------------------------------------------
# Smoke: REST API
# ---------------------------------------------------------------------------

class TestAPISmoke:
    @pytest.fixture
    def client(self):
        app = create_app(_test_settings())
        with TestClient(app) as c:
            yield c

    def test_full_health_flow(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_empty_context_then_transcript(self, client):
        ctx = client.get("/memory/context").json()
        assert ctx["transcript"] == ""
        tx = client.get("/memory/transcript").json()
        assert tx["transcript"] == ""

    def test_events_empty_then_filter(self, client):
        resp = client.get("/memory/events?last_n=10")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    def test_stats_zero_initially(self, client):
        resp = client.get("/pipeline/stats")
        assert resp.status_code == 200
        assert resp.json()["chunks_processed"] == 0.0

    def test_analyse_valid_wav(self, client):
        try:
            import io as _io

            import soundfile as sf
            buf = _io.BytesIO()
            sf.write(buf, np.zeros(SR, dtype=np.float32), SR, format="WAV")
            resp = client.post(
                "/analyse",
                files={"file": ("test.wav", buf.getvalue(), "audio/wav")},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "chunk_id" in data
        except ImportError:
            pytest.skip("soundfile not available")

    def test_analyse_invalid_data(self, client):
        resp = client.post(
            "/analyse",
            files={"file": ("bad.wav", b"\x00\xff\xab\xcd", "audio/wav")},
        )
        assert resp.status_code == 422
