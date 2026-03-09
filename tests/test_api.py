"""
Tests for the FastAPI REST + WebSocket server.

Uses httpx.AsyncClient with TestClient transport for fast in-process testing.
"""

from __future__ import annotations

import io
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ai_ear.api.server import create_app
from ai_ear.core.config import Settings


# Use minimal settings for speed (no real ML models)
def _test_settings() -> Settings:
    return Settings(
        speech_enabled=False,
        whisper_model="tiny",
        emotion_enabled=False,
        music_enabled=False,
        environment_enabled=True,
        audio_sample_rate=16_000,
        audio_chunk_duration_s=1.0,
    )


@pytest.fixture
def app():
    return create_app(_test_settings())


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Health / info
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "uptime_s" in data
        assert data["uptime_s"] >= 0

    def test_health_timestamp_recent(self, client):
        before = time.time()
        resp = client.get("/health")
        after = time.time()
        ts = resp.json()["timestamp"]
        assert before <= ts <= after


class TestInfoEndpoint:
    def test_info_returns_config(self, client):
        resp = client.get("/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["whisper_model"] == "tiny"
        assert data["sample_rate"] == 16_000

    def test_info_version_present(self, client):
        resp = client.get("/info")
        assert "version" in resp.json()


# ---------------------------------------------------------------------------
# Memory endpoints
# ---------------------------------------------------------------------------

class TestMemoryEndpoints:
    def test_context_empty(self, client):
        resp = client.get("/memory/context")
        assert resp.status_code == 200
        data = resp.json()
        assert data["transcript"] == ""
        assert not data["music_detected"]

    def test_transcript_empty(self, client):
        resp = client.get("/memory/transcript")
        assert resp.status_code == 200
        assert resp.json()["transcript"] == ""

    def test_events_empty(self, client):
        resp = client.get("/memory/events")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0
        assert data["events"] == []

    def test_events_invalid_type_returns_400(self, client):
        resp = client.get("/memory/events?event_type=not_a_real_type")
        assert resp.status_code == 400

    def test_context_window_param(self, client):
        resp = client.get("/memory/context?window_s=30")
        assert resp.status_code == 200
        assert resp.json()["window_s"] == 30.0


# ---------------------------------------------------------------------------
# Pipeline stats
# ---------------------------------------------------------------------------

class TestPipelineStats:
    def test_stats_initial_state(self, client):
        resp = client.get("/pipeline/stats")
        assert resp.status_code == 200
        stats = resp.json()
        assert "chunks_processed" in stats
        assert stats["chunks_processed"] == 0.0


# ---------------------------------------------------------------------------
# Audio file analysis (requires soundfile)
# ---------------------------------------------------------------------------

def _make_wav_bytes(duration_s: float = 1.0, sr: int = 16_000) -> bytes:
    """Generate a minimal in-memory WAV file."""
    try:
        import soundfile as sf
        samples = np.zeros(int(sr * duration_s), dtype=np.float32)
        buf = io.BytesIO()
        sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
        return buf.getvalue()
    except ImportError:
        return b""


class TestAnalyseEndpoint:
    def test_analyse_silence_wav(self, client):
        wav = _make_wav_bytes(1.0, 16_000)
        if not wav:
            pytest.skip("soundfile not available")
        resp = client.post(
            "/analyse",
            files={"file": ("silence.wav", wav, "audio/wav")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "chunk_id" in data
        assert "environment" in data

    def test_analyse_invalid_file_returns_422(self, client):
        resp = client.post(
            "/analyse",
            files={"file": ("garbage.wav", b"not audio at all", "audio/wav")},
        )
        assert resp.status_code == 422
