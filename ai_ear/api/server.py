"""
FastAPI REST + WebSocket server for the AI Ear.

Endpoints
---------
GET  /health                   — liveness / readiness probe
GET  /info                     — build info and configuration summary
POST /analyse                  — analyse a single uploaded audio file
GET  /memory/context           — retrieve the current aural context summary
GET  /memory/transcript        — plain-text transcript of recent speech
GET  /memory/events            — recent aural events
WS   /stream                   — real-time WebSocket audio streaming endpoint
GET  /pipeline/stats           — pipeline throughput statistics
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ai_ear.core.config import Settings
from ai_ear.core.memory import AuralMemory
from ai_ear.core.models import AnalysisResult, AuralEvent
from ai_ear.core.pipeline import AudioPipeline

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    timestamp: float
    uptime_s: float


class InfoResponse(BaseModel):
    version: str
    whisper_model: str
    emotion_enabled: bool
    music_enabled: bool
    environment_enabled: bool
    sample_rate: int


class ContextResponse(BaseModel):
    window_s: float
    transcript: str
    dominant_emotions: list[tuple[str, int]]
    dominant_environments: list[tuple[str, int]]
    music_detected: bool
    events: list[dict[str, Any]]
    semantic_tags: list[str]


class StatsResponse(BaseModel):
    chunks_processed: float
    avg_latency_s: float
    min_latency_s: float
    max_latency_s: float


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(settings: Settings | None = None) -> FastAPI:
    """
    Create and configure the FastAPI application.

    This factory pattern allows multiple instances to coexist in tests.
    """
    if settings is None:
        settings = Settings()

    _start_time = time.time()

    # ----------------------------------------------------------------
    # Build pipeline components
    # ----------------------------------------------------------------
    memory = AuralMemory(
        max_results=settings.memory_max_results,
        max_events=settings.memory_max_events,
        context_window_s=settings.memory_context_window_s,
    )

    analyzers = _build_analyzers(settings)
    pipeline = AudioPipeline(analyzers=analyzers, memory=memory)

    # ----------------------------------------------------------------
    # Lifespan: start/stop pipeline with the server
    # ----------------------------------------------------------------
    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[type-arg]
        await pipeline.start()
        log.info("AI Ear API server ready")
        yield
        await pipeline.stop()
        log.info("AI Ear API server shut down")

    app = FastAPI(
        title="The AI Ear",
        description=(
            "Enterprise-grade multi-modal AI audio listening and understanding API. "
            "Hear beyond words: speech, emotion, environment and music — unified."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ----------------------------------------------------------------
    # Health / info
    # ----------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            timestamp=time.time(),
            uptime_s=time.time() - _start_time,
        )

    @app.get("/info", response_model=InfoResponse, tags=["System"])
    async def info() -> InfoResponse:
        return InfoResponse(
            version="0.1.0",
            whisper_model=settings.whisper_model,
            emotion_enabled=settings.emotion_enabled,
            music_enabled=settings.music_enabled,
            environment_enabled=settings.environment_enabled,
            sample_rate=settings.audio_sample_rate,
        )

    # ----------------------------------------------------------------
    # Single-file analysis
    # ----------------------------------------------------------------

    @app.post("/analyse", response_model=AnalysisResult, tags=["Analysis"])
    async def analyse_file(file: UploadFile = File(...)) -> AnalysisResult:
        """
        Upload an audio file (WAV, FLAC, OGG, MP3) and receive a full
        multi-modal analysis result.
        """
        data = await file.read()
        chunk = _load_audio_bytes(data, settings.audio_sample_rate, source_id=file.filename or "upload")
        if chunk is None:
            raise HTTPException(status_code=422, detail="Could not decode audio file")

        from ai_ear.core.models import AudioChunk
        result = await pipeline.process(chunk)
        return result

    # ----------------------------------------------------------------
    # Memory / context endpoints
    # ----------------------------------------------------------------

    @app.get("/memory/context", response_model=ContextResponse, tags=["Memory"])
    async def context(
        window_s: float = Query(default=60.0, gt=0, description="Context window in seconds"),
    ) -> ContextResponse:
        """Return a structured context summary of recently heard audio."""
        summary = memory.context_summary(window_s=window_s)
        return ContextResponse(**summary)

    @app.get("/memory/transcript", tags=["Memory"])
    async def transcript(
        window_s: float = Query(default=60.0, gt=0),
    ) -> dict[str, str]:
        """Return a plain-text transcript of recently heard speech."""
        return {"transcript": memory.transcript(window_s=window_s)}

    @app.get("/memory/events", tags=["Memory"])
    async def events(
        last_n: int = Query(default=50, ge=1, le=500),
        event_type: str | None = Query(default=None),
    ) -> dict[str, Any]:
        """Return recent aural events, optionally filtered by type."""
        from ai_ear.core.models import AuralEventType
        et = None
        if event_type:
            try:
                et = AuralEventType(event_type)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Unknown event_type '{event_type}'")
        evts = memory.recent_events(last_n=last_n, event_type=et)
        return {"events": [e.model_dump() for e in evts], "count": len(evts)}

    # ----------------------------------------------------------------
    # Pipeline stats
    # ----------------------------------------------------------------

    @app.get("/pipeline/stats", response_model=StatsResponse, tags=["System"])
    async def pipeline_stats() -> StatsResponse:
        return StatsResponse(**pipeline.stats)

    # ----------------------------------------------------------------
    # WebSocket streaming
    # ----------------------------------------------------------------

    @app.websocket("/stream")
    async def stream(websocket: WebSocket) -> None:
        """
        Real-time WebSocket audio streaming.

        Protocol
        --------
        Client → Server: binary frames of raw PCM float32 mono audio at the
                         configured sample rate (default 16 kHz).
        Server → Client: JSON-encoded :class:`AnalysisResult` after each chunk.

        The server auto-assembles incoming binary data into chunks of the
        configured window size before running analysis.
        """
        await websocket.accept()
        source_id = f"ws:{uuid.uuid4().hex[:8]}"
        log.info("WebSocket client connected: %s", source_id)

        buffer = np.empty(0, dtype=np.float32)
        chunk_frames = int(settings.audio_sample_rate * settings.audio_chunk_duration_s)
        sample_rate = settings.audio_sample_rate

        async def _dispatch(result: AnalysisResult) -> None:
            try:
                await websocket.send_json(result.model_dump())
            except Exception:
                pass

        pipeline.on_result(_dispatch)

        try:
            while True:
                raw = await websocket.receive_bytes()
                incoming = np.frombuffer(raw, dtype=np.float32)
                buffer = np.concatenate([buffer, incoming])

                while len(buffer) >= chunk_frames:
                    window = buffer[:chunk_frames]
                    buffer = buffer[chunk_frames:]

                    from ai_ear.core.models import AudioChunk
                    chunk = AudioChunk(
                        samples=window,
                        sample_rate=sample_rate,
                        source_id=source_id,
                    )
                    asyncio.create_task(pipeline.process(chunk))
        except WebSocketDisconnect:
            log.info("WebSocket client disconnected: %s", source_id)
        except Exception:
            log.exception("WebSocket error for %s", source_id)
        finally:
            await websocket.close()

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_analyzers(settings: Settings) -> list:
    from ai_ear.analyzers.speech import SpeechAnalyzer
    from ai_ear.analyzers.emotion import EmotionAnalyzer
    from ai_ear.analyzers.environment import EnvironmentAnalyzer
    from ai_ear.analyzers.music import MusicAnalyzer

    analyzers: list = [
        EnvironmentAnalyzer(
            sample_rate=settings.audio_sample_rate,
            noise_gate_db=settings.environment_noise_gate_db,
        ),
    ]
    if settings.speech_enabled:
        analyzers.insert(
            0,
            SpeechAnalyzer(
                model_size=settings.whisper_model,
                language=settings.whisper_language,
                device=settings.whisper_device,
            ),
        )
    if settings.emotion_enabled:
        analyzers.append(
            EmotionAnalyzer(
                model_id=settings.emotion_model,
                device=settings.whisper_device,
                sample_rate=settings.audio_sample_rate,
            )
        )
    if settings.music_enabled:
        analyzers.append(
            MusicAnalyzer(
                sample_rate=settings.audio_sample_rate,
            )
        )
    return analyzers


def _load_audio_bytes(
    data: bytes, target_sr: int, source_id: str = "upload"
) -> "AudioChunk | None":  # type: ignore[name-defined]
    """Decode raw audio bytes into an AudioChunk using soundfile."""
    try:
        import soundfile as sf  # type: ignore[import-untyped]

        buf = io.BytesIO(data)
        audio, sr = sf.read(buf, dtype="float32", always_2d=True)
        mono = audio.mean(axis=1)

        if sr != target_sr:
            try:
                import librosa  # type: ignore[import-untyped]
                mono = librosa.resample(mono, orig_sr=sr, target_sr=target_sr)
            except ImportError:
                pass  # best-effort

        from ai_ear.core.models import AudioChunk
        return AudioChunk(samples=mono, sample_rate=target_sr, source_id=source_id)
    except Exception:
        log.exception("Failed to decode audio bytes")
        return None
