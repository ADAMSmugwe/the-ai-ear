"""
AudioPipeline — concurrent multi-modal analysis engine.

Architecture
------------
The pipeline receives :class:`~ai_ear.core.models.AudioChunk` objects and fans
them out concurrently to all registered :class:`~ai_ear.analyzers.base.BaseAnalyzer`
implementations. Results are fused into a single :class:`~ai_ear.core.models.AnalysisResult`
and forwarded to the :class:`~ai_ear.core.memory.AuralMemory`.

Key design decisions
--------------------
* **Concurrent by default** – all analysers run in parallel via
  ``asyncio.gather``, so a slow model does not block faster ones.
* **Pluggable analysers** – add any :class:`BaseAnalyzer` at construction time.
* **Concurrency-limited** – an internal semaphore caps the number of in-flight
  analyses, naturally applying back-pressure without unbounded queuing.
* **Observable** – every fused result is broadcast to registered callbacks,
  enabling zero-copy fan-out to WebSocket clients, loggers, etc.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections.abc import Callable, Coroutine
from contextlib import suppress
from typing import Any

from ai_ear.core.models import (
    AnalysisResult,
    AudioChunk,
    AuralEvent,
    AuralEventType,
    EnvironmentLabel,
)

log = logging.getLogger(__name__)

# Type alias for result subscribers
ResultCallback = Callable[[AnalysisResult], Coroutine[Any, Any, None]]
EventCallback = Callable[[AuralEvent], Coroutine[Any, Any, None]]


class AudioPipeline:
    """
    Multi-modal audio analysis pipeline.

    Parameters
    ----------
    analyzers:
        Sequence of :class:`~ai_ear.analyzers.base.BaseAnalyzer` instances.
        Each analyser handles one modality (speech, emotion, environment, music).
    memory:
        Optional :class:`~ai_ear.core.memory.AuralMemory` instance.  If
        provided, every :class:`AnalysisResult` and derived :class:`AuralEvent`
        is automatically persisted.
    max_concurrent_chunks:
        How many chunks may be analysed simultaneously.  Prevents unbounded
        memory growth under high throughput.
    """

    def __init__(
        self,
        analyzers: list[Any] | None = None,
        memory: Any = None,
        max_concurrent_chunks: int = 4,
    ) -> None:
        from ai_ear.analyzers.base import BaseAnalyzer  # local import avoids circular dep

        self._analyzers: list[BaseAnalyzer] = analyzers or []
        self._memory = memory
        self._semaphore = asyncio.Semaphore(max_concurrent_chunks)
        self._result_callbacks: list[ResultCallback] = []
        self._event_callbacks: list[EventCallback] = []
        self._running = False
        self._stats = _PipelineStats()

        # Per-source-id state for event generation (avoids cross-talk between sources)
        self._prev_env_by_source: dict[str, EnvironmentLabel | None] = {}
        self._prev_music_active_by_source: dict[str, bool] = {}
        self._prev_speech_active_by_source: dict[str, bool] = {}

    # ------------------------------------------------------------------
    # Analyser registration
    # ------------------------------------------------------------------

    def add_analyzer(self, analyzer: Any) -> None:
        """
        Register an additional analyser at runtime.

        If the pipeline is already running, the analyser's ``load()`` coroutine
        is scheduled as a background task so it is initialised before being
        used for analysis.  If no event loop is running, the caller is
        responsible for awaiting ``analyzer.load()`` before the analyser is
        used.
        """
        self._analyzers.append(analyzer)
        if self._running:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(analyzer.load())
            except RuntimeError:
                log.warning(
                    "add_analyzer() called while running but no event loop is "
                    "active; analyzer.load() will not be called automatically"
                )

    # ------------------------------------------------------------------
    # Callback registration (fan-out subscribers)
    # ------------------------------------------------------------------

    def on_result(self, callback: ResultCallback) -> Callable[[], None]:
        """
        Register an async callback invoked for every AnalysisResult.

        Returns a callable that, when called, unregisters the callback.
        """
        self._result_callbacks.append(callback)

        def _unsubscribe() -> None:
            with suppress(ValueError):
                self._result_callbacks.remove(callback)

        return _unsubscribe

    def on_event(self, callback: EventCallback) -> Callable[[], None]:
        """
        Register an async callback invoked for every AuralEvent.

        Returns a callable that, when called, unregisters the callback.
        """
        self._event_callbacks.append(callback)

        def _unsubscribe() -> None:
            with suppress(ValueError):
                self._event_callbacks.remove(callback)

        return _unsubscribe

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialise all analysers (e.g., load ML models)."""
        self._running = True
        for analyser in self._analyzers:
            await analyser.load()
        log.info("AudioPipeline started with %d analyser(s)", len(self._analyzers))

    async def stop(self) -> None:
        """Tear down all analysers."""
        self._running = False
        for analyser in self._analyzers:
            await analyser.unload()
        log.info("AudioPipeline stopped")

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    async def process(self, chunk: AudioChunk) -> AnalysisResult:
        """
        Analyse a single :class:`AudioChunk` through all registered analysers
        concurrently, dispatch results and events to all registered callbacks,
        and return the fused :class:`AnalysisResult`.
        """
        async with self._semaphore:
            result = await self._process_inner(chunk)
            await self._dispatch_result(result)
            events = self._derive_events(result)
            for event in events:
                await self._dispatch_event(event)
            return result

    async def process_stream(self, chunks: Any) -> None:
        """
        Consume an async-iterable of :class:`AudioChunk` objects indefinitely.

        This is the primary entry-point for continuous listening.  Each chunk
        is awaited in sequence; the semaphore inside :meth:`process` caps the
        number of analyses that may overlap.
        """
        self._running = True
        async for chunk in chunks:
            if not self._running:
                break
            await self._process_and_dispatch(chunk)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _process_and_dispatch(self, chunk: AudioChunk) -> None:
        """Fire-and-forget wrapper around :meth:`process` for use in tasks."""
        try:
            await self.process(chunk)
        except Exception:
            log.exception("Pipeline error processing chunk %s", chunk.source_id)

    async def _process_inner(self, chunk: AudioChunk) -> AnalysisResult:
        chunk_id = _chunk_id(chunk)
        t0 = time.perf_counter()

        # Fan out to all analysers concurrently
        tasks = [asyncio.create_task(a.analyse(chunk)) for a in self._analyzers]
        partial_results = await asyncio.gather(*tasks, return_exceptions=True)

        result = AnalysisResult(
            chunk_id=chunk_id,
            source_id=chunk.source_id,
            timestamp=chunk.timestamp,
            duration_s=chunk.duration_s,
        )

        confidences: list[float] = []
        for pr in partial_results:
            if isinstance(pr, Exception):
                log.warning("Analyser error: %s", pr)
                continue
            _merge_partial(result, pr)
            confidences.append(getattr(pr, "confidence", 1.0))

        result.overall_confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )
        result.semantic_tags = _derive_tags(result)

        elapsed = time.perf_counter() - t0
        self._stats.record(elapsed)
        log.debug("Chunk %s analysed in %.3fs", chunk_id[:8], elapsed)

        if self._memory is not None:
            await self._memory.store_result(result)

        return result

    async def _dispatch_result(self, result: AnalysisResult) -> None:
        for cb in self._result_callbacks:
            try:
                await cb(result)
            except Exception:
                log.exception("Result callback error")

    async def _dispatch_event(self, event: AuralEvent) -> None:
        if self._memory is not None:
            await self._memory.store_event(event)
        for cb in self._event_callbacks:
            try:
                await cb(event)
            except Exception:
                log.exception("Event callback error")

    def _derive_events(self, result: AnalysisResult) -> list[AuralEvent]:
        """Detect state transitions and surface them as AuralEvents."""
        events: list[AuralEvent] = []
        src = result.source_id

        # Environment change (tracked per source)
        env_now = result.environment.dominant if result.environment else None
        prev_env = self._prev_env_by_source.get(src)
        if env_now is not None and env_now != prev_env:
            events.append(
                AuralEvent(
                    event_type=AuralEventType.ENVIRONMENT_CHANGE,
                    source_id=src,
                    description=f"Environment changed to '{env_now.value}'",
                    payload={"previous": prev_env, "current": env_now},
                )
            )
            self._prev_env_by_source[src] = env_now

        # Speech transitions (tracked per source)
        speech_now = result.speech is not None and bool(result.speech.text.strip())
        prev_speech = self._prev_speech_active_by_source.get(src, False)
        if speech_now and not prev_speech:
            events.append(
                AuralEvent(
                    event_type=AuralEventType.SPEECH_STARTED,
                    source_id=src,
                    description="Speech detected",
                )
            )
        elif not speech_now and prev_speech:
            events.append(
                AuralEvent(
                    event_type=AuralEventType.SPEECH_ENDED,
                    source_id=src,
                    description="Speech ended",
                )
            )
        self._prev_speech_active_by_source[src] = speech_now

        # Music transitions (tracked per source)
        music_now = result.music is not None and result.music.is_music
        prev_music = self._prev_music_active_by_source.get(src, False)
        if music_now and not prev_music:
            events.append(
                AuralEvent(
                    event_type=AuralEventType.MUSIC_STARTED,
                    source_id=src,
                    description="Music detected",
                    payload={
                        "tempo_bpm": result.music.tempo_bpm if result.music else None,
                        "key": result.music.key if result.music else None,
                    },
                )
            )
        elif not music_now and prev_music:
            events.append(
                AuralEvent(
                    event_type=AuralEventType.MUSIC_ENDED,
                    source_id=src,
                    description="Music ended",
                )
            )
        self._prev_music_active_by_source[src] = music_now

        # Alarm detection
        if env_now == EnvironmentLabel.ALARM:
            events.append(
                AuralEvent(
                    event_type=AuralEventType.ALARM_DETECTED,
                    source_id=src,
                    description="Alarm sound detected",
                    severity=0.9,
                )
            )

        return events

    @property
    def stats(self) -> dict[str, float]:
        return self._stats.summary()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk_id(chunk: AudioChunk) -> str:
    raw = f"{chunk.source_id}:{chunk.timestamp}:{len(chunk.samples)}"
    return hashlib.sha1(raw.encode()).hexdigest()


def _merge_partial(result: AnalysisResult, partial: Any) -> None:
    """Merge a partial analyser result into the composite AnalysisResult."""
    from ai_ear.analyzers.base import EmotionResult, EnvironmentResult, MusicResult, SpeechResult

    if isinstance(partial, SpeechResult):
        result.speech = partial.segment
    elif isinstance(partial, EmotionResult):
        result.emotion = partial.profile
    elif isinstance(partial, EnvironmentResult):
        result.environment = partial.snapshot
    elif isinstance(partial, MusicResult):
        result.music = partial.profile


def _derive_tags(result: AnalysisResult) -> list[str]:
    """Generate semantic tags from the fused result."""
    tags: list[str] = []

    if result.speech and result.speech.text.strip():
        tags.append("contains_speech")
        lang = result.speech.language
        if lang and lang != "en":
            tags.append(f"language:{lang}")

    if result.emotion:
        tags.append(f"emotion:{result.emotion.dominant.value}")
        if result.emotion.arousal > 0.7:
            tags.append("high_arousal")
        if result.emotion.valence < 0.3:
            tags.append("negative_valence")

    if result.environment:
        tags.append(f"env:{result.environment.dominant.value}")
        if result.environment.snr_db < 10:
            tags.append("noisy_environment")

    if result.music and result.music.is_music:
        tags.append("contains_music")
        if result.music.tempo_bpm:
            if result.music.tempo_bpm > 120:
                tags.append("fast_tempo")
            elif result.music.tempo_bpm < 70:
                tags.append("slow_tempo")

    return tags


class _PipelineStats:
    """Running statistics for pipeline latency."""

    def __init__(self) -> None:
        self._count = 0
        self._total = 0.0
        self._min = float("inf")
        self._max = 0.0

    def record(self, elapsed: float) -> None:
        self._count += 1
        self._total += elapsed
        self._min = min(self._min, elapsed)
        self._max = max(self._max, elapsed)

    def summary(self) -> dict[str, float]:
        avg = self._total / self._count if self._count else 0.0
        return {
            "chunks_processed": float(self._count),
            "avg_latency_s": avg,
            "min_latency_s": self._min if self._count else 0.0,
            "max_latency_s": self._max,
        }
