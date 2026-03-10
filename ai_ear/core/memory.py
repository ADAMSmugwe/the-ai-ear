"""
AuralMemory — temporal context and semantic recall.

The AI Ear is not a passive transcription machine; it *remembers* what it has
heard and can synthesise context across time.  AuralMemory provides:

* **Rolling result store** – bounded deque of recent :class:`AnalysisResult`
  objects with O(1) append and O(n) retrieval.
* **Event log** – timestamped :class:`AuralEvent` log for reactive alerting
  and audit.
* **Context summary** – on-demand natural-language summary of recent activity,
  ready to be injected into an LLM prompt or displayed to a user.
* **Speaker / acoustic fingerprinting hooks** – (extensible) placeholder for
  identifying recurring voices and sound signatures across sessions.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any

from ai_ear.core.models import AnalysisResult, AuralEvent, AuralEventType


class AuralMemory:
    """
    In-process temporal memory for the AI Ear pipeline.

    Parameters
    ----------
    max_results:
        Maximum number of :class:`AnalysisResult` entries to retain.
    max_events:
        Maximum number of :class:`AuralEvent` entries to retain.
    context_window_s:
        How far back (in seconds) the context summary looks.
    """

    def __init__(
        self,
        max_results: int = 500,
        max_events: int = 1000,
        context_window_s: float = 60.0,
    ) -> None:
        self._results: deque[AnalysisResult] = deque(maxlen=max_results)
        self._events: deque[AuralEvent] = deque(maxlen=max_events)
        self._context_window_s = context_window_s

        # Acoustic fingerprint registry (source_id -> list of representative embeddings)
        # Populated by external enrichment; kept as opaque blobs here.
        self._fingerprints: dict[str, list[Any]] = {}

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    async def store_result(self, result: AnalysisResult) -> None:
        """Append a new analysis result (thread-safe for asyncio coroutines)."""
        self._results.append(result)

    async def store_event(self, event: AuralEvent) -> None:
        """Append a new aural event."""
        self._events.append(event)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def recent_results(self, last_n: int | None = None) -> list[AnalysisResult]:
        """Return the most recent ``last_n`` results (all if ``None``)."""
        results = list(self._results)
        if last_n is not None:
            return results[-last_n:]
        return results

    def results_since(self, since_ts: float) -> list[AnalysisResult]:
        """Return all results with ``timestamp >= since_ts``."""
        return [r for r in self._results if r.timestamp >= since_ts]

    def recent_events(
        self,
        last_n: int | None = None,
        event_type: AuralEventType | None = None,
    ) -> list[AuralEvent]:
        """Return recent events, optionally filtered by type."""
        events = list(self._events)
        if event_type is not None:
            events = [e for e in events if e.event_type == event_type]
        if last_n is not None:
            return events[-last_n:]
        return events

    def events_since(
        self,
        since_ts: float,
        event_type: AuralEventType | None = None,
    ) -> list[AuralEvent]:
        """Return all events since ``since_ts``, optionally filtered by type."""
        events = [e for e in self._events if e.timestamp >= since_ts]
        if event_type is not None:
            events = [e for e in events if e.event_type == event_type]
        return events

    # ------------------------------------------------------------------
    # Context synthesis
    # ------------------------------------------------------------------

    def context_summary(self, window_s: float | None = None) -> dict[str, Any]:
        """
        Produce a structured context summary of recent activity.

        This summary is designed to be serialised as JSON and injected
        into an LLM system prompt so that an AI assistant "knows" what
        was recently heard.

        Returns
        -------
        dict with keys:
            - ``window_s`` – time window summarised
            - ``transcript`` – concatenated speech text from the window
            - ``dominant_emotions`` – list of emotion labels with counts
            - ``dominant_environments`` – list of env labels with counts
            - ``music_detected`` – bool
            - ``events`` – list of notable event descriptions
            - ``semantic_tags`` – aggregated unique tags
        """
        window = window_s if window_s is not None else self._context_window_s
        since = time.time() - window
        recent = self.results_since(since)
        events = self.events_since(since)

        transcript_parts = []
        emotions: dict[str, int] = {}
        environments: dict[str, int] = {}
        tags: set[str] = set()
        music_detected = False

        for r in recent:
            if r.speech and r.speech.text.strip():
                transcript_parts.append(r.speech.text.strip())
            if r.emotion:
                label = r.emotion.dominant.value
                emotions[label] = emotions.get(label, 0) + 1
            if r.environment:
                label = r.environment.dominant.value
                environments[label] = environments.get(label, 0) + 1
            if r.music and r.music.is_music:
                music_detected = True
            tags.update(r.semantic_tags)

        return {
            "window_s": window,
            "transcript": " ".join(transcript_parts),
            "dominant_emotions": sorted(emotions.items(), key=lambda x: -x[1]),
            "dominant_environments": sorted(environments.items(), key=lambda x: -x[1]),
            "music_detected": music_detected,
            "events": [
                {"type": e.event_type.value, "description": e.description, "ts": e.timestamp}
                for e in events
            ],
            "semantic_tags": sorted(tags),
        }

    def transcript(self, window_s: float | None = None) -> str:
        """Return a plain-text transcript of recent speech."""
        return self.context_summary(window_s)["transcript"]

    # ------------------------------------------------------------------
    # Acoustic fingerprinting (extensible hooks)
    # ------------------------------------------------------------------

    def register_fingerprint(self, source_id: str, embedding: Any) -> None:
        """
        Register an acoustic embedding for a known source.

        ``embedding`` is intentionally typed as ``Any`` – callers may use
        raw numpy arrays, sentence-transformer vectors, or proprietary blobs.
        """
        self._fingerprints.setdefault(source_id, []).append(embedding)

    def known_sources(self) -> list[str]:
        """Return the list of sources with registered fingerprints."""
        return list(self._fingerprints.keys())

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def result_count(self) -> int:
        return len(self._results)

    @property
    def event_count(self) -> int:
        return len(self._events)

    def clear(self) -> None:
        """Wipe all stored results and events (useful for testing)."""
        self._results.clear()
        self._events.clear()
        self._fingerprints.clear()
