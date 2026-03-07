"""
Tests for ai_ear.core.memory — AuralMemory temporal context.
"""

import asyncio
import time

import pytest

from ai_ear.core.memory import AuralMemory
from ai_ear.core.models import (
    AnalysisResult,
    AuralEvent,
    AuralEventType,
    EmotionLabel,
    EmotionProfile,
    EnvironmentLabel,
    EnvironmentSnapshot,
    MusicProfile,
    SpeechSegment,
)


def _make_result(text: str = "", ts: float | None = None, **kw) -> AnalysisResult:
    result = AnalysisResult(
        chunk_id=f"test_{id(text)}",
        timestamp=ts if ts is not None else time.time(),
        **kw,
    )
    if text:
        result.speech = SpeechSegment(text=text)
    return result


def _make_event(event_type: AuralEventType, ts: float | None = None) -> AuralEvent:
    return AuralEvent(
        event_type=event_type,
        timestamp=ts if ts is not None else time.time(),
    )


class TestAuralMemoryStorage:
    @pytest.mark.asyncio
    async def test_store_and_retrieve_results(self):
        mem = AuralMemory()
        r1 = _make_result("hello")
        r2 = _make_result("world")
        await mem.store_result(r1)
        await mem.store_result(r2)
        assert mem.result_count == 2
        results = mem.recent_results()
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_maxsize_evicts_old_results(self):
        mem = AuralMemory(max_results=3)
        for i in range(5):
            await mem.store_result(_make_result(f"item {i}"))
        assert mem.result_count == 3
        # Most recent 3 should be retained
        texts = [r.speech.text for r in mem.recent_results()]
        assert "item 4" in texts
        assert "item 0" not in texts

    @pytest.mark.asyncio
    async def test_store_and_retrieve_events(self):
        mem = AuralMemory()
        e1 = _make_event(AuralEventType.SPEECH_STARTED)
        e2 = _make_event(AuralEventType.MUSIC_STARTED)
        await mem.store_event(e1)
        await mem.store_event(e2)
        assert mem.event_count == 2

    @pytest.mark.asyncio
    async def test_clear(self):
        mem = AuralMemory()
        await mem.store_result(_make_result("x"))
        await mem.store_event(_make_event(AuralEventType.ANOMALY))
        mem.clear()
        assert mem.result_count == 0
        assert mem.event_count == 0


class TestAuralMemoryRetrieval:
    @pytest.mark.asyncio
    async def test_recent_results_last_n(self):
        mem = AuralMemory()
        for i in range(10):
            await mem.store_result(_make_result(f"item {i}"))
        last3 = mem.recent_results(last_n=3)
        assert len(last3) == 3
        assert last3[-1].speech.text == "item 9"

    @pytest.mark.asyncio
    async def test_results_since(self):
        mem = AuralMemory()
        old_ts = time.time() - 100
        new_ts = time.time()
        await mem.store_result(_make_result("old", ts=old_ts))
        await mem.store_result(_make_result("new", ts=new_ts))
        since = time.time() - 50
        recent = mem.results_since(since)
        assert len(recent) == 1
        assert recent[0].speech.text == "new"

    @pytest.mark.asyncio
    async def test_events_filtered_by_type(self):
        mem = AuralMemory()
        await mem.store_event(_make_event(AuralEventType.SPEECH_STARTED))
        await mem.store_event(_make_event(AuralEventType.MUSIC_STARTED))
        await mem.store_event(_make_event(AuralEventType.SPEECH_ENDED))
        speech = mem.recent_events(event_type=AuralEventType.SPEECH_STARTED)
        assert len(speech) == 1
        assert speech[0].event_type == AuralEventType.SPEECH_STARTED

    @pytest.mark.asyncio
    async def test_events_since(self):
        mem = AuralMemory()
        old_ts = time.time() - 200
        await mem.store_event(_make_event(AuralEventType.ANOMALY, ts=old_ts))
        await mem.store_event(_make_event(AuralEventType.ANOMALY))
        recent = mem.events_since(time.time() - 10)
        assert len(recent) == 1


class TestAuralMemoryContextSummary:
    @pytest.mark.asyncio
    async def test_empty_summary(self):
        mem = AuralMemory()
        summary = mem.context_summary()
        assert summary["transcript"] == ""
        assert not summary["music_detected"]
        assert summary["events"] == []

    @pytest.mark.asyncio
    async def test_transcript_aggregated(self):
        mem = AuralMemory(context_window_s=3600)
        await mem.store_result(_make_result("Hello"))
        await mem.store_result(_make_result("World"))
        summary = mem.context_summary()
        assert "Hello" in summary["transcript"]
        assert "World" in summary["transcript"]

    @pytest.mark.asyncio
    async def test_dominant_emotion_counted(self):
        mem = AuralMemory(context_window_s=3600)
        for _ in range(3):
            r = _make_result()
            r.emotion = EmotionProfile(dominant=EmotionLabel.HAPPY)
            await mem.store_result(r)
        r2 = _make_result()
        r2.emotion = EmotionProfile(dominant=EmotionLabel.SAD)
        await mem.store_result(r2)
        summary = mem.context_summary()
        emotions = dict(summary["dominant_emotions"])
        assert emotions.get("happy", 0) == 3
        assert emotions.get("sad", 0) == 1

    @pytest.mark.asyncio
    async def test_music_detected_flag(self):
        mem = AuralMemory(context_window_s=3600)
        r = _make_result()
        r.music = MusicProfile(is_music=True, tempo_bpm=120.0)
        await mem.store_result(r)
        summary = mem.context_summary()
        assert summary["music_detected"]

    @pytest.mark.asyncio
    async def test_old_results_excluded_from_summary(self):
        mem = AuralMemory(context_window_s=10)
        old_ts = time.time() - 200
        await mem.store_result(_make_result("ancient text", ts=old_ts))
        summary = mem.context_summary(window_s=10)
        assert "ancient text" not in summary["transcript"]

    @pytest.mark.asyncio
    async def test_semantic_tags_aggregated(self):
        mem = AuralMemory(context_window_s=3600)
        r1 = _make_result()
        r1.semantic_tags = ["contains_speech", "emotion:happy"]
        r2 = _make_result()
        r2.semantic_tags = ["contains_music", "emotion:happy"]
        await mem.store_result(r1)
        await mem.store_result(r2)
        summary = mem.context_summary()
        assert "contains_speech" in summary["semantic_tags"]
        assert "contains_music" in summary["semantic_tags"]
        assert "emotion:happy" in summary["semantic_tags"]

    @pytest.mark.asyncio
    async def test_transcript_helper(self):
        mem = AuralMemory(context_window_s=3600)
        await mem.store_result(_make_result("quick brown fox"))
        text = mem.transcript()
        assert "quick brown fox" in text


class TestAuralMemoryFingerprints:
    def test_register_and_list(self):
        mem = AuralMemory()
        mem.register_fingerprint("alice", [1, 2, 3])
        mem.register_fingerprint("alice", [4, 5, 6])
        mem.register_fingerprint("bob", [7, 8, 9])
        sources = mem.known_sources()
        assert "alice" in sources
        assert "bob" in sources

    def test_clear_removes_fingerprints(self):
        mem = AuralMemory()
        mem.register_fingerprint("voice_1", object())
        mem.clear()
        assert mem.known_sources() == []
