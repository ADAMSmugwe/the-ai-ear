"""
enterprise_integration.py — enterprise integration patterns for the AI Ear.

Demonstrates:
1. Injecting custom analysers (BYOM — bring your own model)
2. Wiring pipeline events to an alerting system
3. Exposing the context summary as an LLM system prompt
4. Running the REST API server programmatically

Usage
-----
    # Start the API server
    python examples/enterprise_integration.py serve

    # Test custom analyser injection
    python examples/enterprise_integration.py custom-analyser

    # Generate an LLM-ready context prompt
    python examples/enterprise_integration.py llm-prompt
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

import numpy as np


# ---------------------------------------------------------------------------
# 1. Custom analyser: BYOM example
# ---------------------------------------------------------------------------

async def demo_custom_analyser() -> None:
    """
    Shows how to implement and inject a custom 'keyword spotter' analyser.
    """
    from ai_ear.analyzers.base import BaseAnalyzer, SpeechResult
    from ai_ear.core.memory import AuralMemory
    from ai_ear.core.models import AudioChunk, AuralEventType, SpeechSegment
    from ai_ear.core.pipeline import AudioPipeline

    class KeywordSpotter(BaseAnalyzer):
        """
        Trivial mock keyword spotter.  In production this would wrap a
        real-time on-device keyword detection model (e.g., openwakeword,
        Porcupine, or a custom TFLite model).
        """

        name = "keyword_spotter"

        def __init__(self, keywords: list[str]) -> None:
            self._keywords = [kw.lower() for kw in keywords]
            self._call_count = 0

        async def analyse(self, chunk: AudioChunk) -> SpeechResult:
            # Simulate every 3rd chunk contains a keyword
            self._call_count += 1
            detected = self._call_count % 3 == 0
            text = self._keywords[0] if detected else ""
            return SpeechResult(
                segment=SpeechSegment(
                    text=text,
                    confidence=0.95 if detected else 0.0,
                ),
                confidence=0.95 if detected else 0.0,
            )

    print("\n🔌  Enterprise Integration Demo: Custom Analyser (BYOM)\n")

    keyword_spotter = KeywordSpotter(keywords=["hey ai", "alert", "emergency"])
    memory = AuralMemory()
    pipeline = AudioPipeline(analyzers=[keyword_spotter], memory=memory)

    keywords_detected: list[str] = []

    async def on_result(result):
        if result.speech and result.speech.text:
            keywords_detected.append(result.speech.text)
            print(f"  🔑 Keyword detected: '{result.speech.text}'")

    pipeline.on_result(on_result)
    await pipeline.start()

    SR = 16_000
    for i in range(6):
        chunk = AudioChunk(
            samples=np.zeros(SR * 2, dtype=np.float32),
            sample_rate=SR,
            source_id="enterprise_mic",
        )
        await pipeline.process(chunk)
        await asyncio.sleep(0.01)

    await pipeline.stop()

    print(f"\n  Total keywords detected: {len(keywords_detected)}")
    print(f"  Keywords: {keywords_detected}\n")


# ---------------------------------------------------------------------------
# 2. Alerting system
# ---------------------------------------------------------------------------

async def demo_alerting() -> None:
    """
    Shows how to subscribe to AuralEvents and route them to an alert system.
    """
    from ai_ear.analyzers.environment import EnvironmentAnalyzer
    from ai_ear.core.memory import AuralMemory
    from ai_ear.core.models import AudioChunk, AuralEventType
    from ai_ear.core.pipeline import AudioPipeline
    from ai_ear.utils.audio import generate_tone

    print("\n🚨  Enterprise Integration Demo: Real-Time Alerting\n")

    class AlertSystem:
        """Simulates routing events to PagerDuty / Slack / SIEM etc."""

        def __init__(self) -> None:
            self.alerts: list[dict] = []

        async def handle_event(self, event) -> None:
            if event.severity >= 0.5:
                alert = {
                    "timestamp": event.timestamp,
                    "type": event.event_type.value,
                    "description": event.description,
                    "severity": event.severity,
                    "routing": "P1_PAGERDUTY" if event.severity >= 0.8 else "SLACK_CHANNEL",
                }
                self.alerts.append(alert)
                print(f"  🚨 [{alert['routing']}] {alert['description']} (severity={event.severity:.1f})")
            else:
                print(f"  ℹ️  [{event.event_type.value}] {event.description}")

    alerts = AlertSystem()
    memory = AuralMemory()
    pipeline = AudioPipeline(
        analyzers=[EnvironmentAnalyzer(sample_rate=16_000)],
        memory=memory,
    )
    pipeline.on_event(alerts.handle_event)
    await pipeline.start()

    SR = 16_000

    # Simulate environment changes
    chunks = [
        AudioChunk(samples=np.zeros(SR * 2, dtype=np.float32), sample_rate=SR, source_id="office_mic"),
        AudioChunk(samples=generate_tone(440.0, 2.0, SR), sample_rate=SR, source_id="office_mic"),
        AudioChunk(samples=np.zeros(SR * 2, dtype=np.float32), sample_rate=SR, source_id="office_mic"),
    ]

    for chunk in chunks:
        await pipeline._process_and_dispatch(chunk)
        await asyncio.sleep(0.05)

    await pipeline.stop()
    print(f"\n  Total high-severity alerts routed: {len(alerts.alerts)}\n")


# ---------------------------------------------------------------------------
# 3. LLM context injection
# ---------------------------------------------------------------------------

async def demo_llm_prompt() -> None:
    """
    Shows how to use AuralMemory.context_summary() as an LLM system prompt.
    """
    from ai_ear.analyzers.environment import EnvironmentAnalyzer
    from ai_ear.core.memory import AuralMemory
    from ai_ear.core.models import AudioChunk, EmotionLabel, EmotionProfile, SpeechSegment
    from ai_ear.core.pipeline import AudioPipeline

    print("\n🤖  Enterprise Integration Demo: LLM Context Injection\n")

    memory = AuralMemory(context_window_s=120)
    pipeline = AudioPipeline(
        analyzers=[EnvironmentAnalyzer(sample_rate=16_000)],
        memory=memory,
    )
    await pipeline.start()

    SR = 16_000

    # Simulate a meeting with some recorded speech
    from ai_ear.core.models import AnalysisResult, EnvironmentLabel, EnvironmentSnapshot
    for i, speech in enumerate(["Good morning everyone", "Let's discuss Q4 results", "Sales are up 15%"]):
        r = AnalysisResult(
            chunk_id=f"meeting_{i}",
            timestamp=time.time() - (30 - i * 10),
            source_id="meeting_room",
            speech=SpeechSegment(text=speech, confidence=0.92),
            emotion=EmotionProfile(dominant=EmotionLabel.CALM),
            environment=EnvironmentSnapshot(dominant=EnvironmentLabel.OFFICE),
            semantic_tags=["contains_speech", "emotion:calm", "env:office"],
        )
        await memory.store_result(r)

    await pipeline.stop()

    summary = memory.context_summary()
    system_prompt = _build_system_prompt(summary)

    print("  Generated LLM System Prompt:")
    print("  " + "─" * 60)
    for line in system_prompt.split("\n"):
        print(f"  {line}")
    print("  " + "─" * 60)
    print()


def _build_system_prompt(summary: dict) -> str:
    """Construct a rich system prompt from an AuralMemory context summary."""
    lines = [
        "You are an AI assistant with real-time acoustic awareness.",
        f"You have been listening for the last {summary['window_s']:.0f} seconds.",
        "",
        "ACOUSTIC CONTEXT:",
    ]
    if summary["transcript"]:
        lines.append(f"  Transcribed speech: \"{summary['transcript']}\"")
    else:
        lines.append("  No speech detected recently.")

    if summary["dominant_emotions"]:
        top_emotion = summary["dominant_emotions"][0][0]
        lines.append(f"  Prevailing emotional tone: {top_emotion}")

    if summary["dominant_environments"]:
        top_env = summary["dominant_environments"][0][0]
        lines.append(f"  Acoustic environment: {top_env}")

    if summary["music_detected"]:
        lines.append("  Background music detected.")

    if summary["events"]:
        lines.append(f"  Notable events: {len(summary['events'])} detected.")

    lines += [
        "",
        "Respond naturally, taking this acoustic context into account.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. Programmatic API server
# ---------------------------------------------------------------------------

def demo_serve() -> None:
    """Start the AI Ear REST + WebSocket API server."""
    import uvicorn

    from ai_ear.api.server import create_app
    from ai_ear.core.config import Settings

    print("\n🌐  Starting AI Ear API Server on http://0.0.0.0:8080\n")
    print("  Endpoints:")
    print("    GET  /health          — liveness probe")
    print("    GET  /info            — configuration summary")
    print("    POST /analyse         — analyse an uploaded audio file")
    print("    GET  /memory/context  — aural context summary")
    print("    GET  /memory/transcript — recent speech transcript")
    print("    GET  /memory/events   — recent aural events")
    print("    WS   /stream          — real-time WebSocket audio stream")
    print("    GET  /pipeline/stats  — pipeline throughput stats")
    print()

    settings = Settings()
    app = create_app(settings)
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="AI Ear enterprise integration examples")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("custom-analyser", help="BYOM custom analyser demo")
    sub.add_parser("alerting", help="Real-time alerting demo")
    sub.add_parser("llm-prompt", help="LLM context injection demo")
    sub.add_parser("serve", help="Start the API server")
    args = parser.parse_args()

    if args.command == "custom-analyser":
        asyncio.run(demo_custom_analyser())
    elif args.command == "alerting":
        asyncio.run(demo_alerting())
    elif args.command == "llm-prompt":
        asyncio.run(demo_llm_prompt())
    elif args.command == "serve":
        demo_serve()
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
