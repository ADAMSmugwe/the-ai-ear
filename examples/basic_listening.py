"""
basic_listening.py — minimal end-to-end example of the AI Ear.

Demonstrates how to wire together an AudioListener (file-based in this
example), the AudioPipeline, and the AuralMemory to produce a running
transcript + context summary from a WAV file.

Usage
-----
    python examples/basic_listening.py path/to/audio.wav

    # Or generate synthetic test audio on-the-fly (no file required):
    python examples/basic_listening.py --demo
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np


async def run_demo() -> None:
    """Synthesise 10 seconds of test tones and demonstrate the pipeline."""
    from ai_ear.analyzers.environment import EnvironmentAnalyzer
    from ai_ear.analyzers.music import MusicAnalyzer
    from ai_ear.core.memory import AuralMemory
    from ai_ear.core.models import AudioChunk
    from ai_ear.core.pipeline import AudioPipeline
    from ai_ear.utils.audio import generate_tone, generate_noise

    print("\n🎧  The AI Ear — Demo Mode\n")

    memory = AuralMemory(context_window_s=60)
    pipeline = AudioPipeline(
        analyzers=[
            EnvironmentAnalyzer(sample_rate=16_000),
            MusicAnalyzer(sample_rate=16_000),
        ],
        memory=memory,
    )

    await pipeline.start()

    SR = 16_000
    results_seen = 0

    async def on_result(result):
        nonlocal results_seen
        results_seen += 1
        env = result.environment.dominant.value if result.environment else "?"
        tags = ", ".join(result.semantic_tags) or "(none)"
        print(f"  chunk {results_seen:>3}  env={env:<12}  tags={tags}")

    pipeline.on_result(on_result)

    # Feed 5 synthetic chunks
    chunks = [
        # Chunk 1: silence
        AudioChunk(
            samples=np.zeros(SR * 2, dtype=np.float32),
            sample_rate=SR,
            source_id="demo",
        ),
        # Chunk 2: tone (music-like)
        AudioChunk(
            samples=generate_tone(440.0, 2.0, SR, amplitude=0.6),
            sample_rate=SR,
            source_id="demo",
        ),
        # Chunk 3: noise (ambient)
        AudioChunk(
            samples=generate_noise(2.0, SR, amplitude=0.3),
            sample_rate=SR,
            source_id="demo",
        ),
        # Chunk 4: mixed tone + noise
        AudioChunk(
            samples=generate_tone(880.0, 2.0, SR, amplitude=0.4)
            + generate_noise(2.0, SR, amplitude=0.05),
            sample_rate=SR,
            source_id="demo",
        ),
        # Chunk 5: silence again
        AudioChunk(
            samples=np.zeros(SR * 2, dtype=np.float32),
            sample_rate=SR,
            source_id="demo",
        ),
    ]

    print("  Processing 5 synthetic audio chunks…\n")
    for chunk in chunks:
        result = await pipeline.process(chunk)
        await on_result(result)

    await pipeline.stop()

    print("\n📋  Context Summary (last 60 seconds):\n")
    summary = memory.context_summary()
    print(f"  Transcript       : {summary['transcript'] or '(no speech detected)'}")
    print(f"  Dominant emotions: {summary['dominant_emotions'] or '(none)'}")
    print(f"  Environments     : {summary['dominant_environments']}")
    print(f"  Music detected   : {summary['music_detected']}")
    print(f"  Semantic tags    : {summary['semantic_tags']}")
    print(f"  Events           : {len(summary['events'])} detected")
    print()


async def run_file(path: Path) -> None:
    """Process a real audio file through the pipeline."""
    from ai_ear.analyzers.environment import EnvironmentAnalyzer
    from ai_ear.analyzers.music import MusicAnalyzer
    from ai_ear.core.listener import AudioListener
    from ai_ear.core.memory import AuralMemory
    from ai_ear.core.pipeline import AudioPipeline

    print(f"\n🎧  The AI Ear — Processing: {path.name}\n")

    memory = AuralMemory(context_window_s=300)
    pipeline = AudioPipeline(
        analyzers=[
            EnvironmentAnalyzer(sample_rate=16_000),
            MusicAnalyzer(sample_rate=16_000),
        ],
        memory=memory,
    )
    await pipeline.start()

    listener = AudioListener(sample_rate=16_000, chunk_duration_s=2.0, source_id=path.name)
    count = 0

    async for chunk in listener.ingest_file(path):
        result = await pipeline.process(chunk)
        count += 1
        env = result.environment.dominant.value if result.environment else "?"
        music = "🎵" if (result.music and result.music.is_music) else "  "
        print(f"  [{count:>4}] {music} env={env}")

    await pipeline.stop()

    print(f"\n  Processed {count} chunks from {path.name}\n")
    print("📋  Summary:")
    summary = memory.context_summary()
    print(f"  Environments   : {summary['dominant_environments']}")
    print(f"  Music detected : {summary['music_detected']}")
    print(f"  Tags           : {summary['semantic_tags']}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="AI Ear basic listening demo")
    parser.add_argument("audio_file", nargs="?", help="Path to audio file (WAV/FLAC/OGG)")
    parser.add_argument("--demo", action="store_true", help="Run synthetic demo (no audio file needed)")
    args = parser.parse_args()

    if args.demo or args.audio_file is None:
        asyncio.run(run_demo())
    else:
        path = Path(args.audio_file)
        if not path.exists():
            print(f"Error: file not found: {path}", file=sys.stderr)
            return 1
        asyncio.run(run_file(path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
