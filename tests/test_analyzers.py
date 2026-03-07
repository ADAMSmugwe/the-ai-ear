"""
Tests for the environment and music analysers using synthetic audio.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from ai_ear.analyzers.environment import EnvironmentAnalyzer
from ai_ear.analyzers.music import MusicAnalyzer, _estimate_key
from ai_ear.core.models import AudioChunk, EnvironmentLabel
from ai_ear.utils.audio import generate_noise, generate_silence, generate_tone

SR = 16_000


def _chunk(samples: np.ndarray) -> AudioChunk:
    return AudioChunk(samples=samples, sample_rate=SR)


# ---------------------------------------------------------------------------
# EnvironmentAnalyzer
# ---------------------------------------------------------------------------

class TestEnvironmentAnalyzer:
    @pytest.mark.asyncio
    async def test_silence_classified_as_silence(self):
        analyzer = EnvironmentAnalyzer(sample_rate=SR, noise_gate_db=-50.0)
        await analyzer.load()
        silence = generate_silence(2.0, SR)
        result = await analyzer.analyse(_chunk(silence))
        assert result.snapshot.dominant == EnvironmentLabel.SILENCE

    @pytest.mark.asyncio
    async def test_result_has_valid_scores(self):
        analyzer = EnvironmentAnalyzer(sample_rate=SR)
        await analyzer.load()
        tone = generate_tone(440.0, 2.0, SR)
        result = await analyzer.analyse(_chunk(tone))
        total = sum(result.snapshot.scores.values())
        assert abs(total - 1.0) < 0.01  # scores sum to ~1

    @pytest.mark.asyncio
    async def test_dominant_in_scores(self):
        analyzer = EnvironmentAnalyzer(sample_rate=SR)
        await analyzer.load()
        tone = generate_tone(1000.0, 2.0, SR)
        result = await analyzer.analyse(_chunk(tone))
        dominant_val = result.snapshot.dominant.value
        assert dominant_val in result.snapshot.scores

    @pytest.mark.asyncio
    async def test_noise_floor_below_signal(self):
        analyzer = EnvironmentAnalyzer(sample_rate=SR)
        await analyzer.load()
        tone = generate_tone(440.0, 2.0, SR, amplitude=0.5)
        result = await analyzer.analyse(_chunk(tone))
        assert result.snapshot.noise_floor_db <= 0.0

    @pytest.mark.asyncio
    async def test_snr_non_negative(self):
        analyzer = EnvironmentAnalyzer(sample_rate=SR)
        await analyzer.load()
        noise = generate_noise(2.0, SR, amplitude=0.2)
        result = await analyzer.analyse(_chunk(noise))
        assert result.snapshot.snr_db >= 0.0

    @pytest.mark.asyncio
    async def test_confidence_in_valid_range(self):
        analyzer = EnvironmentAnalyzer(sample_rate=SR)
        await analyzer.load()
        tone = generate_tone(440.0, 2.0, SR)
        result = await analyzer.analyse(_chunk(tone))
        assert 0.0 <= result.confidence <= 1.0


# ---------------------------------------------------------------------------
# MusicAnalyzer
# ---------------------------------------------------------------------------

class TestMusicAnalyzer:
    @pytest.mark.asyncio
    async def test_silence_not_music(self):
        analyzer = MusicAnalyzer(sample_rate=SR)
        await analyzer.load()
        silence = generate_silence(2.0, SR)
        result = await analyzer.analyse(_chunk(silence))
        assert not result.profile.is_music

    @pytest.mark.asyncio
    async def test_energy_normalised(self):
        analyzer = MusicAnalyzer(sample_rate=SR)
        await analyzer.load()
        tone = generate_tone(440.0, 2.0, SR, amplitude=0.5)
        result = await analyzer.analyse(_chunk(tone))
        assert 0.0 <= result.profile.energy <= 1.0

    @pytest.mark.asyncio
    async def test_returns_music_result_type(self):
        from ai_ear.analyzers.base import MusicResult
        analyzer = MusicAnalyzer(sample_rate=SR)
        await analyzer.load()
        tone = generate_tone(440.0, 2.0, SR)
        result = await analyzer.analyse(_chunk(tone))
        assert isinstance(result, MusicResult)

    def test_estimate_key_returns_string(self):
        chroma = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], dtype=float)
        key = _estimate_key(chroma)
        assert isinstance(key, str)
        assert "major" in key or "minor" in key

    def test_estimate_key_c_major(self):
        # Perfect C-major chroma
        chroma = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], dtype=float)
        key = _estimate_key(chroma)
        assert key == "C major"

    def test_estimate_key_g_major(self):
        # G major scale: G A B C D E F# = indices 7, 9, 11, 0, 2, 4, 6
        g_major = np.zeros(12, dtype=float)
        for idx in [7, 9, 11, 0, 2, 4, 6]:
            g_major[idx] = 1.0
        # Weight G (root) heavily so it beats C major unambiguously
        g_major[7] = 3.0
        key = _estimate_key(g_major)
        assert key == "G major"
