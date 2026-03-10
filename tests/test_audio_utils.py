"""
Tests for ai_ear.utils.audio — pure signal-processing utilities.

All tests are dependency-free (numpy only).
"""

import numpy as np
import pytest

from ai_ear.utils.audio import (
    generate_noise,
    generate_silence,
    generate_tone,
    rms_db,
    spectral_centroid_hz,
    spectral_flatness,
    zero_crossing_rate,
)

SR = 16_000


# ---------------------------------------------------------------------------
# Test generators
# ---------------------------------------------------------------------------

class TestGenerators:
    def test_generate_tone_shape(self):
        tone = generate_tone(440.0, duration_s=1.0, sample_rate=SR)
        assert tone.shape == (SR,)
        assert tone.dtype == np.float32

    def test_generate_tone_amplitude(self):
        tone = generate_tone(440.0, duration_s=1.0, sample_rate=SR, amplitude=0.5)
        assert float(np.max(np.abs(tone))) == pytest.approx(0.5, abs=0.01)

    def test_generate_silence(self):
        silence = generate_silence(duration_s=0.5, sample_rate=SR)
        assert silence.shape == (SR // 2,)
        assert np.all(silence == 0.0)

    def test_generate_noise_shape(self):
        noise = generate_noise(duration_s=1.0, sample_rate=SR, amplitude=0.1)
        assert noise.shape == (SR,)

    def test_generate_noise_amplitude(self):
        rng = np.random.default_rng(42)
        noise = generate_noise(duration_s=1.0, sample_rate=SR, amplitude=0.1, rng=rng)
        assert float(np.max(np.abs(noise))) == pytest.approx(0.1, abs=1e-6)

    def test_generate_noise_reproducible(self):
        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)
        n1 = generate_noise(rng=rng1)
        n2 = generate_noise(rng=rng2)
        np.testing.assert_array_equal(n1, n2)


# ---------------------------------------------------------------------------
# Test feature functions
# ---------------------------------------------------------------------------

class TestRmsDb:
    def test_silence_returns_minimum(self):
        silence = generate_silence(1.0, SR)
        assert rms_db(silence) == -120.0

    def test_full_scale_sine(self):
        tone = generate_tone(440.0, 1.0, SR, amplitude=1.0)
        db = rms_db(tone)
        # Full-scale sine RMS ≈ -3 dBFS
        assert -5.0 < db < 0.0

    def test_half_amplitude(self):
        tone_full = generate_tone(440.0, 1.0, SR, amplitude=1.0)
        tone_half = generate_tone(440.0, 1.0, SR, amplitude=0.5)
        # Halving amplitude → -6 dB
        assert rms_db(tone_full) - rms_db(tone_half) == pytest.approx(6.0, abs=0.2)


class TestZeroCrossingRate:
    def test_silence_zcr_is_zero(self):
        silence = generate_silence(1.0, SR)
        assert zero_crossing_rate(silence) == 0.0

    def test_high_frequency_zcr_is_high(self):
        # 4 kHz tone crosses zero ~8000 times per second at 16kHz SR
        tone = generate_tone(4000.0, 1.0, SR)
        zcr = zero_crossing_rate(tone)
        assert zcr > 0.3

    def test_low_frequency_zcr_is_low(self):
        tone = generate_tone(50.0, 1.0, SR)
        zcr = zero_crossing_rate(tone)
        assert zcr < 0.02

    def test_short_signal(self):
        # Single sample should not raise
        assert zero_crossing_rate(np.array([0.5], dtype=np.float32)) == 0.0


class TestSpectralCentroid:
    def test_silence_centroid_is_zero(self):
        silence = generate_silence(0.1, SR)
        assert spectral_centroid_hz(silence, SR) == 0.0

    def test_centroid_increases_with_frequency(self):
        low = spectral_centroid_hz(generate_tone(200.0, 1.0, SR), SR)
        high = spectral_centroid_hz(generate_tone(4000.0, 1.0, SR), SR)
        assert high > low

    def test_centroid_near_tone_frequency(self):
        freq = 1000.0
        tone = generate_tone(freq, 1.0, SR)
        centroid = spectral_centroid_hz(tone, SR)
        # Centroid should be close to the tone frequency
        assert abs(centroid - freq) < freq * 0.1  # within 10%


class TestSpectralFlatness:
    def test_pure_tone_is_low_flatness(self):
        tone = generate_tone(440.0, 1.0, SR)
        sf = spectral_flatness(tone)
        assert sf < 0.1

    def test_white_noise_is_high_flatness(self):
        rng = np.random.default_rng(7)
        noise = generate_noise(1.0, SR, amplitude=0.5, rng=rng)
        sf = spectral_flatness(noise)
        assert sf > 0.3

    def test_silence_does_not_raise(self):
        silence = generate_silence(0.5, SR)
        sf = spectral_flatness(silence)
        assert 0.0 <= sf <= 1.0
