"""
Audio signal-processing utilities.

Pure numpy/scipy implementations used throughout the analyser stack.
These functions are intentionally dependency-light: only numpy is required.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def rms_db(samples: np.ndarray) -> float:
    """
    Compute the RMS energy of ``samples`` in dBFS.

    Returns
    -------
    float
        dBFS value (≤ 0.0).  Silence returns −120.0.
    """
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    if rms < 1e-10:
        return -120.0
    return float(20.0 * np.log10(rms))


def zero_crossing_rate(samples: np.ndarray) -> float:
    """
    Compute the zero-crossing rate (ZCR) of a 1-D signal.

    Returns
    -------
    float
        Fraction of samples where the sign changes (0–1).
    """
    arr = np.asarray(samples, dtype=np.float32)
    if len(arr) < 2:
        return 0.0
    signs = np.sign(arr)
    signs[signs == 0] = 1  # treat zero as positive
    crossings = np.sum(np.abs(np.diff(signs))) / 2
    return float(crossings / (len(arr) - 1))


def spectral_centroid_hz(samples: np.ndarray, sample_rate: int) -> float:
    """
    Compute the spectral centroid (centre of mass of the power spectrum) in Hz.

    Returns
    -------
    float
        Centroid frequency in Hz (0 for silence).
    """
    arr = np.asarray(samples, dtype=np.float32)
    if len(arr) == 0:
        return 0.0
    spectrum = np.abs(np.fft.rfft(arr))
    freqs = np.fft.rfftfreq(len(arr), d=1.0 / sample_rate)
    power = spectrum**2
    total_power = float(np.sum(power))
    if total_power < 1e-10:
        return 0.0
    return float(np.sum(freqs * power) / total_power)


def spectral_flatness(samples: np.ndarray) -> float:
    """
    Compute spectral flatness (Wiener entropy) in [0, 1].

    A value near 1 indicates white noise; near 0 indicates a tonal signal.

    Returns
    -------
    float
        Spectral flatness in [0.0, 1.0].
    """
    arr = np.asarray(samples, dtype=np.float32)
    if len(arr) == 0:
        return 0.0
    power = np.abs(np.fft.rfft(arr)) ** 2 + 1e-10
    geometric_mean = float(np.exp(np.mean(np.log(power))))
    arithmetic_mean = float(np.mean(power))
    if arithmetic_mean < 1e-10:
        return 0.0
    return float(np.clip(geometric_mean / arithmetic_mean, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Test signal generators
# ---------------------------------------------------------------------------

def generate_tone(
    frequency_hz: float = 440.0,
    duration_s: float = 1.0,
    sample_rate: int = 16_000,
    amplitude: float = 0.5,
) -> np.ndarray:
    """
    Generate a pure sine-wave tone.

    Parameters
    ----------
    frequency_hz:
        Fundamental frequency in Hz.
    duration_s:
        Duration in seconds.
    sample_rate:
        Output sample rate.
    amplitude:
        Peak amplitude in [0, 1].

    Returns
    -------
    np.ndarray
        Float32 mono PCM samples.
    """
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * frequency_hz * t)).astype(np.float32)


def generate_silence(duration_s: float = 1.0, sample_rate: int = 16_000) -> np.ndarray:
    """Generate a block of digital silence."""
    return np.zeros(int(sample_rate * duration_s), dtype=np.float32)


def generate_noise(
    duration_s: float = 1.0,
    sample_rate: int = 16_000,
    amplitude: float = 0.1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate white Gaussian noise.

    Parameters
    ----------
    rng:
        Optional numpy random generator for reproducibility.
    """
    rng = rng or np.random.default_rng()
    samples = rng.standard_normal(int(sample_rate * duration_s)).astype(np.float32)
    peak = float(np.max(np.abs(samples)))
    if peak > 0:
        samples = samples / peak * amplitude
    return samples
