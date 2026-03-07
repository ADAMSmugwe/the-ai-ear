"""
EnvironmentAnalyzer — acoustic scene classification.

Classifies the acoustic environment of an audio chunk using signal-processing
features (RMS energy, zero-crossing rate, spectral centroid) combined with
heuristic rules.  This lightweight approach runs in microseconds per chunk
with no GPU requirement.

For production deployments with higher accuracy requirements, the analyser
can be swapped for a YAMNet / PANNs-based DNN classifier by subclassing
``EnvironmentAnalyzer`` and overriding ``_classify_sync``.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ai_ear.analyzers.base import BaseAnalyzer, EnvironmentResult
from ai_ear.core.models import AudioChunk, EnvironmentLabel, EnvironmentSnapshot
from ai_ear.utils.audio import (
    rms_db,
    zero_crossing_rate,
    spectral_centroid_hz,
    spectral_flatness,
)

log = logging.getLogger(__name__)


class EnvironmentAnalyzer(BaseAnalyzer):
    """
    Fast, heuristic-based acoustic environment classifier.

    Parameters
    ----------
    sample_rate:
        Expected input sample rate.
    noise_gate_db:
        Frames below this RMS level are classified as silence.
    """

    name = "environment"

    def __init__(
        self,
        sample_rate: int = 16_000,
        noise_gate_db: float = -50.0,
    ) -> None:
        self._sample_rate = sample_rate
        self._noise_gate_db = noise_gate_db
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="env")

    async def load(self) -> None:
        log.debug("EnvironmentAnalyzer ready (heuristic mode)")

    async def unload(self) -> None:
        self._executor.shutdown(wait=False)

    async def analyse(self, chunk: AudioChunk) -> EnvironmentResult:
        loop = asyncio.get_running_loop()
        snapshot = await loop.run_in_executor(
            self._executor, self._classify_sync, chunk.samples, chunk.sample_rate
        )
        top_score = max(snapshot.scores.values(), default=0.0)
        return EnvironmentResult(snapshot=snapshot, confidence=top_score)

    def _classify_sync(self, samples: np.ndarray, sample_rate: int) -> EnvironmentSnapshot:
        # ----------------------------------------------------------------
        # Feature extraction
        # ----------------------------------------------------------------
        energy_db = rms_db(samples)
        zcr = zero_crossing_rate(samples)
        sc_hz = spectral_centroid_hz(samples, sample_rate)
        sf = spectral_flatness(samples)

        # Estimated noise floor: 10th percentile of frame energies
        frame_size = 512
        frames = [
            samples[i : i + frame_size]
            for i in range(0, len(samples) - frame_size, frame_size)
        ]
        if frames:
            frame_energies = [rms_db(f) for f in frames]
            noise_floor = float(np.percentile(frame_energies, 10))
        else:
            noise_floor = energy_db

        snr = max(0.0, energy_db - noise_floor)

        # ----------------------------------------------------------------
        # Heuristic classification
        # ----------------------------------------------------------------
        scores: dict[str, float] = {label.value: 0.0 for label in EnvironmentLabel}

        if energy_db < self._noise_gate_db:
            scores[EnvironmentLabel.SILENCE.value] = 1.0
            dominant = EnvironmentLabel.SILENCE
        else:
            # Speech heuristics: mid ZCR, mid spectral centroid, low flatness
            speech_score = _sigmoid(zcr, 0.05, 30) * _sigmoid_inv(sf, 0.4, 10)
            # Music heuristics: low ZCR, high spectral centroid, moderate flatness
            music_score = _sigmoid_inv(zcr, 0.1, 30) * _sigmoid(sc_hz, 1500, 0.001)
            # Alarm: high energy + high ZCR + narrow band
            alarm_score = _sigmoid(energy_db, -20, 0.5) * _sigmoid(zcr, 0.15, 20) * _sigmoid_inv(sf, 0.5, 5)
            # Crowd: high energy + moderate ZCR + high flatness
            crowd_score = _sigmoid(energy_db, -30, 0.5) * _sigmoid(sf, 0.5, 5)
            # Traffic: continuous low-mid noise
            traffic_score = _sigmoid(energy_db, -35, 0.2) * _sigmoid_inv(zcr, 0.05, 40) * _sigmoid(sf, 0.6, 5)

            raw = {
                EnvironmentLabel.SPEECH: speech_score,
                EnvironmentLabel.MUSIC: music_score,
                EnvironmentLabel.ALARM: alarm_score,
                EnvironmentLabel.CROWD: crowd_score,
                EnvironmentLabel.TRAFFIC: traffic_score,
                EnvironmentLabel.UNKNOWN: 0.1,
            }

            total = sum(raw.values()) or 1.0
            for label, score in raw.items():
                scores[label.value] = score / total

            dominant = max(raw, key=raw.__getitem__)

        return EnvironmentSnapshot(
            dominant=dominant,
            scores=scores,
            noise_floor_db=noise_floor,
            snr_db=snr,
        )


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: float, midpoint: float, k: float) -> float:
    """Logistic sigmoid centred at ``midpoint`` with steepness ``k``."""
    try:
        return 1.0 / (1.0 + float(np.exp(-k * (x - midpoint))))
    except OverflowError:
        return 0.0


def _sigmoid_inv(x: float, midpoint: float, k: float) -> float:
    return 1.0 - _sigmoid(x, midpoint, k)
