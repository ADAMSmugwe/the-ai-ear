"""
MusicAnalyzer — real-time music characterisation.

Extracts musical features from audio using librosa:
* Tempo (BPM) via beat tracking
* Estimated musical key via chroma vector analysis
* Spectral energy and centroid (timbral brightness)
* Genre hints derived from spectral features

This analyser purposefully avoids heavyweight music classification models to
stay lightweight.  The ``is_music`` flag is set when the spectral profile
matches known music characteristics (harmonic content, regular beat structure).

librosa is an optional dependency; when absent, the analyser returns a
placeholder profile with ``is_music=False``.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ai_ear.analyzers.base import BaseAnalyzer, MusicResult
from ai_ear.core.models import AudioChunk, MusicProfile
from ai_ear.utils.audio import rms_db, spectral_centroid_hz, spectral_flatness

log = logging.getLogger(__name__)

_CHROMATIC_NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Major / minor scale interval patterns (semitones from root)
_MAJOR_PATTERN = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], dtype=float)
_MINOR_PATTERN = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0], dtype=float)


class MusicAnalyzer(BaseAnalyzer):
    """
    Librosa-based music characterisation analyser.

    Parameters
    ----------
    sample_rate:
        Expected input sample rate.
    energy_threshold:
        Minimum spectral energy (normalised) to consider a frame as music.
    """

    name = "music"

    def __init__(
        self,
        sample_rate: int = 16_000,
        energy_threshold: float = 0.05,
    ) -> None:
        self._sample_rate = sample_rate
        self._energy_threshold = energy_threshold
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="music")
        self._librosa_available = False

    async def load(self) -> None:
        try:
            import librosa  # noqa: F401  # type: ignore[import-untyped]
            self._librosa_available = True
            log.debug("MusicAnalyzer: librosa available")
        except ImportError:
            log.warning("librosa not installed – music analysis will be limited")

    async def unload(self) -> None:
        self._executor.shutdown(wait=False)

    async def analyse(self, chunk: AudioChunk) -> MusicResult:
        loop = asyncio.get_running_loop()
        profile = await loop.run_in_executor(
            self._executor, self._analyse_sync, chunk.samples, chunk.sample_rate
        )
        return MusicResult(profile=profile, confidence=0.8 if profile.is_music else 0.5)

    def _analyse_sync(self, samples: np.ndarray, sample_rate: int) -> MusicProfile:
        audio = samples.astype(np.float32)

        # Quick check: is there enough signal?
        energy_db = rms_db(audio)
        if energy_db < -55:
            return MusicProfile(is_music=False, energy=0.0)

        sc_hz = spectral_centroid_hz(audio, sample_rate)
        sf = spectral_flatness(audio)

        # Energy normalised to [0, 1]
        energy_norm = float(np.clip((energy_db + 60) / 60, 0.0, 1.0))

        if not self._librosa_available:
            # Heuristic fallback: high spectral centroid + moderate flatness → music
            is_music = sc_hz > 800 and 0.1 < sf < 0.7 and energy_norm > self._energy_threshold
            return MusicProfile(
                is_music=is_music,
                spectral_centroid_hz=sc_hz,
                energy=energy_norm,
            )

        import librosa  # type: ignore[import-untyped]

        # ----------------------------------------------------------------
        # Tempo
        # ----------------------------------------------------------------
        try:
            tempo_arr, _ = librosa.beat.beat_track(y=audio, sr=sample_rate)
            tempo_bpm = float(tempo_arr[0]) if hasattr(tempo_arr, "__len__") else float(tempo_arr)
        except Exception:
            tempo_bpm = None

        # ----------------------------------------------------------------
        # Key estimation via chroma
        # ----------------------------------------------------------------
        key_str: str | None = None
        try:
            chroma = librosa.feature.chroma_cqt(y=audio, sr=sample_rate)
            chroma_mean = chroma.mean(axis=1)  # (12,)
            key_str = _estimate_key(chroma_mean)
        except Exception:
            pass

        # ----------------------------------------------------------------
        # Is-music heuristics: harmonic richness + beat regularity
        # ----------------------------------------------------------------
        try:
            harmonic, _ = librosa.effects.hpss(audio)
            harmonic_ratio = float(np.sqrt(np.mean(harmonic**2)) / (np.sqrt(np.mean(audio**2)) + 1e-8))
        except Exception:
            harmonic_ratio = 0.0

        is_music = (
            energy_norm > self._energy_threshold
            and harmonic_ratio > 0.3
            and sc_hz > 300
        )

        # ----------------------------------------------------------------
        # Genre hints based on tempo + centroid
        # ----------------------------------------------------------------
        genre_hints: list[str] = []
        if is_music and tempo_bpm is not None:
            if tempo_bpm > 140:
                genre_hints.append("electronic/dance")
            elif tempo_bpm > 110:
                genre_hints.append("pop/rock")
            elif tempo_bpm > 80:
                genre_hints.append("r&b/hip-hop")
            else:
                genre_hints.append("classical/jazz")

        return MusicProfile(
            is_music=is_music,
            tempo_bpm=tempo_bpm,
            key=key_str,
            energy=energy_norm,
            spectral_centroid_hz=sc_hz,
            genre_hints=genre_hints,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _estimate_key(chroma_mean: np.ndarray) -> str:
    """
    Estimate the musical key from a 12-bin chroma vector using template matching.

    When two keys have the same template-match score (e.g. relative major/minor
    pairs which share the same notes), the key whose *root* note has the highest
    energy in the chroma is preferred — a musically sound tiebreaker.
    """
    best_score = -1.0
    best_key = "C major"
    best_root_energy = -1.0

    for root in range(12):
        major_template = np.roll(_MAJOR_PATTERN, root)
        minor_template = np.roll(_MINOR_PATTERN, root)
        major_score = float(np.dot(chroma_mean, major_template))
        minor_score = float(np.dot(chroma_mean, minor_template))
        root_energy = float(chroma_mean[root])

        if major_score > best_score or (
            major_score == best_score and root_energy > best_root_energy
        ):
            best_score = major_score
            best_key = f"{_CHROMATIC_NOTES[root]} major"
            best_root_energy = root_energy

        if minor_score > best_score or (
            minor_score == best_score and root_energy > best_root_energy
        ):
            best_score = minor_score
            best_key = f"{_CHROMATIC_NOTES[root]} minor"
            best_root_energy = root_energy

    return best_key
