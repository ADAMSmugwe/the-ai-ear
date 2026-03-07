"""
EmotionAnalyzer — vocal emotion recognition.

Uses a pre-trained wav2vec2-based speech emotion recognition model from
HuggingFace to infer the speaker's emotional state from raw audio.

The analyser extracts:
* Discrete emotion label (happy, sad, angry, neutral, …)
* Per-class probability scores
* Continuous arousal / valence estimates derived from the discrete scores

Graceful fallback
-----------------
If ``transformers`` or ``torch`` are not installed, the analyser returns a
neutral emotion profile with confidence 0.
"""

from __future__ import annotations

import asyncio
import logging
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ai_ear.analyzers.base import BaseAnalyzer, EmotionResult
from ai_ear.core.models import AudioChunk, EmotionLabel, EmotionProfile

log = logging.getLogger(__name__)

# Map HuggingFace label strings to our EmotionLabel enum
_LABEL_MAP: dict[str, EmotionLabel] = {
    "neutral": EmotionLabel.NEUTRAL,
    "happy": EmotionLabel.HAPPY,
    "happiness": EmotionLabel.HAPPY,
    "joy": EmotionLabel.HAPPY,
    "sad": EmotionLabel.SAD,
    "sadness": EmotionLabel.SAD,
    "angry": EmotionLabel.ANGRY,
    "anger": EmotionLabel.ANGRY,
    "fear": EmotionLabel.FEARFUL,
    "fearful": EmotionLabel.FEARFUL,
    "disgust": EmotionLabel.DISGUSTED,
    "disgusted": EmotionLabel.DISGUSTED,
    "surprised": EmotionLabel.SURPRISED,
    "surprise": EmotionLabel.SURPRISED,
    "calm": EmotionLabel.CALM,
}

# Arousal and valence anchors for known emotions
_AROUSAL: dict[EmotionLabel, float] = {
    EmotionLabel.NEUTRAL: 0.5,
    EmotionLabel.HAPPY: 0.7,
    EmotionLabel.SAD: 0.2,
    EmotionLabel.ANGRY: 0.9,
    EmotionLabel.FEARFUL: 0.8,
    EmotionLabel.DISGUSTED: 0.6,
    EmotionLabel.SURPRISED: 0.8,
    EmotionLabel.CALM: 0.1,
}

_VALENCE: dict[EmotionLabel, float] = {
    EmotionLabel.NEUTRAL: 0.5,
    EmotionLabel.HAPPY: 0.9,
    EmotionLabel.SAD: 0.1,
    EmotionLabel.ANGRY: 0.1,
    EmotionLabel.FEARFUL: 0.2,
    EmotionLabel.DISGUSTED: 0.2,
    EmotionLabel.SURPRISED: 0.6,
    EmotionLabel.CALM: 0.7,
}


class EmotionAnalyzer(BaseAnalyzer):
    """
    Speech emotion recognition via HuggingFace Transformers.

    Parameters
    ----------
    model_id:
        HuggingFace model identifier.
    device:
        PyTorch device string.
    sample_rate:
        Expected input sample rate (the model requires 16 kHz).
    """

    name = "emotion"

    def __init__(
        self,
        model_id: str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        device: str = "cpu",
        sample_rate: int = 16_000,
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._sample_rate = sample_rate
        self._pipeline: object | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="emotion")

    async def load(self) -> None:
        log.info("Loading emotion model '%s'…", self._model_id)
        loop = asyncio.get_running_loop()
        self._pipeline = await loop.run_in_executor(self._executor, self._load_sync)
        if self._pipeline:
            log.info("Emotion model loaded")

    def _load_sync(self) -> object:
        try:
            from transformers import pipeline  # type: ignore[import-untyped]
            return pipeline(
                "audio-classification",
                model=self._model_id,
                device=self._device,
            )
        except Exception:
            log.warning("Emotion model could not be loaded – emotion analysis disabled")
            return None

    async def unload(self) -> None:
        self._pipeline = None
        self._executor.shutdown(wait=False)

    async def analyse(self, chunk: AudioChunk) -> EmotionResult:
        if self._pipeline is None:
            return EmotionResult(
                profile=EmotionProfile(dominant=EmotionLabel.NEUTRAL),
                confidence=0.0,
            )
        loop = asyncio.get_running_loop()
        profile = await loop.run_in_executor(
            self._executor, self._predict_sync, chunk.samples, chunk.sample_rate
        )
        return EmotionResult(profile=profile, confidence=max(profile.scores.values(), default=0.0))

    def _predict_sync(self, samples: np.ndarray, sample_rate: int) -> EmotionProfile:
        try:
            audio = samples.astype(np.float32)
            if sample_rate != 16_000:
                try:
                    import librosa  # type: ignore[import-untyped]
                    audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16_000)
                except ImportError:
                    pass

            results = self._pipeline({"raw": audio, "sampling_rate": 16_000})  # type: ignore[operator]
            scores: dict[str, float] = {r["label"].lower(): r["score"] for r in results}
            top_label_raw = max(scores, key=scores.__getitem__)
            dominant = _LABEL_MAP.get(top_label_raw, EmotionLabel.NEUTRAL)

            # Compute weighted arousal / valence
            arousal = sum(
                scores.get(k, 0.0) * _AROUSAL.get(_LABEL_MAP.get(k, EmotionLabel.NEUTRAL), 0.5)
                for k in scores
            )
            valence = sum(
                scores.get(k, 0.0) * _VALENCE.get(_LABEL_MAP.get(k, EmotionLabel.NEUTRAL), 0.5)
                for k in scores
            )

            return EmotionProfile(
                dominant=dominant,
                scores=scores,
                arousal=min(1.0, max(0.0, arousal)),
                valence=min(1.0, max(0.0, valence)),
            )
        except Exception:
            log.exception("Emotion prediction failed")
            return EmotionProfile(dominant=EmotionLabel.NEUTRAL)
