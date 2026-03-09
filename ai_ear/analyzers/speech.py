"""
SpeechAnalyzer — OpenAI Whisper-based speech recognition.

Whisper is a general-purpose speech recognition model trained on 680,000 hours
of multilingual audio.  This analyser wraps the ``openai-whisper`` Python
package, running inference in a thread pool so the asyncio event loop is never
blocked.

Capabilities
------------
* Automatic language detection
* Word-level timestamps (``word_timestamps=True``)
* Graceful fallback when whisper is not installed (returns empty segment)
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ai_ear.analyzers.base import BaseAnalyzer, SpeechResult
from ai_ear.core.models import AudioChunk, SpeechSegment

log = logging.getLogger(__name__)

_WHISPER_LOAD_LOCK = asyncio.Lock()


class SpeechAnalyzer(BaseAnalyzer):
    """
    Speech-to-text analyser powered by OpenAI Whisper.

    Parameters
    ----------
    model_size:
        Whisper model variant: ``tiny``, ``base``, ``small``, ``medium``, ``large``.
    language:
        Force a specific language code (e.g. ``"en"``).  ``None`` = auto-detect.
    device:
        PyTorch device string: ``"cpu"``, ``"cuda"``, ``"mps"``.
    """

    name = "speech"

    def __init__(
        self,
        model_size: str = "base",
        language: str | None = None,
        device: str = "cpu",
    ) -> None:
        self._model_size = model_size
        self._language = language
        self._device = device
        self._model: object | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper")

    async def load(self) -> None:
        async with _WHISPER_LOAD_LOCK:
            if self._model is not None:
                return
            log.info("Loading Whisper model '%s' on device '%s'…", self._model_size, self._device)
            loop = asyncio.get_running_loop()
            self._model = await loop.run_in_executor(self._executor, self._load_sync)
            log.info("Whisper model loaded")

    def _load_sync(self) -> object:
        try:
            import whisper  # type: ignore[import-untyped]
            return whisper.load_model(self._model_size, device=self._device)
        except Exception:
            log.warning("Whisper model could not be loaded – speech analysis disabled")
            return None

    async def unload(self) -> None:
        # Release the loaded model but keep the executor alive so this
        # analyzer instance can be safely reused with future load()/analyse() calls.
        self._model = None

    async def analyse(self, chunk: AudioChunk) -> SpeechResult:
        if self._model is None:
            return SpeechResult(segment=SpeechSegment(text=""), confidence=0.0)

        loop = asyncio.get_running_loop()
        segment = await loop.run_in_executor(
            self._executor, self._transcribe_sync, chunk.samples, chunk.sample_rate
        )
        return SpeechResult(segment=segment, confidence=segment.confidence)

    def _transcribe_sync(self, samples: np.ndarray, sample_rate: int) -> SpeechSegment:
        try:
            audio = samples.astype(np.float32)
            if sample_rate != 16_000:
                try:
                    import librosa  # type: ignore[import-untyped]
                    audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16_000)
                except ImportError:
                    pass  # best-effort

            opts: dict = {"word_timestamps": True}
            if self._language:
                opts["language"] = self._language

            result = self._model.transcribe(audio, **opts)  # type: ignore[union-attr]

            words: list[dict] = []
            for seg in result.get("segments", []):
                for w in seg.get("words", []):
                    words.append(
                        {"word": w["word"], "start": w["start"], "end": w["end"]}
                    )

            segments = result.get("segments", [])
            start_s = segments[0]["start"] if segments else 0.0
            end_s = segments[-1]["end"] if segments else 0.0

            # Heuristic confidence: avg log-prob mapped to [0,1]
            avg_logprob = (
                sum(s.get("avg_logprob", -1.0) for s in segments) / len(segments)
                if segments
                else -1.0
            )
            confidence = float(min(1.0, max(0.0, 1.0 + avg_logprob)))

            return SpeechSegment(
                text=result.get("text", "").strip(),
                language=result.get("language", "en"),
                confidence=confidence,
                start_s=start_s,
                end_s=end_s,
                words=words,
            )
        except Exception:
            log.exception("Whisper transcription failed")
            return SpeechSegment(text="", confidence=0.0)
