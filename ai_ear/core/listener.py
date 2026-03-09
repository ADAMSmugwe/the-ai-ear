"""
AudioListener — real-time, non-blocking audio capture.

Responsibilities
----------------
* Open a microphone / audio device via sounddevice.
* Push fixed-size :class:`~ai_ear.core.models.AudioChunk` objects into an
  asyncio-friendly queue consumed by the :class:`~ai_ear.core.pipeline.AudioPipeline`.
* Support file-based ingestion for batch / test workflows.
* Provide graceful start / stop semantics.

Design notes
------------
sounddevice callbacks run in a C thread; we bridge to the asyncio event loop
via ``loop.call_soon_threadsafe`` so the rest of the system stays fully async.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

try:
    import sounddevice as sd  # type: ignore[import-untyped]
    _SD_AVAILABLE = True
except Exception:  # pragma: no cover
    _SD_AVAILABLE = False
    log.warning("sounddevice not available – microphone capture disabled")

try:
    import soundfile as sf  # type: ignore[import-untyped]
    _SF_AVAILABLE = True
except Exception:  # pragma: no cover
    _SF_AVAILABLE = False
    log.warning("soundfile not available – file ingestion disabled")

from ai_ear.core.models import AudioChunk


class AudioListener:
    """
    Captures audio from a device or file and yields :class:`AudioChunk` objects.

    Parameters
    ----------
    sample_rate:
        Target sample rate in Hz.  File ingestion resamples to this rate if
        required.
    chunk_duration_s:
        Duration of each emitted chunk in seconds.
    channels:
        Number of input channels.  Output is always mixed-down to mono.
    device_index:
        sounddevice device index.  ``None`` uses the system default.
    source_id:
        Logical name attached to every emitted chunk.
    queue_maxsize:
        Maximum number of chunks buffered before backpressure is applied.
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        chunk_duration_s: float = 2.0,
        channels: int = 1,
        device_index: int | None = None,
        source_id: str = "microphone",
        queue_maxsize: int = 64,
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_duration_s = chunk_duration_s
        self.channels = channels
        self.device_index = device_index
        self.source_id = source_id

        self._chunk_frames = int(sample_rate * chunk_duration_s)
        self._queue: asyncio.Queue[AudioChunk] = asyncio.Queue(maxsize=queue_maxsize)
        self._stream: sd.InputStream | None = None
        self._running = False
        self._buffer = np.empty((0,), dtype=np.float32)
        self._loop: asyncio.AbstractEventLoop | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Open the audio device and begin capturing."""
        if not _SD_AVAILABLE:
            raise RuntimeError(
                "sounddevice is not installed – cannot open microphone. "
                "Install it with: pip install sounddevice"
            )
        self._loop = asyncio.get_running_loop()
        self._running = True
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            device=self.device_index,
            blocksize=self._chunk_frames,
            callback=self._sd_callback,
        )
        self._stream.start()
        log.info(
            "AudioListener started",
            extra={"source_id": self.source_id, "sample_rate": self.sample_rate},
        )

    async def stop(self) -> None:
        """Stop capturing and release the audio device."""
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        log.info("AudioListener stopped", extra={"source_id": self.source_id})

    async def chunks(self) -> AsyncIterator[AudioChunk]:
        """Async generator that yields captured :class:`AudioChunk` objects."""
        while self._running or not self._queue.empty():
            try:
                chunk = await asyncio.wait_for(self._queue.get(), timeout=0.5)
                yield chunk
            except asyncio.TimeoutError:
                continue

    async def ingest_file(self, path: str | Path) -> AsyncIterator[AudioChunk]:
        """
        Yield chunks read from an audio file (WAV, FLAC, OGG, …).

        This allows batch-processing or offline testing without a live
        microphone.
        """
        if not _SF_AVAILABLE:
            raise RuntimeError(
                "soundfile is not installed – cannot read audio files. "
                "Install it with: pip install soundfile"
            )
        path = Path(path)
        with sf.SoundFile(str(path)) as f:
            file_sr = f.samplerate
            file_ch = f.channels

            while True:
                # Read in target-size windows
                frames_to_read = int(self._chunk_frames * (file_sr / self.sample_rate))
                data = f.read(frames_to_read, dtype="float32", always_2d=True)
                if data.shape[0] == 0:
                    break

                # Mix down to mono
                mono = data.mean(axis=1) if file_ch > 1 else data[:, 0]

                # Resample if needed
                if file_sr != self.sample_rate:
                    mono = _resample(mono, file_sr, self.sample_rate)

                chunk = AudioChunk(
                    samples=mono,
                    sample_rate=self.sample_rate,
                    timestamp=time.time(),
                    source_id=f"file:{path.name}",
                )
                yield chunk

    # ------------------------------------------------------------------
    # sounddevice callback (runs in C audio thread)
    # ------------------------------------------------------------------

    def _sd_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,  # cffi struct; not used
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            log.warning("AudioListener stream status: %s", status)

        # Mix to mono
        mono = indata.mean(axis=1) if indata.ndim > 1 else indata[:, 0]

        # Accumulate into buffer and emit complete chunks
        self._buffer = np.concatenate([self._buffer, mono])
        while len(self._buffer) >= self._chunk_frames:
            window = self._buffer[: self._chunk_frames].copy()
            self._buffer = self._buffer[self._chunk_frames :]
            chunk = AudioChunk(
                samples=window,
                sample_rate=self.sample_rate,
                timestamp=time.time(),
                source_id=self.source_id,
            )
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._enqueue_chunk, chunk)


    def _enqueue_chunk(self, chunk: AudioChunk) -> None:
        """
        Called in the asyncio event loop thread (via call_soon_threadsafe).

        Drops the chunk and logs a warning if the queue is full, rather than
        raising QueueFull which would propagate as an unhandled exception.
        """
        try:
            self._queue.put_nowait(chunk)
        except asyncio.QueueFull:
            log.warning(
                "AudioListener queue full (maxsize=%d) – dropping chunk from '%s'",
                self._queue.maxsize,
                self.source_id,
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Simple linear-interpolation resample (librosa not required for basic use)."""
    try:
        import librosa  # type: ignore[import-untyped]
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    except ImportError:
        # Fallback: numpy-only linear interpolation
        duration = len(audio) / orig_sr
        target_len = int(duration * target_sr)
        indices = np.linspace(0, len(audio) - 1, target_len)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
