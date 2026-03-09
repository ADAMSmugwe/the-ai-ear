# Extending the System — Bring Your Own Model (BYOM)

The AI Ear is designed to be fully extensible.  Any modality can be replaced
or augmented by subclassing `BaseAnalyzer`.

---

## The `BaseAnalyzer` Contract

```python
from ai_ear.analyzers.base import BaseAnalyzer

class MyAnalyzer(BaseAnalyzer):
    # Must be unique within a pipeline.
    name = "my_analyzer"

    async def load(self) -> None:
        """Download or initialise the model.  Called once by the pipeline."""
        ...

    async def analyse(self, chunk: AudioChunk) -> PartialResult:
        """
        Analyse one AudioChunk.  Must return one of:
          SpeechResult, EmotionResult, EnvironmentResult, or MusicResult.
        Raising an exception is safe — the pipeline logs and continues.
        """
        ...

    async def unload(self) -> None:
        """Release resources.  Must be safe to call followed by load() again."""
        ...
```

---

## Return Types

| Return type | Merged into `AnalysisResult` field |
|---|---|
| `SpeechResult(segment, confidence)` | `result.speech` |
| `EmotionResult(profile, confidence)` | `result.emotion` |
| `EnvironmentResult(snapshot, confidence)` | `result.environment` |
| `MusicResult(profile, confidence)` | `result.music` |

---

## Example 1: Keyword Spotter

```python
from ai_ear.analyzers.base import BaseAnalyzer, SpeechResult
from ai_ear.core.models import AudioChunk, SpeechSegment

class KeywordSpotter(BaseAnalyzer):
    name = "keyword_spotter"
    KEYWORDS = ["hey computer", "attention", "alert"]

    async def load(self):
        try:
            from openwakeword.model import Model
            self._model = Model(wakeword_models=self.KEYWORDS)
        except ImportError:
            self._model = None

    async def analyse(self, chunk: AudioChunk) -> SpeechResult:
        if self._model is None:
            return SpeechResult(segment=SpeechSegment(text=""), confidence=0.0)
        prediction = self._model.predict(chunk.samples)
        keyword = max(prediction, key=prediction.get)
        score = prediction[keyword]
        text = keyword if score > 0.5 else ""
        return SpeechResult(
            segment=SpeechSegment(text=text, confidence=score),
            confidence=score,
        )
```

---

## Example 2: DNN Environment Classifier (YAMNet)

Replace the heuristic `EnvironmentAnalyzer` with a full DNN:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ai_ear.analyzers.base import BaseAnalyzer, EnvironmentResult
from ai_ear.core.models import AudioChunk, EnvironmentLabel, EnvironmentSnapshot


class YAMNetEnvironmentAnalyzer(BaseAnalyzer):
    name = "environment"  # same name — replaces the heuristic analyser

    def __init__(self, sample_rate: int = 16_000) -> None:
        self._sample_rate = sample_rate
        self._model = None
        self._executor: ThreadPoolExecutor | None = None

    async def load(self) -> None:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yamnet")
        loop = asyncio.get_running_loop()
        self._model = await loop.run_in_executor(self._executor, self._load_sync)

    def _load_sync(self):
        import tensorflow_hub as hub
        return hub.load("https://tfhub.dev/google/yamnet/1")

    async def unload(self) -> None:
        self._model = None
        if self._executor is not None:
            # wait=False returns immediately; any in-flight inference tasks will
            # still complete because the threads are daemon threads.  For
            # production code where you need to guarantee all tasks complete
            # before returning, use wait=True (may block briefly).
            self._executor.shutdown(wait=False)
            self._executor = None

    async def analyse(self, chunk: AudioChunk) -> EnvironmentResult:
        if self._model is None:
            return EnvironmentResult(
                snapshot=EnvironmentSnapshot(dominant=EnvironmentLabel.UNKNOWN),
                confidence=0.0,
            )
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yamnet")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, self._predict_sync, chunk.samples
        )

    def _predict_sync(self, samples: np.ndarray) -> EnvironmentResult:
        scores, embeddings, spectrogram = self._model(samples)
        top_class = int(np.argmax(np.mean(scores, axis=0)))
        # ... map to EnvironmentLabel ...
        return EnvironmentResult(
            snapshot=EnvironmentSnapshot(dominant=EnvironmentLabel.UNKNOWN, scores={}),
            confidence=float(np.max(np.mean(scores, axis=0))),
        )
```

---

## Example 3: Adding a Completely New Modality

You can return any result type and handle it downstream via callbacks:

```python
from dataclasses import dataclass

from ai_ear.analyzers.base import BaseAnalyzer
from ai_ear.core.models import AudioChunk


@dataclass
class GunShotResult:
    detected: bool
    confidence: float


class GunShotDetector(BaseAnalyzer):
    name = "gunshot"

    async def analyse(self, chunk: AudioChunk) -> GunShotResult:
        # Your custom model logic here
        return GunShotResult(detected=False, confidence=0.0)


# Subscribe to results and handle your custom type
pipeline = AudioPipeline(analyzers=[GunShotDetector()])

async def on_result(result):
    # AnalysisResult.speech/emotion/environment/music will be None
    # since GunShotResult doesn't map to any standard field.
    # Access raw partial results via the pipeline callback.
    pass
```

---

## Best Practices

1. **Thread safety**: Run synchronous/blocking model inference in a
   `ThreadPoolExecutor` via `loop.run_in_executor()`.  Never block the
   asyncio event loop.

2. **Restart safety**: In `unload()`, set your executor to `None` and shut it
   down.  In `load()`, recreate it if it is `None`.  This ensures the analyser
   works correctly after a pipeline stop/start cycle.

3. **Executor shutdown**: `executor.shutdown(wait=False)` returns immediately.
   Any in-flight tasks will still complete (threads are daemon threads).  Use
   `wait=True` if you need to guarantee all pending analysis finishes before
   `unload()` returns — note this may block briefly.

4. **Graceful fallback**: Wrap model loading in a `try/except` and return a
   zero-confidence result when the model is unavailable.

5. **Unique `name`**: The `name` attribute must be unique within a pipeline.
   If two analysers share the same name, the second result will overwrite the
   first in the fused `AnalysisResult`.
