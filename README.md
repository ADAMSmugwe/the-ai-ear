# 🎧 The AI Ear

> *"Hear beyond words."*

**The AI Ear** is an outside-the-box, frontier, enterprise-grade AI system that gives machines the ability to **truly hear** — not just transcribe, but holistically understand the acoustic world in real time.

---

## What Makes It Different

Most "AI audio" systems stop at speech-to-text.  The AI Ear goes further:

| Capability | What it means |
|---|---|
| **Multi-modal analysis** | Every audio window is simultaneously analysed for speech, emotion, acoustic environment, and music — in parallel |
| **Temporal memory** | The system *remembers* what it has heard, building a rolling semantic context rather than processing isolated moments |
| **Aural events** | State-machine transitions (speech started, music detected, environment changed, alarm sounded) surface as typed events for downstream alerting |
| **LLM-ready context** | One call to `memory.context_summary()` produces a structured dict ready to inject into any LLM system prompt |
| **Enterprise API** | FastAPI REST + WebSocket server with CORS, structured logging, and full OpenAPI docs |
| **Pluggable analysers** | Bring your own model (BYOM) — swap in any `BaseAnalyzer` subclass at construction time |
| **Zero-copy fan-out** | Results and events broadcast concurrently to all registered callbacks |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        The AI Ear                               │
│                                                                 │
│  ┌─────────────┐    ┌──────────────────────────────────────┐   │
│  │AudioListener│───▶│           AudioPipeline               │   │
│  │ (mic/file)  │    │  ┌──────────┐  ┌──────────────────┐  │   │
│  └─────────────┘    │  │ Speech   │  │   Environment    │  │   │
│                     │  │Analyzer  │  │    Analyzer      │  │   │
│  ┌─────────────┐    │  │(Whisper) │  │  (heuristic +   │  │   │
│  │  REST API   │    │  └──────────┘  │   DNN-ready)    │  │   │
│  │  /analyse   │    │  ┌──────────┐  └──────────────────┘  │   │
│  │  /memory/*  │    │  │ Emotion  │  ┌──────────────────┐  │   │
│  │  /health    │    │  │Analyzer  │  │  MusicAnalyzer   │  │   │
│  └─────────────┘    │  │(wav2vec2)│  │   (librosa)      │  │   │
│                     │  └──────────┘  └──────────────────┘  │   │
│  ┌─────────────┐    │         │ concurrent asyncio.gather │   │
│  │  WebSocket  │    │         ▼                            │   │
│  │  /stream    │    │  ┌────────────┐   ┌──────────────┐  │   │
│  └─────────────┘    │  │   Fusion   │──▶│ AuralMemory  │  │   │
│                     │  │(AnalysisRe-│   │(rolling      │  │   │
│                     │  │  sult +    │   │ context +    │  │   │
│                     │  │ semantic   │   │ events)      │  │   │
│                     │  │  tags)     │   └──────────────┘  │   │
│                     │  └────────────┘                      │   │
│                     └──────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Components

| Module | Role |
|---|---|
| `ai_ear/core/listener.py` | Non-blocking audio capture (microphone + file ingestion) |
| `ai_ear/core/pipeline.py` | Concurrent multi-modal analysis engine |
| `ai_ear/core/memory.py` | Temporal context memory with context summary + transcript |
| `ai_ear/core/models.py` | Strongly-typed Pydantic data models for the entire system |
| `ai_ear/core/config.py` | Pydantic-settings configuration (env-var overridable) |
| `ai_ear/analyzers/speech.py` | OpenAI Whisper speech recognition + word timestamps |
| `ai_ear/analyzers/emotion.py` | wav2vec2 speech emotion recognition |
| `ai_ear/analyzers/environment.py` | Acoustic scene classification (silence/speech/music/alarm/crowd/traffic) |
| `ai_ear/analyzers/music.py` | Tempo, key, energy, and genre-hint extraction via librosa |
| `ai_ear/api/server.py` | FastAPI REST + WebSocket API server |
| `ai_ear/utils/audio.py` | Pure-numpy DSP utilities (RMS, ZCR, spectral centroid, flatness) |

---

## Quick Start

### Install

```bash
pip install -e ".[dev]"
```

> **Heavy ML dependencies** (Whisper, PyTorch, transformers, librosa) are listed in
> `requirements.txt` and `pyproject.toml`.  For a lightweight evaluation, the
> system degrades gracefully when they are absent — speech/emotion analysers
> return empty results; environment/music analysers use fast numpy heuristics.

### Run the API server

```bash
ai-ear serve --host 0.0.0.0 --port 8080
```

Or programmatically:

```python
import uvicorn
from ai_ear.api.server import create_app
from ai_ear.core.config import Settings

app = create_app(Settings())
uvicorn.run(app, host="0.0.0.0", port=8080)
```

Interactive API docs available at `http://localhost:8080/docs`.

### Analyse a file via REST

```bash
curl -X POST http://localhost:8080/analyse \
     -F "file=@interview.wav" | python -m json.tool
```

### Real-time microphone listening (Python)

```python
import asyncio
from ai_ear.core.listener import AudioListener
from ai_ear.core.pipeline import AudioPipeline
from ai_ear.core.memory import AuralMemory
from ai_ear.analyzers.speech import SpeechAnalyzer
from ai_ear.analyzers.emotion import EmotionAnalyzer
from ai_ear.analyzers.environment import EnvironmentAnalyzer
from ai_ear.analyzers.music import MusicAnalyzer

async def main():
    memory = AuralMemory()
    pipeline = AudioPipeline(
        analyzers=[
            SpeechAnalyzer(model_size="base"),
            EmotionAnalyzer(),
            EnvironmentAnalyzer(),
            MusicAnalyzer(),
        ],
        memory=memory,
    )

    async def on_result(result):
        print(f"Speech : {result.speech.text if result.speech else '—'}")
        print(f"Emotion: {result.emotion.dominant.value if result.emotion else '—'}")
        print(f"Tags   : {result.semantic_tags}")

    pipeline.on_result(on_result)
    await pipeline.start()

    listener = AudioListener(sample_rate=16_000, chunk_duration_s=2.0)
    await listener.start()

    try:
        await pipeline.process_stream(listener.chunks())
    finally:
        await listener.stop()
        await pipeline.stop()

asyncio.run(main())
```

### WebSocket streaming (JavaScript client)

```javascript
const ws = new WebSocket("ws://localhost:8080/stream");

ws.onopen = () => {
  // Send raw Float32 PCM chunks (16 kHz mono) as binary frames
  const pcmChunk = new Float32Array(32000); // 2 seconds @ 16 kHz
  ws.send(pcmChunk.buffer);
};

ws.onmessage = (event) => {
  const result = JSON.parse(event.data);
  console.log("Heard:", result.speech?.text);
  console.log("Tags:", result.semantic_tags);
};
```

### LLM context injection

```python
from ai_ear.core.memory import AuralMemory

memory: AuralMemory = ...  # already receiving results from pipeline

# Inject into any LLM system prompt
context = memory.context_summary(window_s=60)
system_prompt = f"""
You are an AI assistant with real-time acoustic awareness.
You have been listening for the last {context['window_s']:.0f} seconds.

ACOUSTIC CONTEXT:
Transcribed speech: "{context['transcript']}"
Prevailing emotion: {context['dominant_emotions'][0][0] if context['dominant_emotions'] else 'neutral'}
Environment: {context['dominant_environments'][0][0] if context['dominant_environments'] else 'unknown'}
Music detected: {context['music_detected']}
"""
```

---

## Configuration

All settings are configurable via environment variables (prefix `AIEAR_`) or a `.env` file:

| Variable | Default | Description |
|---|---|---|
| `AIEAR_WHISPER_MODEL` | `base` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large` |
| `AIEAR_WHISPER_DEVICE` | `cpu` | PyTorch device: `cpu`, `cuda`, `mps` |
| `AIEAR_WHISPER_LANGUAGE` | *(auto)* | Force language code (e.g. `en`) |
| `AIEAR_EMOTION_ENABLED` | `true` | Enable emotion analysis |
| `AIEAR_MUSIC_ENABLED` | `true` | Enable music analysis |
| `AIEAR_ENVIRONMENT_ENABLED` | `true` | Enable environment classification |
| `AIEAR_AUDIO_SAMPLE_RATE` | `16000` | Capture sample rate (Hz) |
| `AIEAR_AUDIO_CHUNK_DURATION_S` | `2.0` | Analysis window size (seconds) |
| `AIEAR_MEMORY_CONTEXT_WINDOW_S` | `60.0` | Rolling context window (seconds) |
| `AIEAR_API_PORT` | `8080` | API server port |
| `AIEAR_LOG_JSON` | `false` | Emit structured JSON logs |

---

## API Reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness / readiness probe |
| `GET` | `/info` | Build info and configuration |
| `POST` | `/analyse` | Analyse an uploaded audio file |
| `GET` | `/memory/context` | Structured context summary |
| `GET` | `/memory/transcript` | Plain-text recent speech |
| `GET` | `/memory/events` | Recent aural events |
| `GET` | `/pipeline/stats` | Pipeline throughput statistics |
| `WS` | `/stream` | Real-time PCM audio streaming |

Full interactive docs: `http://localhost:8080/docs`

---

## Bring Your Own Model (BYOM)

```python
from ai_ear.analyzers.base import BaseAnalyzer, SpeechResult
from ai_ear.core.models import AudioChunk, SpeechSegment

class MyKeywordSpotter(BaseAnalyzer):
    name = "keyword_spotter"

    async def load(self):
        # Load your custom model here
        self._model = load_my_model()

    async def analyse(self, chunk: AudioChunk) -> SpeechResult:
        keyword = self._model.detect(chunk.samples)
        return SpeechResult(
            segment=SpeechSegment(text=keyword or "", confidence=0.95),
            confidence=0.95 if keyword else 0.0,
        )

# Inject into the pipeline
from ai_ear.core.pipeline import AudioPipeline
pipeline = AudioPipeline(analyzers=[MyKeywordSpotter(), ...])
```

---

## Examples

```bash
# Synthetic demo (no audio hardware required)
python examples/basic_listening.py --demo

# Analyse a real audio file
python examples/basic_listening.py path/to/audio.wav

# Enterprise integration patterns
python examples/enterprise_integration.py custom-analyser
python examples/enterprise_integration.py alerting
python examples/enterprise_integration.py llm-prompt
python examples/enterprise_integration.py serve
```

---

## Testing

```bash
# Run the full test suite
pytest

# With coverage
pytest --cov=ai_ear --cov-report=term-missing
```

---

## Aural Events

The pipeline automatically surfaces discrete events for real-time alerting:

| Event | Description |
|---|---|
| `speech_started` | Voice activity detected |
| `speech_ended` | Voice activity ceased |
| `keyword_detected` | Registered keyword recognised |
| `emotion_shift` | Dominant emotion changed |
| `environment_change` | Acoustic scene changed |
| `music_started` | Music onset detected |
| `music_ended` | Music offset detected |
| `alarm_detected` | Alarm sound detected (high severity) |
| `silence_started` | Silence onset |
| `silence_ended` | Silence offset |
| `anomaly` | Unclassified acoustic anomaly |

---

## License

MIT
