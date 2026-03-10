# Architecture

## Overview

The AI Ear is structured as a layered pipeline:

```
┌─────────────────────────────────────────────────────────────────────┐
│                          The AI Ear                                 │
│                                                                     │
│  ┌───────────────┐    ┌───────────────────────────────────────────┐ │
│  │  AudioListener │───▶│             AudioPipeline                 │ │
│  │  (mic / file)  │    │                                           │ │
│  └───────────────┘    │  ┌──────────┐  ┌───────────────────────┐  │ │
│                       │  │ Speech   │  │  EnvironmentAnalyzer  │  │ │
│  ┌───────────────┐    │  │Analyzer  │  │  (heuristic / DNN)    │  │ │
│  │  REST API      │    │  │(Whisper) │  └───────────────────────┘  │ │
│  │  /analyse      │    │  └──────────┘  ┌───────────────────────┐  │ │
│  │  /memory/*     │    │  ┌──────────┐  │    MusicAnalyzer      │  │ │
│  │  /pipeline/... │    │  │ Emotion  │  │    (librosa / BYOM)   │  │ │
│  └───────────────┘    │  │Analyzer  │  └───────────────────────┘  │ │
│                       │  │(wav2vec2)│                               │ │
│  ┌───────────────┐    │  └──────────┘                               │ │
│  │  WebSocket     │    │        │ asyncio.gather (concurrent)  │    │ │
│  │  /stream       │    │        ▼                                   │ │
│  └───────────────┘    │  ┌────────────────┐  ┌────────────────┐    │ │
│                       │  │ AnalysisResult │  │  AuralMemory   │    │ │
│                       │  │ fusion +       │─▶│  rolling ctx   │    │ │
│                       │  │ semantic tags  │  │  + events      │    │ │
│                       │  └────────────────┘  └────────────────┘    │ │
│                       └───────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### AudioListener (`ai_ear/core/listener.py`)

Responsible for capturing raw audio from a device or file and producing
`AudioChunk` objects.

**Key features:**
- Non-blocking capture using [sounddevice](https://python-sounddevice.readthedocs.io/)
- C audio thread bridged to asyncio via `loop.call_soon_threadsafe`
- Automatic mono mix-down from multi-channel input
- File ingestion (WAV, FLAC, OGG) with auto-resampling via librosa
- Bounded asyncio queue with drop-and-log behaviour when full

**Lifecycle:**
```python
listener = AudioListener(sample_rate=16_000, chunk_duration_s=2.0)
await listener.start()           # opens audio device
async for chunk in listener.chunks():
    ...                          # yields AudioChunk objects
await listener.stop()            # closes audio device
```

---

### AudioPipeline (`ai_ear/core/pipeline.py`)

The analysis fan-out and fusion engine.

**Key features:**
- All analysers run **concurrently** via `asyncio.gather`
- Semaphore-based back-pressure (`max_concurrent_chunks`)
- Per-source-id state tracking for event generation (no cross-talk between sources)
- Observable via `on_result()` / `on_event()` callbacks, each returning an unsubscribe callable
- `add_analyzer()` schedules `load()` automatically when the pipeline is running

**Event detection:**
The pipeline tracks state transitions per audio source and emits typed
`AuralEvent` objects:

| Transition | Event |
|---|---|
| Environment label changes | `environment_change` |
| Speech appears | `speech_started` |
| Speech disappears | `speech_ended` |
| Music appears | `music_started` |
| Music disappears | `music_ended` |
| Alarm label detected | `alarm_detected` |

---

### AuralMemory (`ai_ear/core/memory.py`)

Rolling temporal store for analysis results and events.

**Key features:**
- Bounded `deque` storage (configurable `max_results`, `max_events`)
- `context_summary(window_s)` — structured dict ready for LLM injection
- `transcript(window_s)` — plain text of recent speech
- `events_since(ts)` / `recent_events(last_n, event_type)` — filtered event queries
- Acoustic fingerprint registry (extensible, type-agnostic embeddings)

---

### Analysers (`ai_ear/analyzers/`)

All analysers inherit from `BaseAnalyzer`:

```
BaseAnalyzer
├── SpeechAnalyzer    (Whisper)
├── EmotionAnalyzer   (wav2vec2 via Transformers)
├── EnvironmentAnalyzer (numpy heuristics + optional DNN)
└── MusicAnalyzer     (librosa + chroma template matching)
```

**Analyser contract:**
1. `load()` — initialise models / download weights
2. `analyse(chunk)` → typed partial result
3. `unload()` — release resources (safe to call then `load()` again)

**Restart safety:**
All analysers safely recreate their `ThreadPoolExecutor` on `load()` after
`unload()` has been called, enabling pipeline restart.

---

### FastAPI Server (`ai_ear/api/server.py`)

Provides REST and WebSocket interfaces.

**WebSocket safety:**
- Per-connection result callbacks are automatically unregistered on disconnect
  (no callback leaks)
- Chunks are processed by awaiting `pipeline.process()` directly; no unbounded
  task accumulation

---

## Data Flow

```
Microphone/File
       │
       ▼
AudioChunk (float32 PCM, sample_rate, source_id, timestamp)
       │
       ▼ asyncio.gather
┌──────┴──────┬──────────────┬──────────────┐
│SpeechResult │EmotionResult │EnvResult     │MusicResult
└──────┬──────┴──────────────┴──────────────┘
       │ _merge_partial + _derive_tags
       ▼
AnalysisResult (fused, semantic_tags, overall_confidence)
       │
       ├──▶ AuralMemory.store_result()
       ├──▶ result callbacks (WebSocket, logging, ...)
       │
       ▼ _derive_events (per source_id state machine)
AuralEvent[]
       │
       ├──▶ AuralMemory.store_event()
       └──▶ event callbacks (alerting, ...)
```

---

## Threading Model

| Component | Thread |
|---|---|
| asyncio event loop | Main thread |
| sounddevice callback | C audio thread (via `call_soon_threadsafe`) |
| Whisper inference | `ThreadPoolExecutor` (1 worker) |
| Emotion inference | `ThreadPoolExecutor` (1 worker) |
| Environment classification | `ThreadPoolExecutor` (2 workers) |
| Music analysis | `ThreadPoolExecutor` (2 workers) |

All analyser executors are created fresh on `load()` and shut down on `unload()`,
making them safe to restart.
