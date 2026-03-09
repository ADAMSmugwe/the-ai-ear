# API Reference

Base URL: `http://localhost:8080` (default)

Interactive docs: `http://localhost:8080/docs` (Swagger UI)

---

## System Endpoints

### `GET /health`

Liveness and readiness probe.

**Response** `200 OK`
```json
{
  "status": "ok",
  "timestamp": 1741525355.0,
  "uptime_s": 42.1
}
```

---

### `GET /info`

Configuration summary.

**Response** `200 OK`
```json
{
  "version": "0.1.0",
  "whisper_model": "base",
  "emotion_enabled": true,
  "music_enabled": true,
  "environment_enabled": true,
  "sample_rate": 16000
}
```

---

### `GET /pipeline/stats`

Throughput statistics for the analysis pipeline.

**Response** `200 OK`
```json
{
  "chunks_processed": 127.0,
  "avg_latency_s": 0.043,
  "min_latency_s": 0.012,
  "max_latency_s": 0.201
}
```

---

## Analysis Endpoint

### `POST /analyse`

Upload an audio file and receive a full multi-modal analysis result.

**Request**  
Content-Type: `multipart/form-data`  
Field: `file` — audio file (WAV, FLAC, OGG, MP3)

**Response** `200 OK` — `AnalysisResult`
```json
{
  "chunk_id": "a3f1b8c9d...",
  "source_id": "interview.wav",
  "timestamp": 1741525355.0,
  "duration_s": 2.0,
  "overall_confidence": 0.87,
  "speech": {
    "text": "Good morning everyone",
    "language": "en",
    "confidence": 0.92,
    "start_s": 0.0,
    "end_s": 1.4,
    "words": [
      {"word": "Good", "start": 0.0, "end": 0.3},
      {"word": "morning", "start": 0.35, "end": 0.8}
    ]
  },
  "emotion": {
    "dominant": "neutral",
    "scores": {"neutral": 0.7, "happy": 0.2, "sad": 0.1},
    "arousal": 0.4,
    "valence": 0.65
  },
  "environment": {
    "dominant": "speech",
    "scores": {"speech": 0.82, "silence": 0.05, "music": 0.03},
    "noise_floor_db": -58.0,
    "snr_db": 22.5
  },
  "music": {
    "is_music": false,
    "tempo_bpm": null,
    "key": null,
    "energy": 0.32,
    "spectral_centroid_hz": 1420.0,
    "genre_hints": []
  },
  "semantic_tags": ["contains_speech", "env:speech"]
}
```

**Error Responses**
- `422 Unprocessable Entity` — audio could not be decoded

---

## Memory Endpoints

### `GET /memory/context`

Retrieve a structured context summary of recently heard audio.

**Query parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `window_s` | float | 60.0 | Context window in seconds |

**Response** `200 OK`
```json
{
  "window_s": 60.0,
  "transcript": "Good morning everyone. Let's discuss Q4 results.",
  "dominant_emotions": [["neutral", 28], ["happy", 4]],
  "dominant_environments": [["speech", 22], ["silence", 8]],
  "music_detected": false,
  "events": [
    {
      "event_type": "speech_started",
      "source_id": "microphone",
      "description": "Speech detected",
      "timestamp": 1741525355.0,
      "severity": 0.0,
      "payload": {}
    }
  ],
  "semantic_tags": ["contains_speech", "env:speech"]
}
```

---

### `GET /memory/transcript`

Plain-text transcript of recently heard speech.

**Query parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `window_s` | float | 60.0 | Transcript window in seconds |

**Response** `200 OK`
```json
{"transcript": "Good morning everyone. Let's discuss Q4 results."}
```

---

### `GET /memory/events`

Recent aural events, optionally filtered by type.

**Query parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `last_n` | int | 50 | Maximum number of events to return |
| `event_type` | string | null | Filter to a specific event type |

**Valid `event_type` values:**
`environment_change`, `speech_started`, `speech_ended`,
`music_started`, `music_ended`, `alarm_detected`, `custom`

**Response** `200 OK`
```json
{
  "events": [
    {
      "event_type": "environment_change",
      "source_id": "microphone",
      "description": "Environment changed to 'alarm'",
      "timestamp": 1741525400.0,
      "severity": 0.0,
      "payload": {"previous": "speech", "current": "alarm"}
    }
  ],
  "count": 1
}
```

**Error responses:**
- `400 Bad Request` — unknown `event_type` value

---

## WebSocket Streaming

### `WS /stream`

Real-time bi-directional audio streaming.

**Protocol**

| Direction | Format | Description |
|---|---|---|
| Client → Server | Binary | Raw PCM `float32` little-endian mono at `audio_sample_rate` (default 16 kHz) |
| Server → Client | JSON text | `AnalysisResult` JSON object after each assembled chunk |

The server automatically assembles incoming binary data into windows of
`audio_chunk_duration_s` seconds (default 2 s) before running analysis.
Partial frames are buffered until a full window is available.

**Example connection (Python websockets library):**

```python
import asyncio
import json
import numpy as np
import websockets

async def stream():
    async with websockets.connect("ws://localhost:8080/stream") as ws:
        # Send 2 seconds of silence
        silence = np.zeros(32000, dtype=np.float32)
        await ws.send(silence.tobytes())

        result = json.loads(await ws.recv())
        print("Environment:", result["environment"]["dominant"])
        print("Tags:", result["semantic_tags"])

asyncio.run(stream())
```

**Notes:**
- Each WebSocket connection registers a per-connection result callback that is
  automatically unregistered when the connection closes.
- Analysis results are only dispatched to the connection that sent the audio.

---

## Data Models

### `AnalysisResult`

| Field | Type | Description |
|---|---|---|
| `chunk_id` | string | SHA-1 of source_id + timestamp + length |
| `source_id` | string | Origin identifier (e.g. "microphone", "file:interview.wav") |
| `timestamp` | float | Unix epoch of the chunk start |
| `duration_s` | float | Duration of the audio window in seconds |
| `overall_confidence` | float | Average confidence across active analysers |
| `speech` | `SpeechSegment \| null` | Speech recognition result |
| `emotion` | `EmotionProfile \| null` | Emotion recognition result |
| `environment` | `EnvironmentSnapshot \| null` | Scene classification |
| `music` | `MusicProfile \| null` | Music characterisation |
| `semantic_tags` | string[] | Auto-derived tags (e.g. "contains_speech", "env:speech") |

### `SpeechSegment`

| Field | Type | Description |
|---|---|---|
| `text` | string | Transcribed text |
| `language` | string | ISO 639-1 language code |
| `confidence` | float | [0.0, 1.0] |
| `start_s` | float | Segment start offset in seconds |
| `end_s` | float | Segment end offset in seconds |
| `words` | object[] | Word-level timestamps: `{word, start, end}` |

### `EmotionProfile`

| Field | Type | Description |
|---|---|---|
| `dominant` | string | Dominant emotion label |
| `scores` | object | Per-label probability scores |
| `arousal` | float | Activation/energy dimension [0.0, 1.0] |
| `valence` | float | Positive/negative dimension [0.0, 1.0] |

### `EnvironmentSnapshot`

| Field | Type | Description |
|---|---|---|
| `dominant` | string | Dominant scene label |
| `scores` | object | Per-label probabilities |
| `noise_floor_db` | float | Estimated noise floor in dBFS |
| `snr_db` | float | Estimated signal-to-noise ratio in dB |

**Scene labels:** `silence`, `speech`, `music`, `alarm`, `crowd`, `traffic`, `office`, `unknown`

### `MusicProfile`

| Field | Type | Description |
|---|---|---|
| `is_music` | bool | Whether this window contains music |
| `tempo_bpm` | float \| null | Estimated tempo in BPM |
| `key` | string \| null | Estimated musical key (e.g. "C major") |
| `energy` | float | Normalised spectral energy [0.0, 1.0] |
| `spectral_centroid_hz` | float | Timbral brightness in Hz |
| `genre_hints` | string[] | Coarse genre suggestions |

### `AuralEvent`

| Field | Type | Description |
|---|---|---|
| `event_type` | string | One of the event type values listed above |
| `source_id` | string | Origin of the event |
| `description` | string | Human-readable description |
| `timestamp` | float | Unix epoch |
| `severity` | float | Urgency score [0.0, 1.0] (0.9 for alarms) |
| `payload` | object | Event-specific data |
