# Configuration

All settings can be overridden via environment variables prefixed with `AIEAR_`.

## Full Settings Reference

| Environment Variable | Type | Default | Description |
|---|---|---|---|
| `AIEAR_AUDIO_SAMPLE_RATE` | int | `16000` | Target audio sample rate in Hz |
| `AIEAR_AUDIO_CHUNK_DURATION_S` | float | `2.0` | Duration of each analysis window in seconds |
| `AIEAR_SPEECH_ENABLED` | bool | `true` | Enable Whisper speech recognition |
| `AIEAR_WHISPER_MODEL` | string | `"base"` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large` |
| `AIEAR_WHISPER_LANGUAGE` | string \| null | `null` | Force language (e.g. `"en"`). `null` = auto-detect |
| `AIEAR_WHISPER_DEVICE` | string | `"cpu"` | PyTorch device: `"cpu"`, `"cuda"`, `"mps"` |
| `AIEAR_EMOTION_ENABLED` | bool | `true` | Enable wav2vec2 emotion recognition |
| `AIEAR_EMOTION_MODEL` | string | `"ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"` | HuggingFace model ID |
| `AIEAR_MUSIC_ENABLED` | bool | `true` | Enable librosa music analysis |
| `AIEAR_ENVIRONMENT_ENABLED` | bool | `true` | Enable environment classification |
| `AIEAR_ENVIRONMENT_NOISE_GATE_DB` | float | `-50.0` | Frames below this dBFS level are classified as silence |
| `AIEAR_MEMORY_MAX_RESULTS` | int | `500` | Rolling buffer size for analysis results |
| `AIEAR_MEMORY_MAX_EVENTS` | int | `1000` | Rolling buffer size for aural events |
| `AIEAR_MEMORY_CONTEXT_WINDOW_S` | float | `60.0` | Default context window for `context_summary()` |
| `AIEAR_API_HOST` | string | `"0.0.0.0"` | API server listen host |
| `AIEAR_API_PORT` | int | `8080` | API server listen port |
| `AIEAR_API_WORKERS` | int | `1` | Number of uvicorn worker processes |
| `AIEAR_API_CORS_ORIGINS` | list[str] | `["*"]` | CORS allowed origins |

## Examples

**Minimal footprint (no ML models):**
```bash
AIEAR_SPEECH_ENABLED=false AIEAR_EMOTION_ENABLED=false AIEAR_MUSIC_ENABLED=false ai-ear serve
```

**Fast speech-only on GPU:**
```bash
AIEAR_WHISPER_MODEL=small AIEAR_WHISPER_DEVICE=cuda AIEAR_EMOTION_ENABLED=false AIEAR_MUSIC_ENABLED=false ai-ear serve
```

**Force English, larger model:**
```bash
AIEAR_WHISPER_MODEL=medium AIEAR_WHISPER_LANGUAGE=en ai-ear serve
```

**Using a `.env` file:**

Create `.env` in the project root:
```
AIEAR_WHISPER_MODEL=small
AIEAR_WHISPER_DEVICE=cpu
AIEAR_EMOTION_ENABLED=false
AIEAR_API_PORT=9090
```

Then:
```bash
# pydantic-settings picks up .env automatically
ai-ear serve
```
