# User Guide

## Prerequisites

- Python 3.10+
- For speech recognition: `openai-whisper` and PyTorch
- For emotion recognition: `transformers` and PyTorch
- For music analysis: `librosa`
- For microphone capture: `sounddevice` (and a working audio device)

Install everything:

```bash
pip install -e ".[dev]"
```

All ML dependencies are optional — the system degrades gracefully when they
are absent.

---

## Guide 1: Analysing an Audio File

The simplest way to get started is to analyse a pre-recorded file.

```python
import asyncio
from ai_ear.analyzers.environment import EnvironmentAnalyzer
from ai_ear.analyzers.music import MusicAnalyzer
from ai_ear.analyzers.speech import SpeechAnalyzer
from ai_ear.core.listener import AudioListener
from ai_ear.core.memory import AuralMemory
from ai_ear.core.pipeline import AudioPipeline

async def analyse_file(path: str) -> None:
    memory = AuralMemory(context_window_s=300)
    pipeline = AudioPipeline(
        analyzers=[
            SpeechAnalyzer(model_size="base"),  # uses Whisper
            EnvironmentAnalyzer(),
            MusicAnalyzer(),
        ],
        memory=memory,
    )
    await pipeline.start()

    listener = AudioListener(sample_rate=16_000, chunk_duration_s=2.0)
    async for chunk in listener.ingest_file(path):
        result = await pipeline.process(chunk)
        print(f"[{result.timestamp:.1f}] env={result.environment.dominant.value}")
        if result.speech and result.speech.text:
            print(f"  Speech: {result.speech.text}")

    await pipeline.stop()

    print("\n--- Context Summary ---")
    summary = memory.context_summary()
    print(f"Transcript: {summary['transcript']}")
    print(f"Environments: {summary['dominant_environments']}")

asyncio.run(analyse_file("interview.wav"))
```

Run the bundled demo (no audio file required):

```bash
python examples/basic_listening.py --demo
```

---

## Guide 2: Real-Time Microphone Listening

```python
import asyncio
from ai_ear.analyzers.environment import EnvironmentAnalyzer
from ai_ear.analyzers.speech import SpeechAnalyzer
from ai_ear.core.listener import AudioListener
from ai_ear.core.memory import AuralMemory
from ai_ear.core.pipeline import AudioPipeline

async def listen() -> None:
    memory = AuralMemory()
    pipeline = AudioPipeline(
        analyzers=[SpeechAnalyzer(), EnvironmentAnalyzer()],
        memory=memory,
    )

    async def on_result(result):
        text = result.speech.text if result.speech else ""
        env = result.environment.dominant.value if result.environment else "?"
        print(f"env={env}  speech={text!r}")

    pipeline.on_result(on_result)
    await pipeline.start()

    listener = AudioListener(sample_rate=16_000, chunk_duration_s=2.0)
    await listener.start()
    try:
        await pipeline.process_stream(listener.chunks())
    except KeyboardInterrupt:
        pass
    finally:
        await listener.stop()
        await pipeline.stop()

asyncio.run(listen())
```

---

## Guide 3: Real-Time Alerting

Subscribe to `AuralEvent` objects emitted when the acoustic scene changes.

```python
import asyncio
from ai_ear.analyzers.environment import EnvironmentAnalyzer
from ai_ear.core.listener import AudioListener
from ai_ear.core.models import AuralEvent, AuralEventType
from ai_ear.core.pipeline import AudioPipeline

async def listen_with_alerts() -> None:
    pipeline = AudioPipeline(analyzers=[EnvironmentAnalyzer()])

    async def on_event(event: AuralEvent) -> None:
        if event.event_type == AuralEventType.ALARM_DETECTED:
            print(f"🚨 ALARM DETECTED (severity={event.severity:.1f})")
        elif event.event_type == AuralEventType.ENVIRONMENT_CHANGE:
            print(f"ℹ️  Environment → {event.payload['current'].value}")

    pipeline.on_event(on_event)
    await pipeline.start()

    listener = AudioListener()
    await listener.start()
    try:
        await pipeline.process_stream(listener.chunks())
    finally:
        await listener.stop()
        await pipeline.stop()

asyncio.run(listen_with_alerts())
```

---

## Guide 4: LLM Context Injection

Use `AuralMemory.context_summary()` to inject acoustic context into any
large-language-model system prompt.

```python
from ai_ear.core.memory import AuralMemory

memory: AuralMemory = ...  # already populated by the pipeline

context = memory.context_summary(window_s=60)

system_prompt = f"""
You are an AI assistant with real-time acoustic awareness.
You have been listening for the last {context['window_s']:.0f} seconds.

ACOUSTIC CONTEXT:
  Transcribed speech: "{context['transcript']}"
  Prevailing emotion: {context['dominant_emotions'][0][0] if context['dominant_emotions'] else 'neutral'}
  Acoustic environment: {context['dominant_environments'][0][0] if context['dominant_environments'] else 'unknown'}
  Music playing: {context['music_detected']}
  Notable events: {len(context['events'])}

Respond naturally, taking this acoustic context into account.
"""
```

---

## Guide 5: Running the API Server

Start the REST + WebSocket server:

```bash
# Default: http://0.0.0.0:8080
ai-ear serve

# Custom host/port
ai-ear serve --host 127.0.0.1 --port 9090
```

Or programmatically:

```python
import uvicorn
from ai_ear.api.server import create_app
from ai_ear.core.config import Settings

app = create_app(Settings(whisper_model="small", audio_sample_rate=16000))
uvicorn.run(app, host="0.0.0.0", port=8080)
```

Interactive API docs are available at `http://localhost:8080/docs`.

---

## Guide 6: WebSocket Streaming (Browser / JavaScript)

Stream raw PCM audio from a browser microphone in real time:

```javascript
async function streamMicrophone() {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const ctx = new AudioContext({ sampleRate: 16000 });
  const source = ctx.createMediaStreamSource(stream);
  const processor = ctx.createScriptProcessor(32000, 1, 1);  // 2s chunks

  const ws = new WebSocket("ws://localhost:8080/stream");

  processor.onaudioprocess = (e) => {
    const samples = e.inputBuffer.getChannelData(0);  // Float32Array
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(samples.buffer);
    }
  };

  ws.onmessage = (event) => {
    const result = JSON.parse(event.data);
    console.log("Transcript:", result.speech?.text ?? "");
    console.log("Environment:", result.environment?.dominant ?? "");
    console.log("Tags:", result.semantic_tags);
  };

  source.connect(processor);
  processor.connect(ctx.destination);
}
```

---

## Guide 7: Subscribing and Unsubscribing from Callbacks

Both `on_result()` and `on_event()` return an unsubscribe callable:

```python
async def my_callback(result):
    print(result.semantic_tags)

# Subscribe
unsubscribe = pipeline.on_result(my_callback)

# ... later, when done (e.g. on WebSocket disconnect)
unsubscribe()
```

This is especially important in server contexts where callbacks must not
outlive the connection that registered them.

---

## Guide 8: Bring Your Own Model (BYOM)

Any `BaseAnalyzer` subclass can be injected at construction time:

```python
from ai_ear.analyzers.base import BaseAnalyzer, SpeechResult
from ai_ear.core.models import AudioChunk, SpeechSegment
from ai_ear.core.pipeline import AudioPipeline

class MyKeywordSpotter(BaseAnalyzer):
    name = "keyword_spotter"
    KEYWORDS = ["hey computer", "alert", "emergency"]

    async def load(self):
        # Load your custom model here (e.g. openwakeword, Porcupine)
        self._model = load_my_model()

    async def analyse(self, chunk: AudioChunk) -> SpeechResult:
        detected = self._model.detect(chunk.samples, chunk.sample_rate)
        text = detected.keyword if detected else ""
        return SpeechResult(
            segment=SpeechSegment(text=text, confidence=detected.score if detected else 0.0),
            confidence=detected.score if detected else 0.0,
        )

# Plug it into the pipeline
pipeline = AudioPipeline(
    analyzers=[
        MyKeywordSpotter(),
        EnvironmentAnalyzer(),
    ]
)
```

You can also add analysers at runtime while the pipeline is running.
`add_analyzer()` schedules `load()` as a background asyncio task, so the
analyser will be ready after the current event loop iteration but **not**
immediately upon return.  If you need the analyser to be loaded before
processing the next chunk, await `load()` yourself first:

```python
await pipeline.start()

# Option 1: let the pipeline schedule load() in the background
pipeline.add_analyzer(MyNewAnalyzer())
# Note: the analyser may not be fully loaded on the very next process() call

# Option 2: load it yourself first for guaranteed readiness
analyzer = MyNewAnalyzer()
await analyzer.load()
pipeline.add_analyzer(analyzer)
# analyzer is guaranteed ready immediately
```

---

## Troubleshooting

### "sounddevice not available"
Install sounddevice: `pip install sounddevice`  
On Linux you may also need: `sudo apt-get install libportaudio2`

### "Whisper model could not be loaded"
The default `base` model requires ~140 MB download on first use.  
Use a smaller model: `AIEAR_WHISPER_MODEL=tiny`  
Or disable speech: `AIEAR_SPEECH_ENABLED=false`

### High CPU usage
- Use a smaller Whisper model: `AIEAR_WHISPER_MODEL=tiny`
- Increase chunk size: `AIEAR_AUDIO_CHUNK_DURATION_S=4.0`
- Disable unused analysers: `AIEAR_EMOTION_ENABLED=false`

### "QueueFull" warnings in logs
The `AudioListener` queue is full.  Reduce `chunk_duration_s` or increase
`queue_maxsize` in the `AudioListener` constructor.
