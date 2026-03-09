# Deployment Guide

## Development

```bash
pip install -e ".[dev]"
ai-ear serve
```

---

## Docker

### Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# System dependencies for sounddevice / librosa
RUN apt-get update && apt-get install -y \
    libportaudio2 \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
COPY ai_ear/ ./ai_ear/

RUN pip install --no-cache-dir .

EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["ai-ear", "serve", "--host", "0.0.0.0", "--port", "8080"]
```

### Build & Run

```bash
docker build -t ai-ear:latest .

docker run -p 8080:8080 \
  -e AIEAR_WHISPER_MODEL=base \
  -e AIEAR_EMOTION_ENABLED=false \
  ai-ear:latest
```

---

## Docker Compose (with GPU)

```yaml
version: "3.9"
services:
  ai-ear:
    build: .
    ports:
      - "8080:8080"
    environment:
      AIEAR_WHISPER_MODEL: small
      AIEAR_WHISPER_DEVICE: cuda
      AIEAR_EMOTION_ENABLED: "true"
      AIEAR_MUSIC_ENABLED: "true"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
```

---

## Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-ear
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ai-ear
  template:
    metadata:
      labels:
        app: ai-ear
    spec:
      containers:
      - name: ai-ear
        image: ai-ear:latest
        ports:
        - containerPort: 8080
        env:
        - name: AIEAR_WHISPER_MODEL
          value: "base"
        - name: AIEAR_EMOTION_ENABLED
          value: "false"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
```

---

## Production Recommendations

### Hardware

| Configuration | Recommended |
|---|---|
| Speech only (tiny model) | 2 CPU cores, 1 GB RAM |
| Speech + Environment | 4 CPU cores, 2 GB RAM |
| Full stack (GPU) | 4 CPU + NVIDIA RTX 3060+, 8 GB RAM |

### Performance Tuning

```bash
# Use a smaller Whisper model
AIEAR_WHISPER_MODEL=tiny

# Increase chunk size to reduce call frequency (trade latency for throughput)
AIEAR_AUDIO_CHUNK_DURATION_S=4.0

# Disable unused analysers
AIEAR_EMOTION_ENABLED=false
AIEAR_MUSIC_ENABLED=false
```

### Scaling

The AI Ear is designed as a single-process service.  To scale horizontally:

1. Run multiple instances behind a load balancer (each with its own state).
2. Share `AuralMemory` state via an external store (Redis, PostgreSQL) by
   subclassing `AuralMemory` with a custom backend.
3. Use a message broker (Kafka, RabbitMQ) to distribute `AudioChunk` objects
   to a pool of workers.

### Security

- Set `AIEAR_API_CORS_ORIGINS` to explicit domain names in production.
- Place behind a TLS-terminating reverse proxy (nginx, Caddy).
- For WebSocket TLS, use `wss://` in your clients.
- The server does not perform authentication; add OAuth2/JWT middleware as
  needed.

### Logging

```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
)
```

Or use `PYTHONLOGLEVEL=DEBUG` for verbose output from all components.

---

## Model Caching

Whisper models are cached by the `openai-whisper` library (default: `~/.cache/whisper/`).  
HuggingFace models are cached in `~/.cache/huggingface/hub/`.

In Docker, mount a persistent volume to avoid re-downloading on each start:

```bash
docker run -p 8080:8080 \
  -v /data/model-cache/whisper:/root/.cache/whisper \
  -v /data/model-cache/hf:/root/.cache/huggingface \
  ai-ear:latest
```
