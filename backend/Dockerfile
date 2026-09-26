# DermaAI Backend Dockerfile - Production Container Image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV DERMAAI_DEVICE=cpu

# Install minimal system dependencies for OpenCV and healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements (CPU PyTorch wheels for fast, lightweight build)
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy model architecture definitions, frozen model checkpoints, and backend application
COPY ai/src /app/ai/src
COPY ai/models/experiment8_best_model.pt /app/ai/models/experiment8_best_model.pt
COPY ai/models/experiment13_best_model.pt /app/ai/models/experiment13_best_model.pt
COPY backend /app/backend

# Expose backend port
EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/health || exit 1

# Launch production server honoring $PORT environment variable
CMD ["sh", "-c", "python -m uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT}"]
