FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential cmake git \
    libopenblas-dev liblapack-dev \
    libx11-dev libgtk-3-dev \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir git+https://github.com/ageitgey/face_recognition_models

COPY . .
RUN mkdir -p static recordings

ENV PORT=8080
EXPOSE 8080

# Cloud Run: bind to $PORT, single worker keeps face-recognition memory sane; bump timeout for Gemini+TTS
CMD exec gunicorn app:app --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 180
