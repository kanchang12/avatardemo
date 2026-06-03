FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential cmake git \
    libopenblas-dev liblapack-dev \
    libx11-dev libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir git+https://github.com/ageitgey/face_recognition_models

COPY . .
RUN mkdir -p static

EXPOSE 5000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120"]
