FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    git \
    curl \
    wget \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# Download YOLO weights at build time so they're baked into the image
RUN python download_models.py

# Download demo classroom image from GitHub
RUN curl -fL -o /app/activity_web/backend/static/demo_classroom.jpg \
    "https://raw.githubusercontent.com/PixxySynchronous/PRISM-AI-REPO/main/activity_web/backend/static/demo_classroom.jpg"

RUN mkdir -p /app/runtime/uploads /app/runtime/outputs /app/runtime/attendance

ENV PORT=8080
ENV ACTIVITY_WEB_RUNTIME_DIR=/app/runtime
ENV PYTHONPATH=/app

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "600", "--workers", "1", "activity_web.backend.app:app"]
