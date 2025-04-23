FROM python:3.11-slim

# Install required system packages
RUN apt-get update && apt-get install -y \
    espeak-ng \
    ffmpeg \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Set Coqui TTS cache path (optional)
ENV COQUI_TTS_CACHE=/root/.local/share/tts

# Install Python dependencies
RUN pip install --no-cache-dir \
    torch \
    TTS[all] \
    torchaudio \
    numpy==1.24.3

# Set working directory
WORKDIR /app

# This is where HF mounts your model repo with handler.py and model files
ENV INFERENCE_SERVER_HOME=/repository

EXPOSE 80