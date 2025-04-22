FROM python:3.11-slim

# Install necessary packages: espeak-ng (for Coqui), ffmpeg (for torchaudio), git
RUN apt-get update && apt-get install -y \
    espeak-ng \
    git \
    ffmpeg \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Set environment variable for Coqui cache
ENV COQUI_TTS_CACHE=/root/.local/share/tts

# Install Python packages
RUN pip install --no-cache-dir \
    TTS[all] \
    torch \
    numpy==1.24.3

# Set working directory
WORKDIR /app

# Copy handler file
COPY handler.py .

# Clone the private Hugging Face model repo using HF_TOKEN as a build arg
ARG HF_TOKEN
RUN git clone https://user:${HF_TOKEN}@huggingface.co/santoshr24/VegetaTTSv1.5 model

# Set model paths as ENV if needed
ENV MODEL_PATH=/app/model/checkpoint_1082083.pth
ENV CONFIG_PATH=/app/model/config_tts.json

# Required for Hugging Face Inference Endpoint to discover the handler
ENV INFERENCE_SERVER_HOME=/app
