# ==========================================
# Stage 1: Builder (Compiles FlashAttention)
# ==========================================
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS builder

# Set env to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system build tools and python
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch (must match CUDA version 12.1)
RUN pip install --no-cache-dir torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Install FlashAttention 2
# --no-build-isolation is critical to link against the container's CUDA
RUN pip install --no-cache-dir "flash-attn>=2.0" --no-build-isolation

# ==========================================
# Stage 2: Runtime (Smaller Image)
# ==========================================
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies (libsndfile is needed for audio I/O)
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy python packages from builder
COPY --from=builder /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages

# Install Qwen-TTS and API dependencies
# Note: qwen-tts might pull transformers, ensure versions align.
RUN pip install --no-cache-dir \
    qwen-tts \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    pynvml \
    scipy \
    soundfile \
    accelerate \
    transformers>=4.37.0

# Copy application code
COPY . /app

# Expose API port
EXPOSE 8000

# Run Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]