# Use NVIDIA CUDA base image with Python 3.12
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
# Use Python 3.12.9 base image
FROM python:3.12.9-bullseye
# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# # Install system dependencies
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     python3.12 \
#     python3-pip \
#     python3.12-venv \
#     git \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create and activate virtual environment
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch>=2.6.0 \
    runpod>=1.7.7 \
    sentencepiece>=0.2.0 \
    transformers>=4.49.0 \
    flash-attn

# Clone and install IndicTransToolkit
RUN git clone https://github.com/AI4Bharat/IndicTransToolkit.git && \
    cd IndicTransToolkit && \
    pip install -e .

# Copy project files
COPY indicTrans_serverless.py .
COPY pyproject.toml .

# Set default command
CMD ["python3", "-c", "from indicTrans_serverless import handler; import runpod; runpod.serverless.start({'handler': handler})"]
