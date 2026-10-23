# DeerCam on a server without a GPU: MegaDetector and DINOv2 run on the CPU, the AI is an API.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DATA_DIR=/data HF_HOME=/data/models/hf TORCH_DEVICE=cpu
# OpenCV (pulled in by ultralytics) needs these
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# The CPU build of torch is a fraction of the size of the CUDA one
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY deercam deercam

VOLUME /data
EXPOSE 8600
CMD ["python", "-m", "deercam", "serve", "--host", "0.0.0.0"]
