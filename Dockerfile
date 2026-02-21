# Medusa I2I - Qwen-Image-Edit-2511 (LightX2V)
# Image legere : les modeles sont telecharges au runtime sur le network volume
# Supporte 2 modes : GPU Pod (interactif) et RunPod Serverless (API)
#
# Multi-stage build : devel (compile LightX2V + SageAttention) -> runtime (execute)
# Build:  DOCKER_BUILDKIT=1 docker build --platform linux/amd64 -t medusa-i2i .
#
# GPU Pod:     docker run --gpus all -p 8888:8888 -v /workspace:/workspace medusa-i2i
# Serverless:  docker run --gpus all -e SERVERLESS=true -v /workspace:/workspace medusa-i2i

# ============================================================
# Stage 1 : builder (compile LightX2V, SageAttention, PyTorch)
# ============================================================
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    PYTHONUNBUFFERED=1 \
    CMAKE_BUILD_PARALLEL_LEVEL=8

# --- Build dependencies + Python 3.12 ---
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev python3-pip \
        build-essential gcc ninja-build git && \
    ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    python3.12 -m venv /opt/venv && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:$PATH"

# --- PyTorch 2.10.0 (CUDA 12.8) ---
RUN pip install --no-cache-dir "torch<2.8" torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128

# --- Core Python tooling ---
RUN pip install --no-cache-dir packaging setuptools wheel

# --- Runtime deps (requirements.txt) ---
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# --- Diffusers from git (branche main, requis pour Qwen-Image-Edit-2511) ---
RUN pip install --no-cache-dir git+https://github.com/huggingface/diffusers

# --- SageAttention 2.2.0 (compile depuis source, requis CUDA 12.8+) ---
# TORCH_CUDA_ARCH_LIST requis car pas de GPU au build time (RTX 4090 = 8.9)
RUN TORCH_CUDA_ARCH_LIST="8.9" \
    pip install --no-cache-dir git+https://github.com/thu-ml/SageAttention.git@v2.2.0 --no-build-isolation

# --- LightX2V from source (--no-deps pour eviter torch<=2.8.0 downgrade) ---
RUN git clone https://github.com/ModelTC/LightX2V /opt/lightx2v && \
    cd /opt/lightx2v && \
    pip install --no-cache-dir --no-deps -v .

# --- Deps LightX2V manquantes (torch/diffusers/transformers deja installes) ---
RUN pip install --no-cache-dir \
    einops loguru scipy gguf qtorch prometheus_client \
    imageio imageio-ffmpeg opencv-python \
    decord av pydantic \
    numpy tqdm tokenizers ftfy aiofiles psutil \
    vllm==0.8.5.post1

# --- Verification ABI ---
RUN python -c "\
import torch; print(f'PyTorch {torch.__version__}'); \
assert torch.__version__.startswith('2.7'), f'ERREUR: torch={torch.__version__}, attendu 2.7.x'; \
from vllm import _custom_ops as vllm_ops; print('vllm CUTLASS ops OK'); \
import sageattention; print('SageAttention OK')"

# ============================================================
# Stage 2 : runtime (pas de compilateur, pas de headers)
# ============================================================
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    RUNPOD_INIT_TIMEOUT=600

# --- Runtime dependencies only ---
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev \
        curl aria2 \
        libgl1 libglib2.0-0 \
        gcc \
        google-perftools tini && \
    ln -sf /usr/bin/python3.12 /usr/bin/python && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# --- Copy venv depuis builder ---
COPY --from=builder /opt/venv /opt/venv

# --- Copy LightX2V configs (requis par le pipeline) ---
COPY --from=builder /opt/lightx2v/configs /opt/lightx2v/configs

# --- Application ---
RUN mkdir -p /app
COPY src/start.sh /start.sh
RUN chmod +x /start.sh

COPY src/pipeline.py /app/pipeline.py
COPY src/handler.py /app/handler.py
COPY scripts/audit_volume.py /app/audit_volume.py

WORKDIR /app

EXPOSE 8888

HEALTHCHECK NONE

ENTRYPOINT ["tini", "-s", "--"]
CMD ["/start.sh"]
