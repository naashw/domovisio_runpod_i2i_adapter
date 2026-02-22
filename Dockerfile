# Medusa I2I - Qwen-Image-Edit-2511 (LightX2V)
# Image legere : les modeles sont telecharges au runtime sur le network volume
# Supporte 2 modes : GPU Pod (interactif) et RunPod Serverless (API)
#
# Multi-stage build : torch-base (PyTorch) -> builder (deps + LightX2V) -> runtime
# Build:  DOCKER_BUILDKIT=1 docker build --platform linux/amd64 -t medusa-i2i .
#
# GPU Pod:     docker run --gpus all -p 8888:8888 -v /workspace:/workspace medusa-i2i
# Serverless:  docker run --gpus all -e SERVERLESS=true -v /workspace:/workspace medusa-i2i

# ============================================================
# Stage 1 : torch-base (CUDA devel + Python 3.12 + PyTorch ~8 GB)
# Change rarement — cache quasi permanent
# ============================================================
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS torch-base

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

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

# --- PyTorch 2.10.x (CUDA 12.8) — layer ~8 GB, change rarement ---
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install "torch>=2.10,<2.11" torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128

# --- Core Python tooling ---
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install packaging setuptools wheel

# ============================================================
# Stage 2 : builder (deps applicatives + LightX2V)
# Invalide si requirements.txt ou LightX2V change
# ============================================================
FROM torch-base AS builder

# H100 only — force sm_90 pour TOUTES les compilations CUDA
ENV TORCH_CUDA_ARCH_LIST="9.0"

# --- Runtime deps (requirements.txt) ---
COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r /tmp/requirements.txt

# --- SageAttention 2 from source (sm_90 Hopper, attention primaire) ---
RUN git clone --depth 1 https://github.com/thu-ml/SageAttention /tmp/sageattention && \
    cd /tmp/sageattention && \
    MAX_JOBS=2 uv pip install --no-build-isolation -v .

# --- diffusers ---
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install git+https://github.com/huggingface/diffusers

# --- DeepGEMM (FP8 backend, sm_90 Hopper) ---
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install deep-gemm

RUN git clone --depth 1 https://github.com/ModelTC/LightX2V /opt/lightx2v && \
    cd /opt/lightx2v && \
    uv pip install --no-deps -v .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
    einops loguru scipy gguf prometheus_client \
    imageio imageio-ffmpeg opencv-python \
    decord av pydantic \
    numpy tqdm tokenizers ftfy aiofiles psutil

# --- Verification ABI (pas de GPU au build → lightx2v verifie au runtime) ---
RUN python -c "\
import torch; print(f'PyTorch {torch.__version__}'); \
assert torch.__version__.startswith('2.10'), f'ERREUR: torch={torch.__version__}, attendu 2.10.x'; \
import triton; print(f'Triton {triton.__version__} OK'); \
"

# ============================================================
# Stage 3 : runtime (pas de compilateur, pas de headers)
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
        curl \
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
