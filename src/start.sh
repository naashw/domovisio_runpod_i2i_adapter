#!/bin/bash
set -euo pipefail

# -----------------------------------------------
# Signal handling
# -----------------------------------------------
CHILD_PIDS=()

cleanup() {
    echo "[medusa] Arret en cours..."
    for pid in "${CHILD_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait
    exit 0
}

trap cleanup SIGTERM SIGINT SIGQUIT

echo "============================================"
echo "  Medusa I2I - Qwen-Image-Edit-2511"
echo "============================================"

# -----------------------------------------------
# 1. tcmalloc (optimisation memoire)
# -----------------------------------------------
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
if [ -n "$TCMALLOC" ]; then
    export LD_PRELOAD="$TCMALLOC"
    echo "[medusa] tcmalloc charge: $TCMALLOC"
else
    echo "[medusa] tcmalloc non trouve, pas d'optimisation memoire"
fi

# -----------------------------------------------
# 2. Workspace / Network volume
# -----------------------------------------------
if [ -d "/runpod-volume" ]; then
    WORKSPACE="${WORKSPACE:-/runpod-volume}"
else
    WORKSPACE="${WORKSPACE:-/workspace}"
fi
MODELS_DIR="${WORKSPACE}/models"

echo "[medusa] Workspace: $WORKSPACE"
echo "[medusa] Models dir: $MODELS_DIR"

mkdir -p "${MODELS_DIR}"

# Exporter pour handler.py
export MODELS_DIR="$MODELS_DIR"
export VOLUME_ROOT="$WORKSPACE"

# -----------------------------------------------
# 3. Telechargement des modeles (hf_xet)
# -----------------------------------------------
echo "[medusa] Verification des modeles..."

python - "$MODELS_DIR" <<'PYEOF'
import os, sys

models_dir = sys.argv[1]

# --- Qwen-Image-Edit-2511 (sans poids transformer BF16, ~37GB) ---
qwen_dir = os.path.join(models_dir, "Qwen-Image-Edit-2511")
if os.path.isdir(qwen_dir) and os.path.isfile(os.path.join(qwen_dir, "model_index.json")):
    print("[medusa] Deja present: Qwen-Image-Edit-2511/")
else:
    print("[medusa] Telechargement: Qwen-Image-Edit-2511/ ...")
    from huggingface_hub import snapshot_download
    snapshot_download(
        "Qwen/Qwen-Image-Edit-2511",
        local_dir=qwen_dir,
        ignore_patterns=["*.gguf", "*.bin", "flax_*", "tf_*", "transformer/*.safetensors"],
    )
    print("[medusa] OK: Qwen-Image-Edit-2511/")

# --- Transformer FP8 Lightning fuse (8-step, ~20GB) ---
fp8_name = "qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_8steps_v1.0.safetensors"
fp8_path = os.path.join(models_dir, fp8_name)
if os.path.isfile(fp8_path):
    size_mb = os.path.getsize(fp8_path) / (1024 * 1024)
    print(f"[medusa] Deja present: {fp8_name} ({size_mb:.0f} MB)")
else:
    print(f"[medusa] Telechargement: {fp8_name} ...")
    from huggingface_hub import hf_hub_download
    hf_hub_download(
        repo_id="lightx2v/Qwen-Image-Edit-2511-Lightning",
        filename=fp8_name,
        local_dir=models_dir,
    )
    print(f"[medusa] OK: {fp8_name}")
PYEOF

echo "[medusa] Tous les modeles sont prets."

# -----------------------------------------------
# 4. Audit du volume (dry run — signale sans supprimer)
# -----------------------------------------------
echo "[medusa] Audit du volume..."
python /app/audit_volume.py --volume "$WORKSPACE" || true

# -----------------------------------------------
# 5. Demarrage
# -----------------------------------------------
if [ "${SERVERLESS:-}" = "true" ] || [ -n "${RUNPOD_ENDPOINT_ID:-}" ]; then
    # ===== MODE SERVERLESS =====
    echo "[medusa] Mode: SERVERLESS (RunPod API)"

    # Dossiers persistants sur le network volume
    OUTPUT_DIR="${WORKSPACE}/output"
    CACHE_DIR="${WORKSPACE}/cache"
    mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"
    export OUTPUT_VOLUME_DIR="$OUTPUT_DIR"
    export CACHE_DIR="$CACHE_DIR"
    echo "[medusa] Output dir: $OUTPUT_DIR"
    echo "[medusa] Cache dir: $CACHE_DIR"

    exec env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512 \
         TORCH_CUDA_ARCH_LIST="9.0" \
         CUDA_MODULE_LOADING=LAZY \
         TORCH_CUDNN_BENCHMARK=1 \
         python /app/handler.py

else
    # ===== MODE GPU POD =====
    echo "[medusa] Mode: GPU POD (interactif)"

    # JupyterLab avec token securise
    if command -v jupyter &>/dev/null; then
        if [ -z "${JUPYTER_TOKEN:-}" ]; then
            JUPYTER_TOKEN=$(python -c "import secrets; print(secrets.token_hex(32))")
            echo "[medusa] JupyterLab token genere: ${JUPYTER_TOKEN}"
        fi

        jupyter lab \
            --ip=0.0.0.0 \
            --port=8888 \
            --no-browser \
            --allow-root \
            --ServerApp.token="${JUPYTER_TOKEN}" \
            --ServerApp.allow_origin='*' \
            --notebook-dir="${WORKSPACE}" &
        CHILD_PIDS+=($!)
        echo "[medusa] JupyterLab demarre sur port 8888"
    fi

    echo "[medusa] GPU Pod pret. Pipeline disponible via Python."
    echo "[medusa] Pour lancer le handler manuellement : python /app/handler.py"

    # Garder le container en vie
    wait
fi
