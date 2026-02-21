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
# 3. Fonction de telechargement
# -----------------------------------------------
download_model() {
    local url="$1"
    local dest_dir="$2"
    local min_size="${3:-1000000}"
    local filename
    filename=$(basename "$url")
    local filepath="${dest_dir}/${filename}"

    if [ -f "$filepath" ]; then
        local size
        size=$(stat -c%s "$filepath" 2>/dev/null || echo 0)
        if [ "$size" -gt "$min_size" ]; then
            echo "[medusa] Deja present: $filename ($(numfmt --to=iec "$size"))"
            return 0
        fi
        echo "[medusa] Corrompu (${size}B < min ${min_size}B), re-telechargement: $filename"
    fi

    echo "[medusa] Telechargement: $filename"
    aria2c -x 16 -s 16 -k 1M \
        -d "$dest_dir" -o "$filename" \
        "$url" \
        --console-log-level=error \
        --summary-interval=0 \
        --check-certificate=true \
        --file-allocation=none \
        --max-tries=3 \
        --retry-wait=5 \
        --timeout=600

    local final_size
    final_size=$(stat -c%s "$filepath" 2>/dev/null || echo 0)
    if [ "$final_size" -lt "$min_size" ]; then
        echo "[medusa] ERREUR: $filename trop petit (${final_size}B < min ${min_size}B)"
        return 1
    fi
}

# -----------------------------------------------
# 4. Telechargement des modeles (en parallele)
# -----------------------------------------------
echo "[medusa] Demarrage des telechargements..."
DOWNLOAD_PIDS=()

# --- Qwen-Image-Edit-2511 (sans poids transformer BF16, ~37GB) ---
QWEN_DIR="${MODELS_DIR}/Qwen-Image-Edit-2511"
if [ -d "$QWEN_DIR" ] && [ -f "$QWEN_DIR/model_index.json" ]; then
    echo "[medusa] Deja present: Qwen-Image-Edit-2511/"
else
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen-Image-Edit-2511', local_dir='$QWEN_DIR',
    ignore_patterns=['*.gguf', '*.bin', 'flax_*', 'tf_*', 'transformer/*.safetensors'])
" &
    DOWNLOAD_PIDS+=($!)
fi

# --- Transformer FP8 pre-quantise avec Lightning fuse (~20GB) ---
download_model \
    "https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning.safetensors" \
    "${MODELS_DIR}" 10000000000 &
DOWNLOAD_PIDS+=($!)

# Attendre tous les telechargements et verifier les codes retour
echo "[medusa] Attente fin des telechargements..."
DOWNLOAD_FAILED=0
for pid in "${DOWNLOAD_PIDS[@]}"; do
    if ! wait "$pid"; then
        DOWNLOAD_FAILED=1
    fi
done

if [ "$DOWNLOAD_FAILED" -eq 1 ]; then
    echo "[medusa] ERREUR: Un ou plusieurs telechargements ont echoue"
    exit 1
fi
echo "[medusa] Tous les modeles sont prets."

# -----------------------------------------------
# 5. Audit du volume (dry run — signale sans supprimer)
# -----------------------------------------------
echo "[medusa] Audit du volume..."
python /app/audit_volume.py --volume "$WORKSPACE" || true

# -----------------------------------------------
# 6. Demarrage
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

    exec env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python /app/handler.py

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
