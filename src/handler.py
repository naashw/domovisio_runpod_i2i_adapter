"""
Handler RunPod Serverless pour Medusa I2I (Qwen-Image-Edit-2511).

API simplifiee : image + preset/prompt → image editee PNG/JPEG.
Reprend les fonctionnalites cles du handler I2V :
  - Dedup cache par hash
  - Sauvegarde outputs sur network volume
  - Resolution dynamique (aspect ratio preserve, align 8px)
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import random
import shutil
import tempfile
import time

import requests
import runpod
from PIL import Image

from pipeline import MedusaI2IPipeline, PRESETS, DEFAULT_NEGATIVE_PROMPT

logging.basicConfig(level=logging.INFO, format="[handler] %(message)s")
log = logging.getLogger("handler")

# --- Constantes ---

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
OUTPUT_VOLUME_DIR = os.environ.get("OUTPUT_VOLUME_DIR", "/runpod-volume/output")
CACHE_DIR = os.environ.get("CACHE_DIR", "/runpod-volume/cache")
VOLUME_ROOT = os.environ.get("VOLUME_ROOT", "/runpod-volume")
MODELS_DIR = os.environ.get("MODELS_DIR", "/runpod-volume/models")

MAX_DIMENSION = 1024
DEFAULT_RESOLUTION = 768


# --- Utilitaires ---


def get_disk_usage_mb(path: str = "/") -> float:
    """Retourne l'espace utilise en MB."""
    stat = shutil.disk_usage(path)
    return stat.used / (1024 * 1024)


def resolve_image(image_data: str) -> str:
    """URL ou base64 → chemin fichier temporaire."""
    if image_data.startswith(("http://", "https://")):
        log.info("Telechargement image: %s", image_data[:80])
        resp = requests.get(image_data, timeout=30)
        resp.raise_for_status()
        data = resp.content
        log.info("Image telechargee: %.0f KB", len(data) / 1024)
    else:
        data = base64.b64decode(image_data)
        log.info("Image base64 decodee: %.0f KB", len(data) / 1024)

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.write(data)
    tmp.close()
    return tmp.name


def compute_target_resolution(
    image_path: str,
    default_resolution: int = DEFAULT_RESOLUTION,
    max_dimension: int = MAX_DIMENSION,
) -> tuple[int, int]:
    """Resolution cible en preservant l'aspect ratio.

    - Pas d'upscale (image plus petite que default_resolution → garder telle quelle)
    - Scale down si plus grande que default_resolution
    - Aligne sur 8px pour compatibilite VAE
    - Clamp au max_dimension (1024)

    Returns:
        (height, width) alignes sur 8 pixels.
    """
    img = Image.open(image_path)
    w, h = img.size
    max_dim = max(w, h)

    if max_dim > default_resolution:
        scale = default_resolution / max_dim
        w = round(w * scale / 8) * 8
        h = round(h * scale / 8) * 8
    else:
        w = round(w / 8) * 8
        h = round(h / 8) * 8

    w = max(min(w, max_dimension), 8)
    h = max(min(h, max_dimension), 8)
    return h, w


def compute_input_hash(job_input: dict) -> str:
    """Hash deterministe de l'input pour dedup."""
    raw = json.dumps(job_input, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def lookup_cache(input_hash: str) -> list[dict] | None:
    """Cherche des outputs existants dans le cache volume pour ce hash."""
    cache_path = os.path.join(CACHE_DIR, "dedup", input_hash)
    if not os.path.isdir(cache_path):
        return None

    files = [
        f for f in os.listdir(cache_path)
        if os.path.isfile(os.path.join(cache_path, f))
    ]
    if not files:
        return None

    outputs = []
    for filename in files:
        filepath = os.path.join(cache_path, filename)
        ext = os.path.splitext(filename)[1].lower()
        file_type = "image" if ext in IMAGE_EXTENSIONS else "file"
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        s3_key = os.path.relpath(filepath, VOLUME_ROOT)
        outputs.append({
            "filename": filename,
            "content_type": file_type,
            "size_mb": round(size_mb, 2),
            "volume_path": filepath,
            "s3_key": s3_key,
        })
    return outputs


def save_to_cache(input_hash: str, source_file: str) -> None:
    """Copie l'output dans le cache volume indexe par hash."""
    cache_path = os.path.join(CACHE_DIR, "dedup", input_hash)
    os.makedirs(cache_path, exist_ok=True)
    filename = os.path.basename(source_file)
    shutil.copy2(source_file, os.path.join(cache_path, filename))


def collect_output(source_file: str, job_id: str) -> dict:
    """Copie l'output sur le volume et retourne les metadonnees."""
    dest_dir = os.path.join(OUTPUT_VOLUME_DIR, job_id)
    os.makedirs(dest_dir, exist_ok=True)

    filename = os.path.basename(source_file)
    dest_path = os.path.join(dest_dir, filename)
    shutil.copy2(source_file, dest_path)

    ext = os.path.splitext(filename)[1].lower()
    file_type = "image" if ext in IMAGE_EXTENSIONS else "file"
    size_mb = os.path.getsize(dest_path) / (1024 * 1024)
    s3_key = os.path.relpath(dest_path, VOLUME_ROOT)

    log.info("%s: %.1f MB (%s) -> %s/", filename, size_mb, file_type, dest_dir)
    return {
        "filename": filename,
        "content_type": file_type,
        "size_mb": round(size_mb, 2),
        "volume_path": dest_path,
        "s3_key": s3_key,
    }


# --- Handler ---

pipeline: MedusaI2IPipeline | None = None


def handler(job: dict) -> dict:
    """Handler RunPod pour edition d'image I2I."""
    global pipeline

    job_id = job.get("id", f"unknown-{int(time.time())}")
    job_input = job.get("input", {})

    # --- Dedup cache EN PREMIER (filesystem only, pas besoin du pipeline) ---
    input_hash = compute_input_hash(job_input)
    cached = lookup_cache(input_hash)
    if cached:
        log.info("Cache hit (%s) - %d fichier(s), skip execution", input_hash, len(cached))
        return {"images": cached, "cached": True}

    # --- Parse input ---
    image_data = job_input.get("image")
    if not image_data:
        return {"error": "Le champ 'image' est requis (URL https ou base64)"}

    preset = job_input.get("preset")
    prompt = job_input.get("prompt")

    if not preset and not prompt:
        return {"error": "Le champ 'preset' ou 'prompt' est requis"}

    if preset and preset not in PRESETS:
        return {"error": f"Preset inconnu: {preset}. Choix: {list(PRESETS.keys())}"}

    final_prompt = prompt or PRESETS[preset]
    negative_prompt = job_input.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
    seed = job_input.get("seed", random.randint(0, 2**32 - 1))
    output_format = job_input.get("output_format", "PNG").upper()
    if output_format not in ("PNG", "JPEG"):
        output_format = "PNG"

    # --- Lazy init pipeline (seulement si cache miss) ---
    if pipeline is None:
        log.info("Pipeline non init — cold start...")
        pipeline = init_pipeline()

    disk_before = get_disk_usage_mb()
    log.info("Job %s - Disque avant: %.0f MB", job_id, disk_before)

    # --- Resolve image ---
    tmp_image = None
    try:
        tmp_image = resolve_image(image_data)

        # --- Resolution dynamique ---
        height, width = compute_target_resolution(tmp_image)
        log.info("Resolution cible: %dx%d (aspect ratio preserve, align 8px)", width, height)

        # --- Generate ---
        output_dir = tempfile.mkdtemp(prefix="medusa_")
        ext = ".jpg" if output_format == "JPEG" else ".png"
        output_filename = f"medusa_i2i_{job_id}{ext}"
        output_path = os.path.join(output_dir, output_filename)

        start_time = time.time()
        pipeline.generate(
            image_path=tmp_image,
            prompt=final_prompt,
            seed=seed,
            height=height,
            width=width,
            output_path=output_path,
            negative_prompt=negative_prompt,
            output_format=output_format,
        )
        elapsed = time.time() - start_time
        log.info("Generation terminee: %.1fs", elapsed)

        # --- Collect output ---
        output_meta = collect_output(output_path, job_id)

        # --- Save to dedup cache ---
        save_to_cache(input_hash, output_path)
        log.info("Cache sauvegarde (%s)", input_hash)

        # --- Cleanup ---
        shutil.rmtree(output_dir, ignore_errors=True)

        disk_after = get_disk_usage_mb()
        freed = disk_before - disk_after
        log.info("Disque apres: %.0f MB (libere: %.0f MB)", disk_after, freed)

        return {"images": [output_meta]}

    except Exception as e:
        log.error("Erreur generation: %s", e, exc_info=True)
        return {"error": str(e)}

    finally:
        if tmp_image and os.path.isfile(tmp_image):
            os.unlink(tmp_image)


# --- Init & Start ---


def init_pipeline() -> MedusaI2IPipeline:
    """Initialise le pipeline au demarrage du worker."""
    log.info("Initialisation MedusaI2IPipeline...")
    p = MedusaI2IPipeline(models_dir=MODELS_DIR)
    p.load()
    log.info("Pipeline pret.")
    return p


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
