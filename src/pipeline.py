"""
MedusaI2IPipeline — LightX2V-based image editing with Qwen-Image-Edit-2511.

Pipeline FP8 pre-quantise (Lightning fuse) + SageAttention pour edition d'images
immobilier/renovation sur RunPod Serverless (RTX 4090, 24GB VRAM).
"""

from __future__ import annotations

import logging
import os
import tempfile

from PIL import Image

log = logging.getLogger("medusa")

# --- Presets immobilier/renovation ---

PRESETS: dict[str, str] = {
    "renovation-after": (
        "Show the completed renovation of this room, modern finishes, "
        "clean walls, new flooring, professional result"
    ),
    "style-modern": (
        "Transform this interior to modern contemporary style, clean lines, "
        "neutral tones, minimalist furniture"
    ),
    "add-furniture": (
        "Add modern furniture to this empty room, sofa, table, chairs, "
        "tasteful decoration, realistic placement"
    ),
    "style-scandinavian": (
        "Transform this interior to Scandinavian style, light wood, "
        "white walls, minimal decor, cozy atmosphere"
    ),
    "style-industrial": (
        "Transform this interior to industrial loft style, exposed brick, "
        "metal fixtures, raw materials"
    ),
    "virtual-staging": (
        "Stage this empty room with furniture, decor, plants, warm lighting, "
        "ready for real estate listing"
    ),
    "exterior-refresh": (
        "Refresh the exterior of this building, clean facade, landscaping, "
        "modern entrance"
    ),
    "lighting-warm": (
        "Change the lighting to warm ambient evening light, cozy atmosphere, "
        "soft shadows"
    ),
    "declutter": (
        "Remove clutter from this room, clean organized space, "
        "keep furniture and structure"
    ),
    "enhance-quality": (
        "Enhance image quality, improve sharpness, better lighting, "
        "professional real estate photography"
    ),
}

DEFAULT_NEGATIVE_PROMPT = (
    "blurry, out of focus, low quality, distorted, watermark, "
    "logo, text, subtitle, banner, signature, username, "
    "compressed artifacts, jpeg artifacts, noise, grainy"
)

VALID_NUM_STEPS = (4, 8)


class MedusaI2IPipeline:
    """Pipeline I2I utilisant LightX2V avec Qwen-Image-Edit-2511 FP8 pre-quantise."""

    def __init__(self, models_dir: str) -> None:
        self.models_dir = models_dir
        self._model_path = os.path.join(models_dir, "Qwen-Image-Edit-2511")
        self._fp8_paths: dict[int, str] = {
            4: os.path.join(
                models_dir,
                "qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_4steps_v1.0.safetensors",
            ),
            8: os.path.join(
                models_dir,
                "qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_8steps_v1.0.safetensors",
            ),
        }
        self._pipe = None
        self._loaded_steps: int | None = None

    def load(self, num_steps: int = 4) -> None:
        """Charge le pipeline LightX2V avec le checkpoint FP8 pour num_steps.

        1. LightX2VPipeline avec model_cls qwen-image-edit-2511, task i2i
        2. FP8 quantization via enable_quantize (fp8-sgl)
        3. CPU offload par blocs (text encoder + transformer jamais en VRAM ensemble)
        4. SageAttention via create_generator (sage_attn2)
        """
        from lightx2v import LightX2VPipeline

        fp8_path = self._fp8_paths[num_steps]
        if not os.path.isfile(fp8_path):
            raise FileNotFoundError(f"Checkpoint FP8 introuvable: {fp8_path}")

        log.info("Chargement LightX2V pipeline (%d-step)...", num_steps)

        pipe = LightX2VPipeline(
            model_path=self._model_path,
            model_cls="qwen-image-edit-2511",
            task="i2i",
        )

        log.info("Activation quantization FP8: %s", os.path.basename(fp8_path))
        pipe.enable_quantize(
            dit_quantized=True,
            dit_quantized_ckpt=fp8_path,
            quant_scheme="fp8-sgl",
        )

        log.info("Activation CPU offload (block granularity)...")
        pipe.enable_offload(
            cpu_offload=True,
            offload_granularity="block",
            text_encoder_offload=True,
        )

        # Tenter sage_attn2, fallback sur torch (SDPA) si indisponible
        attn_mode = "sage_attn2"
        try:
            import sageattention  # noqa: F401
        except ImportError:
            log.warning("SageAttention indisponible, fallback sur torch SDPA")
            attn_mode = "torch"

        log.info("Configuration generateur (%s, %d steps)...", attn_mode, num_steps)
        pipe.create_generator(
            attn_mode=attn_mode,
            resize_mode="adaptive",
            infer_steps=num_steps,
            guidance_scale=1.0,
        )

        self._pipe = pipe
        self._loaded_steps = num_steps
        log.info("Pipeline LightX2V pret (%d-step, FP8, SageAttention)", num_steps)

    def ensure_steps(self, num_steps: int) -> None:
        """Recharge le pipeline si le nombre de steps change."""
        if self._loaded_steps != num_steps:
            log.info("Swap FP8: %d-step -> %d-step", self._loaded_steps, num_steps)
            self.load(num_steps)

    def generate(
        self,
        image_path: str,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        output_path: str,
        num_steps: int = 4,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        output_format: str = "PNG",
    ) -> str:
        """Genere une image editee et la sauvegarde.

        Returns:
            Chemin du fichier de sortie.
        """
        if self._pipe is None:
            raise RuntimeError("Pipeline non charge — appeler load() d'abord")

        self.ensure_steps(num_steps)

        # Resize image source au format cible (LANCZOS pour qualite photo)
        img = Image.open(image_path).convert("RGB").resize(
            (width, height), Image.Resampling.LANCZOS,
        )
        tmp_input = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        try:
            img.save(tmp_input.name, format="PNG")
            tmp_input.close()

            # LightX2V ecrit directement le resultat au save_result_path
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Forcer l'extension selon output_format pour save_result_path
            self._pipe.generate(
                seed=seed,
                image_path=tmp_input.name,
                prompt=prompt,
                negative_prompt=negative_prompt,
                save_result_path=output_path,
            )
        finally:
            if os.path.isfile(tmp_input.name):
                os.unlink(tmp_input.name)

        # Si output_format est JPEG et LightX2V a ecrit un PNG, convertir
        if output_format == "JPEG" and output_path.lower().endswith((".jpg", ".jpeg")):
            result_img = Image.open(output_path)
            if result_img.mode == "RGBA":
                result_img = result_img.convert("RGB")
            result_img.save(output_path, format="JPEG", quality=95)

        log.info("Image sauvegardee: %s", output_path)
        return output_path
