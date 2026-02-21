"""
MedusaI2IPipeline — Diffusers-based image editing with Qwen-Image-Edit-2511.

Pipeline FP8 pre-quantise (Lightning fuse) pour edition d'images
immobilier/renovation sur RunPod Serverless (RTX 4090, 24GB VRAM).
"""

from __future__ import annotations

import logging
import os

import torch
from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel
from PIL import Image
from safetensors.torch import load_file

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

LIGHTNING_STEPS = 4
LIGHTNING_TRUE_CFG_SCALE = 1.0


class MedusaI2IPipeline:
    """Pipeline I2I utilisant Diffusers avec Qwen-Image-Edit-2511 FP8 pre-quantise."""

    def __init__(self, models_dir: str) -> None:
        self.models_dir = models_dir
        self._model_path = os.path.join(models_dir, "Qwen-Image-Edit-2511")
        self._fp8_path = os.path.join(
            models_dir,
            "qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_4steps_v1.0.safetensors",
        )
        self._pipe: QwenImageEditPlusPipeline | None = None

    @staticmethod
    def _log_vram(label: str) -> None:
        """Log l'utilisation VRAM courante (alloue + reserve)."""
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 2**30
            rsvd = torch.cuda.memory_reserved() / 2**30
            log.info("VRAM [%s]: %.2fGB alloc, %.2fGB reserved", label, alloc, rsvd)

    def load(self) -> None:
        """Charge le pipeline avec transformer FP8 pre-quantise (Lightning fuse).

        1. Transformer FP8 via from_config + load_state_dict (pas d'alloc BF16)
        2. Pipeline complet (text encoder BF16 sur CPU)
        3. enable_model_cpu_offload — pas de .to("cuda")
        """
        log.info("Chargement transformer FP8 pre-quantise...")
        config = QwenImageTransformer2DModel.load_config(
            self._model_path, subfolder="transformer",
        )
        with torch.device("meta"):
            transformer = QwenImageTransformer2DModel.from_config(config)
        state_dict = load_file(self._fp8_path, device="cpu")
        # Filtrer les weight_scale (metadata quantization, pas attendu par le modele)
        scale_keys = [k for k in state_dict if k.endswith(".weight_scale")]
        for k in scale_keys:
            del state_dict[k]
        if scale_keys:
            log.info("Filtre %d cles weight_scale du state_dict", len(scale_keys))
        transformer.load_state_dict(state_dict, assign=True, strict=True)
        log.info("Transformer FP8 charge (%s)", os.path.basename(self._fp8_path))

        log.info("Chargement pipeline complet...")
        self._pipe = QwenImageEditPlusPipeline.from_pretrained(
            self._model_path,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )

        # torch.compile : ~2.4x speedup (opt-in)
        if os.environ.get("TORCH_COMPILE", "").lower() in ("1", "true"):
            log.info("Compilation torch.compile du transformer...")
            try:
                self._pipe.transformer = torch.compile(
                    self._pipe.transformer, mode="reduce-overhead",
                )
                log.info("torch.compile actif (premiere inference plus lente)")
            except Exception as e:
                log.warning("torch.compile echoue, fallback sans compilation: %s", e)

        # CPU offload : text encoder (14GB) et transformer (10GB) jamais en VRAM en meme temps
        self._pipe.enable_model_cpu_offload()

        self._log_vram("pipeline pret")

    @torch.inference_mode()
    def generate(
        self,
        image_path: str,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        output_path: str,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        output_format: str = "PNG",
    ) -> str:
        """Genere une image editee et la sauvegarde.

        Returns:
            Chemin du fichier de sortie.
        """
        if self._pipe is None:
            raise RuntimeError("Pipeline non charge — appeler load() d'abord")

        img = Image.open(image_path).convert("RGB").resize(
            (width, height), Image.Resampling.LANCZOS,
        )
        generator = torch.Generator(device="cpu").manual_seed(seed)

        result = self._pipe(
            image=img,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=LIGHTNING_STEPS,
            true_cfg_scale=LIGHTNING_TRUE_CFG_SCALE,
            height=height,
            width=width,
            generator=generator,
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result.images[0].save(output_path, format=output_format)
        log.info("Image sauvegardee: %s", output_path)
        return output_path
