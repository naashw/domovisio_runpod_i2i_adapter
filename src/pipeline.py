"""
MedusaI2IPipeline — LightX2V-based image editing with Qwen-Image-Edit-2511.

Pipeline FP8 pre-quantise (Lightning fuse) + SageAttention 2 pour edition d'images
immobilier/renovation sur RunPod Serverless (H100, 80GB VRAM).
FP8 via DeepGEMM (fp8-b128-deepgemm), fallback fp8-triton. Pas de CPU offload.
"""

from __future__ import annotations

import gc
import logging
import os
import tempfile

import numpy as np
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

NUM_STEPS = 8
FP8_FILENAME = "qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_8steps_v1.0.safetensors"


class MedusaI2IPipeline:
    """Pipeline I2I utilisant LightX2V avec Qwen-Image-Edit-2511 FP8 pre-quantise."""

    def __init__(self, models_dir: str) -> None:
        self.models_dir = models_dir
        self._model_path = os.path.join(models_dir, "Qwen-Image-Edit-2511")
        self._fp8_path = os.path.join(models_dir, FP8_FILENAME)
        self._pipe = None
        self._loaded = False

    def _enable_fp8(self, pipe: object) -> None:
        """Active la quantization FP8 avec fallback: fp8-b128-deepgemm → fp8-triton."""
        schemes = ["fp8-b128-deepgemm", "fp8-triton"]
        for scheme in schemes:
            try:
                log.info("Activation quantization FP8 (%s): %s", scheme, FP8_FILENAME)
                pipe.enable_quantize(
                    dit_quantized=True,
                    dit_quantized_ckpt=self._fp8_path,
                    quant_scheme=scheme,
                )
                log.info("FP8 backend actif: %s", scheme)
                return
            except Exception as exc:
                log.warning("FP8 %s echoue (%s), tentative suivante...", scheme, exc)
        raise RuntimeError("Aucun backend FP8 disponible (essayes: %s)" % ", ".join(schemes))

    def load(self) -> None:
        """Charge le pipeline LightX2V avec le checkpoint FP8 8-step.

        1. LightX2VPipeline avec model_cls qwen-image-edit-2511, task i2i
        2. FP8 quantization via enable_quantize (fp8-b128-deepgemm, fallback fp8-triton)
        3. Pas de CPU offload (H100 80GB VRAM — tout en GPU)

        create_generator() est appele dans generate() pour passer height/width dynamiquement.

        Note: torch.compile n'est pas utilise car LightX2V gere l'inference via des kernels
        Triton internes (transformer_infer). Le pipeline n'expose pas d'attribut `transformer`
        standard — le modele est accessible via pipe.runner.model apres create_generator().
        """
        from lightx2v import LightX2VPipeline

        if not os.path.isfile(self._fp8_path):
            raise FileNotFoundError(f"Checkpoint FP8 introuvable: {self._fp8_path}")

        log.info("Chargement LightX2V pipeline (%d-step)...", NUM_STEPS)

        pipe = LightX2VPipeline(
            model_path=self._model_path,
            model_cls="qwen-image-edit-2511",
            task="i2i",
        )

        self._enable_fp8(pipe)

        # H100 80GB VRAM — pas de CPU offload, tout reste en GPU

        self._pipe = pipe
        self._loaded = True
        log.info("Pipeline LightX2V pret (%d-step, FP8)", NUM_STEPS)

        # Warmup : pre-compile les kernels Triton JIT + pre-alloue la memoire CUDA
        self._warmup()

    def _warmup(self) -> None:
        """Warmup avec inference dummy (768x768) pour pre-compiler les kernels Triton."""
        import torch

        log.info("Warmup inference (768x768, pre-compilation kernels Triton)...")
        try:
            warmup_img = Image.new("RGB", (768, 768), color=(128, 128, 128))
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            warmup_img.save(tmp.name, format="PNG")
            tmp.close()

            warmup_out = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            warmup_out.close()

            self._pipe.create_generator(
                attn_mode="sage_attn2",
                infer_steps=NUM_STEPS,
                guidance_scale=1.0,
                height=768,
                width=768,
            )
            self._pipe.generate(
                seed=0,
                image_path=tmp.name,
                prompt="warmup",
                negative_prompt="",
                save_result_path=warmup_out.name,
            )
            log.info("Warmup termine")
        except Exception as e:
            log.warning("Warmup echoue (non bloquant): %s", e)
        finally:
            for f in [tmp.name, warmup_out.name]:
                if os.path.isfile(f):
                    os.unlink(f)
            gc.collect()
            torch.cuda.empty_cache()

    @staticmethod
    def _is_nan_output(image_path: str) -> bool:
        """Detecte une image corrompue (NaN/inf directs ou tout noir/blanc/uniforme)."""
        try:
            img = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32)
            if np.isnan(img).any() or np.isinf(img).any():
                return True
            mean = img.mean()
            std = img.std()
            if mean < 1.0 or mean > 254.0 or std < 2.0:
                return True
        except Exception:
            return True
        return False

    def ensure_loaded(self) -> None:
        """Charge le pipeline si pas encore fait."""
        if not self._loaded:
            self.load()

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
        import torch

        self.ensure_loaded()

        # Configurer le generateur avec la resolution cible de cette requete
        log.info(
            "Configuration generateur (sage_attn2, %d steps, %dx%d)...",
            NUM_STEPS, width, height,
        )
        self._pipe.create_generator(
            attn_mode="sage_attn2",
            infer_steps=NUM_STEPS,
            guidance_scale=1.0,
            height=height,
            width=width,
        )

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

            if self._is_nan_output(output_path):
                log.error("SA2 NaN detecte — image corrompue: %s", output_path)
        finally:
            if os.path.isfile(tmp_input.name):
                os.unlink(tmp_input.name)
            gc.collect()
            torch.cuda.empty_cache()

        # Si output_format est JPEG et LightX2V a ecrit un PNG, convertir
        if output_format == "JPEG" and output_path.lower().endswith((".jpg", ".jpeg")):
            result_img = Image.open(output_path)
            if result_img.mode == "RGBA":
                result_img = result_img.convert("RGB")
            result_img.save(output_path, format="JPEG", quality=95)

        log.info("Image sauvegardee: %s", output_path)
        return output_path
