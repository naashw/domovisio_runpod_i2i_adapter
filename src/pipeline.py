"""
MedusaI2IPipeline — LightX2V-based image editing with Qwen-Image-Edit-2511.

Pipeline FP8 pre-quantise (Lightning fuse) + SageAttention 2 pour edition d'images
immobilier/renovation sur RunPod Serverless (H100, 80GB VRAM).
FP8 via CUTLASS kernels (fp8-vllm), pas de CPU offload, torch.compile regional.
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

    def load(self, num_steps: int) -> None:
        """Charge le pipeline LightX2V avec le checkpoint FP8 pour num_steps.

        1. LightX2VPipeline avec model_cls qwen-image-edit-2511, task i2i
        2. FP8 quantization via enable_quantize (fp8-vllm, CUTLASS kernels)
        3. Pas de CPU offload (H100 80GB VRAM — tout en GPU)
        4. torch.compile regional sur blocs DiT (warmup au cold start)

        create_generator() est appele dans generate() pour passer height/width dynamiquement.
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

        log.info("Activation quantization FP8 (CUTLASS): %s", os.path.basename(fp8_path))
        pipe.enable_quantize(
            dit_quantized=True,
            dit_quantized_ckpt=fp8_path,
            quant_scheme="fp8-vllm",
        )

        # H100 80GB VRAM — pas de CPU offload, tout reste en GPU

        # torch.compile regional sur les blocs DiT
        self._apply_torch_compile(pipe)

        self._pipe = pipe
        self._loaded_steps = num_steps
        log.info("Pipeline LightX2V pret (%d-step, FP8 CUTLASS)", num_steps)

        # Warmup torch.compile avec inference dummy
        self._warmup()

    @staticmethod
    def _apply_torch_compile(pipe: object) -> None:
        """Applique torch.compile regional sur les blocs DiT du transformer."""
        import torch

        try:
            transformer = getattr(pipe, "transformer", None)
            if transformer is None:
                log.warning("torch.compile: transformer non trouve, skip")
                return

            compiled = 0
            for name, module in transformer.named_children():
                if "block" in name.lower() or "layer" in name.lower():
                    try:
                        compiled_module = torch.compile(
                            module,
                            backend="inductor",
                            mode="max-autotune-no-cudagraphs",
                            fullgraph=False,
                        )
                        setattr(transformer, name, compiled_module)
                        compiled += 1
                    except Exception as e:
                        log.warning("torch.compile skip bloc %s: %s", name, e)

            if compiled:
                log.info("torch.compile: %d blocs DiT compiles", compiled)
            else:
                log.warning("torch.compile: aucun bloc DiT trouve")
        except Exception as e:
            log.warning("torch.compile non disponible, skip: %s", e)

    def _warmup(self) -> None:
        """Warmup torch.compile avec une inference dummy (768x768, image grise)."""
        import torch

        log.info("Warmup torch.compile (768x768, image grise)...")
        try:
            warmup_img = Image.new("RGB", (768, 768), color=(128, 128, 128))
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            warmup_img.save(tmp.name, format="PNG")
            tmp.close()

            warmup_out = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            warmup_out.close()

            self._pipe.create_generator(
                attn_mode="sage_attn2",
                infer_steps=self._loaded_steps,
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
        """Detecte une image corrompue (NaN → tout noir/blanc/uniforme)."""
        try:
            img = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32)
            mean = img.mean()
            std = img.std()
            if mean < 1.0 or mean > 254.0 or std < 2.0:
                return True
        except Exception:
            return True
        return False

    def ensure_steps(self, num_steps: int) -> None:
        """Recharge le pipeline si le checkpoint FP8 (4-step vs 8-step) change."""
        if self._loaded_steps != num_steps:
            log.info("Swap FP8: %d-step -> %d-step", self._loaded_steps, num_steps)
            # Liberer l'ancien pipeline avant de charger le nouveau
            import torch

            self._pipe = None
            gc.collect()
            torch.cuda.empty_cache()
            log.info("Memoire liberee (gc + cuda.empty_cache)")
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
        self.ensure_steps(num_steps)

        # Configurer le generateur avec la resolution cible de cette requete
        log.info(
            "Configuration generateur (sage_attn2, %d steps, %dx%d)...",
            num_steps, width, height,
        )
        self._pipe.create_generator(
            attn_mode="sage_attn2",
            infer_steps=num_steps,
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

            # Fallback Flash Attention 3 si SA2 produit des NaN
            if self._is_nan_output(output_path):
                log.warning("SA2 NaN detecte — fallback flash_attn3")
                self._pipe.create_generator(
                    attn_mode="flash_attn3",
                    infer_steps=num_steps,
                    guidance_scale=1.0,
                    height=height,
                    width=width,
                )
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
