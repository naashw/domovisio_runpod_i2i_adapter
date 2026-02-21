"""Test local du pipeline Medusa I2I (sans RunPod)."""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline import MedusaI2IPipeline, PRESETS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[test] %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test local Medusa I2I")
    parser.add_argument("--image", required=True, help="Chemin image source")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), help="Preset d'edition")
    parser.add_argument("--prompt", help="Prompt libre (priorite sur preset)")
    parser.add_argument("--output", default="/tmp/medusa_i2i_test.png", help="Chemin de sortie")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--models-dir", default="/runpod-volume/models", help="Dossier modeles")
    args = parser.parse_args()

    if not args.preset and not args.prompt:
        parser.error("--preset ou --prompt requis")

    prompt = args.prompt or PRESETS[args.preset]

    pipeline = MedusaI2IPipeline(models_dir=args.models_dir)
    pipeline.load()

    # Resolution (simplifie pour test local)
    from PIL import Image  # noqa: E402

    img = Image.open(args.image)
    w, h = img.size
    max_dim = max(w, h)
    if max_dim > 768:
        scale = 768 / max_dim
        w = round(w * scale / 8) * 8
        h = round(h * scale / 8) * 8
    else:
        w = round(w / 8) * 8
        h = round(h / 8) * 8

    pipeline.generate(
        image_path=args.image,
        prompt=prompt,
        seed=args.seed,
        height=h,
        width=w,
        output_path=args.output,
    )
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
