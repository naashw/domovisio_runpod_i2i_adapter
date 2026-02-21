# CLAUDE.md - Medusa I2I (Qwen-Image-Edit-2511)

## Projet

Pipeline Image-to-Image editing utilisant Qwen-Image-Edit-2511 avec NF4 quantization.
Inference directe via Diffusers (sans ComfyUI).
Use case : edition d'images immobilier/renovation — before/after, changement de style, virtual staging.

## Contexte Technique

- **Runtime** : Diffusers (Python direct) sur RunPod Serverless (RTX 4090, 24GB VRAM)
- **Modele principal** : Qwen-Image-Edit-2511 (NF4 quantize via bitsandbytes, ~17-18GB VRAM transformer)
- **Acceleration** : Lightning LoRA 4 steps fusionnee au cold start
- **Approche** : Pipeline 1 passe, true_cfg_scale=1.0 (CFG desactive, modele distille)
- **Output** : PNG/JPEG, resolution max 1024px, default 768px
- **PyTorch** : ~2.7 (CUDA 12.8)

## Structure

- `Dockerfile` — image Docker multi-stage (devel -> runtime), sans ComfyUI
- `docker-compose.yml` — lancement local GPU
- `src/start.sh` — telecharge modeles + lance handler
- `src/pipeline.py` — classe MedusaI2IPipeline (NF4 + Lightning LoRA)
- `src/handler.py` — handler RunPod serverless (API simplifiee)
- `scripts/test_local.py` — test hors RunPod

## API Input

Avec preset :
```json
{
  "input": {
    "image": "https://example.com/photo.jpg",
    "preset": "renovation-after",
    "seed": 42,
    "output_format": "PNG"
  }
}
```

Avec prompt libre :
```json
{
  "input": {
    "image": "base64...",
    "prompt": "Add modern kitchen cabinets and granite countertops",
    "seed": 12345
  }
}
```

## API Output

```json
{
  "images": [{
    "filename": "medusa_i2i_abc123.png",
    "content_type": "image",
    "size_mb": 1.2,
    "volume_path": "/runpod-volume/output/abc123/medusa_i2i_abc123.png",
    "s3_key": "output/abc123/medusa_i2i_abc123.png"
  }]
}
```

## Presets Disponibles

- `renovation-after` : renovation terminee, finitions modernes
- `style-modern` : style contemporain moderne
- `add-furniture` : ajout de meubles modernes
- `style-scandinavian` : style scandinave
- `style-industrial` : style loft industriel
- `virtual-staging` : staging immobilier
- `exterior-refresh` : rafraichissement facade
- `lighting-warm` : eclairage chaud ambiant
- `declutter` : desencombrement
- `enhance-quality` : amelioration qualite photo

## Conventions

- Prefixe output : `medusa_i2i`
- Reponse API : `images[]` avec `s3_key` + `volume_path`
- Input images : URL https OU base64
- Output images : `/runpod-volume/output/{job_id}/`
- Dedup cache par hash : `/runpod-volume/cache/dedup/{hash}/`

## Points d'Attention

- Le transformer est charge en NF4 (bitsandbytes) pour tenir dans 24GB VRAM
- Lightning LoRA est fusionnee definitivement au load (`fuse_lora` + `unload_lora_weights`)
- `true_cfg_scale=1.0` desactive la guidance (modele distille, single forward pass)
- Resolution max 1024px, default 768px, alignement 8px
- Pas d'upscale des images sources plus petites que 768px
- `transformers>=4.46` requis (pas de pin strict comme I2V)
- Pas de warmup embeddings — Diffusers gere tout en interne
- Pas de baked checkpoint — NF4 est charge directement
