# CLAUDE.md - Medusa I2I (Qwen-Image-Edit-2511)

## Projet

Pipeline Image-to-Image editing utilisant Qwen-Image-Edit-2511 avec FP8 layerwise casting.
Inference directe via Diffusers (sans ComfyUI).
Use case : edition d'images immobilier/renovation — before/after, changement de style, virtual staging.

## Contexte Technique

- **Runtime** : Diffusers (Python direct) sur RunPod Serverless (RTX 4090, 24GB VRAM)
- **Modele principal** : Qwen-Image-Edit-2511 (FP8 layerwise casting via Diffusers, ~10GB VRAM transformer)
- **Acceleration** : Lightning LoRA 4 steps fusionnee au cold start
- **Approche** : Pipeline 1 passe, true_cfg_scale=1.0 (CFG desactive, modele distille)
- **CPU offload** : `enable_model_cpu_offload()` — text encoder (14GB) et transformer (10GB) jamais en VRAM en meme temps
- **VRAM peak** : ~14GB (vs ~17-18GB avec NF4)
- **Output** : PNG/JPEG, resolution max 1024px, default 768px
- **PyTorch** : ~2.7 (CUDA 12.8)

## Structure

- `Dockerfile` — image Docker multi-stage (devel -> runtime), sans ComfyUI
- `docker-compose.yml` — lancement local GPU
- `src/start.sh` — telecharge modeles + lance handler
- `src/pipeline.py` — classe MedusaI2IPipeline (FP8 layerwise casting + Lightning LoRA)
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

- Le transformer est charge en BF16 puis converti en FP8 via `enable_layerwise_casting(storage_dtype=float8_e4m3fn, compute_dtype=bfloat16)`
- **Ordre critique** : fuse Lightning LoRA AVANT `enable_layerwise_casting` (poids doivent etre BF16 pour la fusion)
- `enable_model_cpu_offload()` gere le placement GPU — NE PAS appeler `.to("cuda")`
- `torch.Generator(device="cpu")` requis avec CPU offload (pas `"cuda"`)
- Lightning LoRA est fusionnee definitivement au load (`fuse_lora` + `unload_lora_weights`)
- `true_cfg_scale=1.0` desactive la guidance (modele distille, single forward pass)
- Resolution max 1024px, default 768px, alignement 8px
- Pas d'upscale des images sources plus petites que 768px
- `transformers>=4.46` requis (pas de pin strict comme I2V)
- Pas de warmup embeddings — Diffusers gere tout en interne
- Repo BF16 complet (~57GB) necessaire sur disque — layerwise_casting charge BF16 puis stocke FP8 en memoire
