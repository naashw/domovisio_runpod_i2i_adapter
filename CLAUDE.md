# CLAUDE.md - Medusa I2I (Qwen-Image-Edit-2511)
<!-- mirroring test -->

## Projet

Pipeline Image-to-Image editing utilisant Qwen-Image-Edit-2511 avec transformer FP8 pre-quantise (Lightning fuse).
Inference via LightX2V + SageAttention 2 + CUTLASS FP8 (sans ComfyUI).
Use case : edition d'images immobilier/renovation — before/after, changement de style, virtual staging.

## Contexte Technique

- **Runtime** : LightX2V (Python direct) sur RunPod Serverless (H100, 80GB VRAM)
- **Modele principal** : Qwen-Image-Edit-2511 (transformer FP8 pre-quantise avec Lightning fuse)
- **Fichiers FP8** : `qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_{4,8}steps_v1.0.safetensors` (~20GB chaque) de `lightx2v/Qwen-Image-Edit-2511-Lightning`
- **LightX2V** : Gere nativement FP8, SageAttention 2, Flash Attention 3
- **FP8 backend** : `fp8-vllm` (CUTLASS kernels, optimise Hopper TensorCore)
- **Attention** : SageAttention 2 (Hopper-optimise, fallback Flash Attention 3 si NaN)
- **Approche** : Pipeline 1 passe, guidance_scale=1.0 (CFG desactive, modele distille), 4 ou 8 steps configurable
- **CPU offload** : desactive (H100 80GB VRAM, tout en GPU)
- **torch.compile** : regional sur blocs DiT (inductor, max-autotune-no-cudagraphs), warmup au cold start
- **VRAM peak** : ~40GB (tout en GPU, pas d'offload)
- **Output** : PNG/JPEG, resolution max 1024px, default 768px
- **PyTorch** : 2.10.x (CUDA 12.8) — installe avec `--no-deps` pour eviter les conflits

## Structure

- `Dockerfile` — image Docker multi-stage (torch-base -> builder -> runtime)
- `docker-compose.yml` — lancement local GPU
- `src/start.sh` — telecharge modeles (2 checkpoints FP8) + lance handler
- `src/pipeline.py` — classe MedusaI2IPipeline (LightX2V, FP8 CUTLASS, sage_attn2, torch.compile, resolution dynamique par requete)
- `src/handler.py` — handler RunPod serverless (API simplifiee)
- `scripts/test_local.py` — test hors RunPod
- `scripts/audit_volume.py` — audit fichiers volume

## API Input

Avec preset :
```json
{
  "input": {
    "image": "https://example.com/photo.jpg",
    "preset": "renovation-after",
    "seed": 42,
    "num_steps": 4,
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
    "seed": 12345,
    "num_steps": 8
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
- Input images : URL https OU base64 (max 20MB)
- Output images : `/runpod-volume/output/{job_id}/`
- Dedup cache par hash : `/runpod-volume/cache/dedup/{hash}/`

## Points d'Attention

- LightX2V gere nativement le FP8 via `enable_quantize()` — pas de chargement manuel (meta device / load_state_dict / filtrage weight_scale)
- Le fichier FP8 contient deja Lightning fuse — pas de `load_lora_weights` / `fuse_lora`
- Le repo Qwen est telecharge SANS les poids transformer BF16 (`ignore_patterns=['transformer/*.safetensors']`) — economie ~20GB disque
- H100 80GB VRAM — pas de CPU offload, pas de `enable_offload()`, tout reste en GPU
- `guidance_scale=1.0` desactive la guidance (modele distille, single forward pass)
- `num_steps` configurable par requete : 4 (rapide, defaut) ou 8 (meilleure qualite)
- Swap 4-step ↔ 8-step = reload complet du pipeline (~30-60s)
- Resolution max 1024px, default 768px, alignement 8px
- Pas d'upscale des images sources plus petites que 768px
- `transformers>=4.46` requis
- Resize images source avec LANCZOS (meilleure qualite downscale pour photos immobilieres)
- Limite upload images : 20MB max (check Content-Length HEAD + taille reelle)
- SageAttention 2 via `attn_mode="sage_attn2"` (Hopper-optimise) — fallback `flash_attn3` si NaN detecte (image noire/blanche/uniforme)
- SA2 + FP8 NaN fix merge dec 2025 (Issue #288) — Issue #199 (SA2 desactive) ne s'applique qu'avec CPU offload actif
- `create_generator()` appele dans `generate()` (pas dans `load()`) pour passer `height`/`width` dynamiquement a chaque requete — sinon LightX2V utilise ses defauts (480x832) et le runner interne recalcule a ~1664x928, ce qui degrade la qualite
- `create_generator()` sans `resize_mode` — evite l'override par la lookup table adaptive de Qwen
- Les configs LightX2V sont copiees depuis `/opt/lightx2v/configs/` dans le Dockerfile
- `fp8-vllm` utilise les kernels CUTLASS (Hopper TensorCore) — vllm installe avec `--no-deps` pour eviter les conflits torch
- Un `quant_scheme` invalide tombe silencieusement sur le backend Default (BF16) sans erreur — toujours verifier dans les logs que le bon backend est charge
- torch.compile regional sur blocs DiT — `max-autotune-no-cudagraphs` (resolution variable par requete), `fullgraph=False`, encapsule dans try/except
- Warmup torch.compile au cold start (~30-60s supplementaire) avec inference dummy 768x768
- PyTorch 2.10.x (CUDA 12.8) — installe avec `--no-deps` pour eviter le downgrade
- `TORCH_CUDA_ARCH_LIST="9.0"` pour cibler Hopper (sm_90)
