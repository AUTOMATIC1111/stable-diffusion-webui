#!/usr/bin/env python3
"""Generate remaining GenAI documentation files."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SD_TUTORIALS = {
    "01-fundamentals.md": """# Stable Diffusion fundamentals

## What diffusion does
Stable Diffusion starts from random noise in **latent space** (a compressed representation of an image) and iteratively denoises it guided by your text prompt until a recognizable image emerges.

## Core terms

| Term | Meaning |
|------|---------|
| **Checkpoint** | Full trained model weights (UNet + text encoder + VAE bundled or partial) |
| **Latent space** | Lower-dimensional grid the UNet denoises (not pixel space) |
| **Text encoder** | Converts prompt tokens to conditioning vectors (CLIP for SD1.5) |
| **VAE** | Encodes/decodes between pixels and latents |
| **Positive prompt** | What you want |
| **Negative prompt** | What to suppress |
| **Seed** | RNG seed — same seed + settings → reproducible image |
| **Sampler** | Algorithm for each denoise step (euler, dpmpp_2m, etc.) |
| **Scheduler** | Noise schedule paired with sampler |
| **Steps** | Number of denoise iterations (more ≠ always better) |
| **CFG / guidance** | How strongly prompt constrains image (typical 5–8) |
| **Width / height** | Output resolution (affects VRAM quadratically) |
| **Denoising strength** | img2img: how much to change input (0=none, 1=full) |
| **LoRA** | Small add-on weights for style/subject |
| **ControlNet** | Structural guidance from edges, depth, pose, etc. |

## VRAM vs unified memory
- **NVIDIA VRAM**: dedicated GPU memory; OOM stops generation
- **Apple unified memory**: CPU and GPU share pool; pressure causes swapping

Continue to [02-first-image.md](02-first-image.md).
""",
    "02-first-image.md": """# Your first image

## Model requirement
Download **SD 1.5** `v1-5-pruned-emaonly.safetensors` (accept Hugging Face license manually) to:
`models/comfyui/checkpoints/`

## Steps
1. `.\scripts\windows\launch-comfyui.ps1` (or macOS equivalent)
2. Open http://127.0.0.1:8188
3. **Load** → `workflows/comfyui/beginner-text-to-image.json`
4. Confirm checkpoint name in **CheckpointLoaderSimple** node
5. Click **Queue Prompt**

## Exact settings in starter workflow
| Setting | Value |
|---------|-------|
| Positive | cozy reading nook, warm afternoon sunlight... |
| Negative | blurry, low quality, watermark... |
| Seed | 42 (fixed) |
| Resolution | 512 × 512 |
| Sampler | euler |
| Steps | 25 |
| CFG | 7.0 |

## Output
Images save with prefix `genai_beginner` in ComfyUI output folder.

## Reproduce
Keep seed **42** and same checkpoint — output should match.

## Variations
Change seed to `randomize` or increment seed for controlled diversity.

Next: [03-prompt-engineering.md](03-prompt-engineering.md)
""",
    "03-prompt-engineering.md": """# Prompt engineering

Structure prompts: **subject → action → composition → camera → lighting → environment → style**.

## Example 1 — portrait
**Prompt:** `portrait of a woman, three-quarter view, soft window light, neutral background, 85mm lens, shallow depth of field, natural skin texture, photorealistic`

**Settings:** 512×512, steps 25, CFG 7, euler

**Why it works:** Specifies framing, lens, and light before style token.

**Failure mode:** `beautiful woman` alone → generic, oversaturated faces.

**Refined:** Add `subtle catchlight in eyes, muted color palette` and negative `plastic skin, oversharpened`.

## Example 2 — product
**Prompt:** `matte black headphones on white marble surface, top-down flat lay, soft studio softbox, minimal shadows, commercial product photography`

**Negative:** `text, logo, watermark, cluttered background`

## Token weighting
ComfyUI CLIP encode accepts `(keyword:1.2)` in prompt text for emphasis.

## Iteration loop
1. Generate at 512 fast
2. Fix composition in prompt
3. Increase steps only if detail lacking
4. Raise CFG slightly for prompt adherence — watch for burned colors above 10
""",
    "04-models-checkpoints-and-vaes.md": """# Models, checkpoints, and VAEs

## Families
- **SD 1.5**: 512 native, vast LoRA ecosystem
- **SDXL**: 1024 native, two-encoder pipeline
- **Flux / SD3**: newer architectures — verify ComfyUI node support

## Compatibility
Do not load SDXL checkpoint in SD1.5 workflow nodes.

## VAE
Some checkpoints include baked VAE; external VAE fixes color/face issues. Place in `models/comfyui/vae/`.

## Comparison method
Lock seed, steps, sampler; change only checkpoint for fair test.

See [../models/README.md](../models/README.md).
""",
    "05-loras-and-embeddings.md": """# LoRAs and embeddings

## LoRA
Small weight files modifying UNet and/or text encoder. Typical strength **0.6–1.0** — start at 0.8.

Place in `models/comfyui/loras/`. Load with **LoraLoader** node in ComfyUI.

## Embeddings
Textual inversion vectors in `models/comfyui/embeddings/`. Trigger via token in prompt.

## Compatibility
LoRA trained for SD1.5 only works with SD1.5 base.
""",
    "06-samplers-schedulers-steps-and-cfg.md": """# Samplers, schedulers, steps, and CFG

## Steps
- 15–25 often sufficient for SD1.5 euler/dpm variants
- Diminishing returns after ~30 for many samplers

## CFG
- Low (3–5): creative, loose
- Mid (6–8): balanced default
- High (10+): harsh contrast, artifact risk

## Fair A/B
Change one knob at a time; lock seed.

## Oversaturation
Reduce CFG or add negative `oversaturated, hdr`.
""",
    "07-image-to-image.md": """# Image-to-image

Feed an init image + denoising strength.

- **0.3–0.5**: color/lighting tweaks, preserve structure
- **0.6–0.8**: stronger restyle
- **1.0**: near txt2img (weak structure retention)

Prepare square or crop consistently. See workflow `workflows/comfyui/image-to-image.json`.
""",
    "08-inpainting-and-outpainting.md": """# Inpainting and outpainting

## Inpainting
Mask the region to replace. Feather mask edges 4–8px to avoid seams.

Use cases: object removal, face/hand fix, background cleanup.

## Outpainting
Extend canvas; mask new areas; moderate denoise (0.7–0.85).

Workflow: `workflows/comfyui/inpainting.json`.
""",
    "09-controlnet-and-image-guidance.md": """# ControlNet and image guidance

ControlNet adds edge/depth/pose maps as conditioning.

Starter approach: **Canny edge** for composition lock. Place models in `models/comfyui/controlnet/`.

Do not install every custom node — use stock ComfyUI ControlNet nodes when available.

Workflow: `workflows/comfyui/controlled-generation.json`.
""",
    "10-upscaling-and-restoration.md": """# Upscaling and restoration

## Latent upscale
Upscale in latent space before final VAE decode — efficient.

## Pixel upscale
ESRGAN/UltraSharp in `models/comfyui/upscale_models/`.

## When not to upscale
Heavily artifacted base — upscaler amplifies flaws.

Workflow: `workflows/comfyui/upscale.json`.
""",
    "11-comfyui-fundamentals.md": """# ComfyUI fundamentals

## Nodes and wires
- **Nodes** = operations
- **Sockets** = typed ports (MODEL, CLIP, LATENT, IMAGE, VAE)
- Data flows left → right

## Essential chain
CheckpointLoader → CLIP encode (+/-) → KSampler → VAEDecode → SaveImage

## Queue
Each **Queue Prompt** runs the graph. Cached nodes skip re-execution when inputs unchanged.

## Workflow JSON
Save/load from UI. Committed starters in `workflows/comfyui/`.

## Missing nodes
Red nodes = unknown type — do not auto-install; review custom node source first.
""",
    "12-building-comfyui-workflows.md": """# Building ComfyUI workflows

1. Start from `beginner-text-to-image.json`
2. Add LoraLoader after checkpoint if needed
3. Insert ControlNetApply for guided generation
4. Use **Preview Image** before Save for fast iteration
5. Save incremental versions with descriptive names

Export JSON to `workflows/comfyui/` for Git tracking (no embedded private images).
""",
    "13-performance-and-memory.md": """# Performance and memory

## Windows RTX 4090
- SD1.5 512 batch 1: fast
- SDXL 1024: monitor VRAM in doctor/nvidia-smi
- Model load delay on first run is normal

## Apple Silicon
- MPS warmup on first generation
- Reduce resolution if swap increases

## Interrupt
Use ComfyUI cancel — if frozen, check logs; hard kill loses in-flight job only.

## Intel / CPU
Expect 10–50× slower; use low resolution for prompt iteration.
""",
    "14-model-management.md": """# Model management in practice

Track downloads in `config/model-manifest.json`.

## Hashes
Verify SHA256 from model card when provided.

## Licensing
Commercial use depends on checkpoint license — read model card.

## Malicious models
Prefer official repos; avoid random mirror sites.

Full guide: [../../models/README.md](../../models/README.md).
""",
    "15-troubleshooting.md": """# Stable Diffusion troubleshooting

| Symptom | Cause | Action |
|---------|-------|--------|
| Black image | VAE mismatch | Try external VAE or different checkpoint |
| CUDA OOM | Resolution too high | Lower size or batch |
| Red nodes | Missing node type | Use stock workflow or install reviewed custom node |
| Slow every gen | CPU mode | Re-run setup; verify CUDA/MPS |
| Wrong model | Filename mismatch | Fix CheckpointLoader widget |

**Diagnostic:** `doctor.ps1` / `doctor.sh`  
**Logs:** `logs/comfyui/`, ComfyUI terminal output
""",
}

FF_TUTORIALS = {
    "01-overview.md": """# FaceFusion overview

FaceFusion detects faces in source and target media, swaps identity, optionally enhances faces/frames, and re-encodes video with audio preservation.

## vs Stable Diffusion
SD generates images from noise; FaceFusion manipulates existing pixels/video frames.

## Key concepts
- **Source**: face identity donor (authorized only)
- **Target**: image or video to modify
- **Face swapper model**: ONNX identity transfer
- **Execution provider**: cuda / directml / coreml / cpu
- **Jobs**: batch/headless configurations

Launch: `launch-facefusion.ps1` or `.sh`

No Roop in this project.
""",
    "02-first-image-face-swap.md": """# First image face swap

## Authorized media only
Use your own face or documented consent + license.

## Requirements
- Clear frontal or three-quarter source face, well lit
- Target with visible face, similar angle when possible

## UI steps
1. Launch FaceFusion
2. Select source image (inputs/facefusion/source/)
3. Select target image
4. Choose face swapper model (defaults from FaceFusion)
5. Execution provider: cuda (Windows NVIDIA) or coreml/cpu (Mac)
6. Output to outputs/facefusion/
7. Run / preview

## Failed detection
Try higher-resolution source, better lighting, or manual face selector in UI.

Next: [03-video-face-swap.md](03-video-face-swap.md)
""",
    "03-video-face-swap.md": """# Video face swap

## Before long videos
Process **5–10 second clip** first.

## Pipeline
1. FFmpeg extracts frames
2. FaceFusion processes each frame
3. Re-encode video; audio copied when configured

## Disk space
Temp frames in `runtime/facefusion-temp/` — ensure 2× video size free.

## Flicker
Caused by detection inconsistency — use consistent source angle; enable enhancement cautiously.

## Audio missing
Verify FFmpeg on PATH; check FaceFusion output encoder settings.
""",
    "04-multiple-faces.md": """# Multiple faces

FaceFusion indexes faces left-to-right or by detection order.

Map source faces to target indices explicitly in UI before processing group scenes.

Preview each mapping on a single frame before full video.
""",
    "05-face-selection-and-indexing.md": """# Face selection and indexing

Lower detection confidence threshold only if faces are missed — increases false positives.

For occluded faces, partial swaps may fail; mask/occlusion handling is limited — set realistic expectations.
""",
    "06-enhancement-and-output-quality.md": """# Enhancement and output quality

Enhancement can fix blur but causes **plastic skin** if pushed too high.

Match lighting between source and target when possible.

Check edge blending around hair and jaw — common artifact regions.
""",
    "07-batch-and-headless-operation.md": """# Batch and headless operation

Verify flags on pinned 3.6.1:
```bash
python facefusion.py --help
python facefusion.py run --help
python facefusion.py headless-run --help
```

Typical patterns:
```bash
python facefusion.py force-download
python facefusion.py headless-run -s SOURCE -t TARGET -o OUTPUT
```

Use placeholder paths from `workflows/facefusion/example-image-job.ini`.

Job system commands (if available in 3.6.1): check `job-list`, `job-create` via --help output.
""",
    "08-performance.md": """# FaceFusion performance

| Platform | Provider | Relative speed |
|----------|----------|----------------|
| RTX 4090 | cuda | Fast |
| Windows | directml | Medium |
| Apple Silicon | coreml | Medium-variable |
| Any | cpu | Slow |

Video scales linearly with frame count × resolution.
""",
    "09-troubleshooting.md": """# FaceFusion troubleshooting

| Symptom | Cause | Diagnostic | Fix |
|---------|-------|------------|-----|
| No face detected | Poor source | doctor + preview | Better source photo |
| CUDA provider missing | Wrong onnxruntime | doctor ONNX line | Re-run setup-facefusion |
| FFmpeg error | Not installed | `ffmpeg -version` | Install FFmpeg |
| Port occupied | Other Gradio app | doctor port check | Change port / stop app |
| Temp disk full | Long video | Check facefusion-temp | Clear temp, shorten clip |

Logs: `logs/facefusion/setup.log`, terminal output.
""",
    "10-consent-provenance-and-safety.md": """# Consent, provenance, and safety

## Prohibited uses
- Non-consensual intimate imagery
- Sexualized depictions of minors
- Fraudulent impersonation or identity theft
- Fabricated legal/financial/political evidence
- Evading consent or forensic detection
- Presenting manipulated media as authentic evidence

## Permitted sources
- Your own likeness
- Consenting adults with documented permission
- Licensed stock/media
- Synthetic/public-domain test fixtures

## Provenance workflow
1. Keep original unedited media
2. Save FaceFusion settings / job config
3. Record app version from upstreams.lock.json
4. Date outputs; label manipulated media where sharing
5. Retain authorization records for commercial work

Private media stays in git-ignored `inputs/` and `outputs/` only.
""",
}

def write_if_missing(path: Path, content: str) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content.strip() + "\n", encoding="utf-8")
        print(f"created {path.relative_to(ROOT)}")

for name, body in SD_TUTORIALS.items():
    write_if_missing(ROOT / "docs/stable-diffusion" / name, body)

for name, body in FF_TUTORIALS.items():
    write_if_missing(ROOT / "docs/facefusion" / name, body)

print("done")
