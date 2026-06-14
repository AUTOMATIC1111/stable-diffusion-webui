# Your first image

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
