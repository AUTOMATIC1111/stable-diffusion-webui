# LoRAs and embeddings

## LoRA
Small weight files modifying UNet and/or text encoder. Typical strength **0.6–1.0** — start at 0.8.

Place in `models/comfyui/loras/`. Load with **LoraLoader** node in ComfyUI.

## Embeddings
Textual inversion vectors in `models/comfyui/embeddings/`. Trigger via token in prompt.

## Compatibility
LoRA trained for SD1.5 only works with SD1.5 base.
