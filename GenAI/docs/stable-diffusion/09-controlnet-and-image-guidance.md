# ControlNet and image guidance

ControlNet adds edge/depth/pose maps as conditioning.

Starter approach: **Canny edge** for composition lock. Place models in `models/comfyui/controlnet/`.

Do not install every custom node — use stock ComfyUI ControlNet nodes when available.

Workflow: `workflows/comfyui/controlled-generation.json`.
