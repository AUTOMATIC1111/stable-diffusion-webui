# Inpainting and outpainting

## Inpainting
Mask the region to replace. Feather mask edges 4–8px to avoid seams.

Use cases: object removal, face/hand fix, background cleanup.

## Outpainting
Extend canvas; mask new areas; moderate denoise (0.7–0.85).

Workflow: `workflows/comfyui/inpainting.json`.
