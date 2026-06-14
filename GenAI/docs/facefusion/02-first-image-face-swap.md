# First image face swap

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
