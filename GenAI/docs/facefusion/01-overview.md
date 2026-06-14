# FaceFusion overview

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
