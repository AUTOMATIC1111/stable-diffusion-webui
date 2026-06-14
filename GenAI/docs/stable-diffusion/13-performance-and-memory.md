# Performance and memory

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
