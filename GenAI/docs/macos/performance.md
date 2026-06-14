# macOS performance (Apple Silicon)

## MPS behavior
- First generation compiles shaders — expect slower first run
- Some operators fall back to CPU without warning
- Monitor Memory pressure in Activity Monitor

## Starting settings
| Model | Resolution | Batch |
|-------|------------|-------|
| SD1.5 | 512 | 1 |
| SDXL | 1024 | 1 |

## FaceFusion video
Process short test clips first. Temp frames use significant disk and memory.

See [stable-diffusion/13-performance-and-memory.md](../stable-diffusion/13-performance-and-memory.md).
