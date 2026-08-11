# State of Things and Next Work

Last updated: 2026-08-11

Target machine: 16 GB Apple M1 Mac mini

Branch: `dev`
Last committed head before this sprint: `771259243a5a6e9a938dcedab80999512b78f5fb`

## Current result

This fork remains Automatic1111 with targeted MPS and Metal acceleration rather than a separate inference engine. The working tree now adds two measured changes:

1. An opt-in, coarse MPS stage profiler enabled with `A1111_MPS_PROFILE=1`.
2. FP16 VAE as the tracked launch default only on M1-family Macs.

The normal path adds no profiler synchronization. Intel and non-M1 Apple Silicon retain `--no-half-vae` until separately validated. Automatic1111's existing NaN recovery remains enabled and retries VAE decode in FP32 if necessary.

## Reference workload

- Prompt: `a dog`
- Negative prompt: empty
- Checkpoint hash: `8ecad70a19`
- Steps: 5
- Sampler: DPM++ SDE
- Schedule: Karras
- CFG: 1.15
- Seed: `3163229250`
- Size: 384×640
- Clip skip: 2
- NGMS: 1.0, all steps
- Batch: 1

Five warm profiled runs produced these medians:

| VAE precision | Client wall time | Sampler | VAE decode + transfer |
| --- | ---: | ---: | ---: |
| FP32 | 8.450 s | 6.715 s | 1.536 s |
| FP16 | 7.795 s | 6.666 s | 0.972 s |

FP16 VAE saved about 0.65 seconds end to end and reduced VAE time by about 37%. The nine UNet calls were unchanged: five calls with input `2×4×80×48` and four with `1×4×80×48`, for 14 total batch elements. NGMS is responsible for the four batch-one calls.

## Output validation

FP32 and FP16 VAE output was compared for three fixed-seed generations at 384×640 and 512×512:

| Case | Mean absolute RGB delta | PSNR | Largest 8-bit channel delta | Byte-identical channels |
| --- | ---: | ---: | ---: | ---: |
| `a dog` | 0.0257 | 64.02 dB | 1 | 97.43% |
| `a dog and a cat` | 0.0229 | 64.54 dB | 1 | 97.71% |
| `1man, batman, looking out across the city` | 0.0227 | 64.58 dB | 1 | 97.73% |

All repeated FP16 runs were deterministic. No NaN, green, black, or corrupted images were observed. A normal `./webui.sh` launch reproduced the validated FP16 output hash.

## Working-tree changes

- `modules/mps_stage_profile.py`: request/stage timing, memory snapshots, UNet call accounting, JSON report.
- `modules/processing.py`: coarse generation-stage boundaries.
- `modules/sd_samplers_cfg_denoiser.py`: profiler-only UNet call/shape accounting.
- `webui-macos-env.sh`: M1-family FP16 VAE default; conservative fallback elsewhere.
- `test/test_mps_stage_profile.py`: profiler behavior and disabled-path checks.
- `test/test_macos_launch_defaults.py`: M1, M1 Max, M3, and Intel launch behavior.
- `README.md`: launch, quality, performance, profiling, and troubleshooting documentation.

The focused suite currently passes 21 tests. Python compilation, shell syntax, `git diff --check`, native Metal self-tests, an API generation, and a normal WebUI launch also pass.

## What the profile says next

With FP16 VAE enabled, the sampler/UNet is now roughly 87% of measured generation time. Image conversion and orchestration are negligible. Another material gain cannot come from unified-memory copies, PNG conversion, conditioning, or more VAE tuning; it must reduce UNet work or execute the UNet more efficiently.

Do not revisit these rejected directions without new evidence:

- DPM++ 2M substitution: it changes the desired LCM result.
- FP8 on M1: there is no matching M1 hardware acceleration path.
- Per-operator MPS timing events on PyTorch 2.3: isolated event synchronization hung on the tested system.
- The previous block-level MPSGraph prototype: it was about 1% slower end to end and had a larger numerical delta.

## Recommended next sprint

The next useful experiment is an opt-in static UNet executor, not another broad rewrite. Keep the normal Automatic1111 model and sampler interfaces, and cache a compiled path by checkpoint, latent shape, conditional batch shape, and active network state. Start with the exact SD 1.x reference workload and refuse unsupported inputs rather than silently changing behavior.

Suggested gates:

1. Support only txt2img, SD 1.x, batch 1, 384×640 and 512×512, no ControlNet, and no live model mutation in the prototype.
2. Preserve all nine DPM++ SDE evaluations and both batch-one and batch-two call shapes.
3. Compare five warm alternating runs against the current PyTorch MPS path.
4. Require at least a 5% end-to-end improvement before expanding compatibility.
5. Require deterministic output and report image deviation; fall back to PyTorch for every unsupported or failed graph.
6. Add LoRA-aware cache invalidation before considering a default-on route.

This is significant engine work. If the static executor cannot clear the 5% end-to-end gate on this M1, the current approximately 7.8-second profiled result is the practical stopping point for compatibility-preserving changes on the pinned PyTorch 2.3 runtime.
