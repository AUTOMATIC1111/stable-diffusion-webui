# State of Things and Next Work

Last updated: 2026-08-11

Target machine: 16 GB Apple M1 Mac mini

Branch: `dev`
Last committed head before this sprint: `771259243a5a6e9a938dcedab80999512b78f5fb`

## Current result

This fork remains Automatic1111 with targeted MPS and Metal acceleration rather than a separate inference engine. The current optimization stack includes three measured changes from the latest sprints:

1. An opt-in, coarse MPS stage profiler enabled with `A1111_MPS_PROFILE=1`.
2. FP16 VAE as the tracked launch default only on M1-family Macs.
3. An exact-parity fused Metal GEGLU path enabled by default on compatible Mac inference.

The GEGLU kernel uses a 128 KB table generated once by PyTorch MPS to preserve every possible FP16 GELU result, then combines table lookup and multiplication in one Metal dispatch. Exact SD 1.x batch-one and batch-two tests were byte-identical to PyTorch. A fixed-process 512×512 DPM++ SDE A/B produced identical PNG hashes and saved approximately 0.12–0.27 seconds in every matched pair. An active LCM LoRA output was also byte-identical with the fusion on and off.

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
- A later real-weight MPSGraph ResBlock/down-stage executable: after comparing against the fork's fused GroupNorm baseline, it improved the measured stage by only about 1.4% and failed the integration gate.
- TorchScript fixed-shape UNet tracing: warm results were inconsistent and lost their benefit after cache loss while retaining enough state to increase unified-memory pressure.

## Recommended next sprint

Do not immediately revisit static UNet tracing or block-level MPSGraph; both have now failed measured gates on this M1. The next compatibility-preserving experiments should remain narrow transformer micro-fusions, with fused LayerNorm as the leading candidate. Packed self-attention QKV or cross-attention KV projection is a larger follow-up only if LoRA and model-mutation invalidation can be made exact.

Suggested gates:

1. Preserve all nine DPM++ SDE evaluations and both batch-one and batch-two call shapes.
2. Compare alternating warm runs against the current PyTorch MPS path.
3. Require exact tensor parity for lookup-based or algebraically identical fusions; otherwise report image deviation explicitly.
4. Require a positive end-to-end result, not only an isolated kernel win.
5. Preserve LoRA, ControlNet, dynamic resolution, model switching, and training fallbacks.

A separate whole-UNet Metal, MLX, or Core ML backend remains the only plausible route to a large additional gain. That is significant engine work and should be treated as a new backend rather than another Automatic1111 micro-optimization.
