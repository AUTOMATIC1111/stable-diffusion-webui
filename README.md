# Stable Diffusion WebUI Metal

An Apple Silicon performance fork of [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui), focused on faster and more memory-aware inference through PyTorch MPS and native Metal kernels.

The normal Automatic1111 interface, API, checkpoint layout, samplers, LoRA syntax, and extension structure are preserved. The fork adds a selective Metal attention path, fused GroupNorm + SiLU and exact-parity GEGLU kernels, unified-memory-aware attention fallback, an M1-validated FP16 VAE path, and tested macOS dependency defaults. Stable Diffusion 1.x inference—particularly short DPM++ SDE runs—is the primary optimization target.

> [!IMPORTANT]
> This is an experimental performance fork, not a new Stable Diffusion engine. It favors measured M1 inference performance and safe fallback behavior over broad hardware tuning. If a native Metal path is unavailable or fails its startup test, the WebUI falls back to the corresponding PyTorch implementation.

## Current project snapshot

The current tested head is [`6eefbb40`](https://github.com/dmikey/stable-diffusion-webui-metal/commit/6eefbb402d177ec5166dbb364ea8e313d1bdb206) on `dev`. It remains recognizably Automatic1111: the performance work is concentrated in MPS routing, a small native Metal extension, macOS launch defaults, profiling, benchmarks, and tests.

| Measure | Value |
| --- | ---: |
| Automatic1111 base | [`1937682a`](https://github.com/AUTOMATIC1111/stable-diffusion-webui/commit/1937682a20f7f0442311a1ede68f9f0cb480163b) |
| Base version | `v1.10.1-96-g1937682a` |
| Current Metal head | [`6eefbb40`](https://github.com/dmikey/stable-diffusion-webui-metal/commit/6eefbb402d177ec5166dbb364ea8e313d1bdb206) |
| Current version | `v1.10.1-104-g6eefbb40` |
| Code relationship | 8 fork commits ahead of the selected Automatic1111 base |
| Changed tracked paths | 29, including this README and the roadmap |
| Changed implementation/test paths | 27 |
| Total delta | 2,585 insertions, 271 deletions |

The eight fork commits are:

1. Apple Silicon dependencies, attention routing, unified-memory budgeting, and benchmark foundation.
2. Removal of obsolete MPS safety operations on modern PyTorch.
3. Metal Flash Attention command-buffer coalescing.
4. Native fused GroupNorm + SiLU for compatible inference blocks.
5. Documentation and launch configuration cleanup.
6. Coarse MPS stage profiling and the M1 FP16 VAE default.
7. Exact-parity native fused GEGLU.
8. Clip skip 2 as the built-in default.

Most added code is isolated Metal code, profiling, benchmark utilities, and tests. The fork does **not** require a new checkpoint format, prompt syntax, REST API contract, or UI workflow.

<details>
<summary>Current implementation surface</summary>

| Area | Added | Modified |
| --- | --- | --- |
| Metal runtime | `modules/mps_flash_attention.py`<br>`modules/mps_fused_ops.py`<br>`modules/mps_utils.py` | `modules/mac_specific.py`<br>`modules/sd_hijack_optimizations.py`<br>`modules/sd_hijack_unet.py`<br>`modules/sub_quadratic_attention.py` |
| Profiling | `modules/mps_stage_profile.py` | `modules/processing.py`<br>`modules/sd_samplers_cfg_denoiser.py` |
| Startup and defaults | `requirements_macos.txt` | `modules/launch_utils.py`<br>`modules/shared_options.py`<br>`requirements_versions.txt`<br>`webui-macos-env.sh`<br>`webui-user.sh` |
| Native build and benchmarks | `scripts/install_mps_flash_attention.py`<br>`scripts/mps_fused_group_norm.mm`<br>`scripts/benchmark_mps_attention.py`<br>`scripts/benchmark_mps_unet_ops.py`<br>`scripts/benchmark_mps_geglu_probe.py` | — |
| Tests | `test/test_macos_launch_defaults.py`<br>`test/test_mps_flash_attention.py`<br>`test/test_mps_fused_ops.py`<br>`test/test_mps_stage_profile.py`<br>`test/test_mps_utils.py`<br>`test/test_sub_quadratic_attention.py` | — |

</details>

You can reproduce the comparison locally:

```bash
git rev-list --left-right --count 1937682a...6eefbb40
git diff --shortstat 1937682a..6eefbb40
git diff --name-status 1937682a..6eefbb40
```

## What is different?

### Selective Metal Flash Attention

On supported Apple Silicon inference shapes, Automatic mode prefers a native Metal Flash Attention implementation derived from the `mps-flash-sdpa` package.

- Routes measured SD 1.x head dimensions (`40`, `80`, and `160`) with at least 192 query tokens to the native kernel.
- Supports both self-attention and cross-attention on the measured path.
- Encodes work on PyTorch's current MPS command buffer instead of forcing a submission after every attention call.
- Uses PyTorch scaled dot product attention for unsupported shapes, training, masks, grouped-query attention, non-FP16 tensors, or runtime failure.
- Builds from `mps-flash-sdpa==0.1.0` source on first launch, checks the downloaded artifact against PyPI's published SHA-256 metadata, and installs only into the local virtual environment.
- Runs an isolated GPU self-test before enabling the native route, so a native crash cannot take down the main WebUI process during capability detection.

The attention choices appear under **Settings → Optimizations → Cross attention optimization**:

- `Automatic`: use Metal Flash Attention when its startup test succeeds.
- `mps-flash`: explicitly select the native Metal route with PyTorch fallback.
- `mps-adaptive`: use native PyTorch attention while it fits a unified-memory budget, then fall back to sub-quadratic attention.
- `sub-quadratic`, `sdp`, and the other upstream implementations remain available for comparison and compatibility.

### Fused GroupNorm + SiLU

A native inference-only Metal kernel combines GroupNorm and SiLU in one dispatch for compatible contiguous FP16 tensors.

- Used in compatible SD/SGM UNet residual blocks.
- Used in compatible VAE residual blocks when the VAE is running in FP16.
- Preserves the normal PyTorch path for CPU, non-FP16 tensors, training/autograd, incompatible layouts, missing affine parameters, or runtime errors.
- Enabled by default through **Settings → Optimizations → Fuse GroupNorm and SiLU on Apple Silicon**.

This is a focused fusion; convolutions and residual additions still use PyTorch MPS. A larger block-level MPSGraph prototype was tested and deliberately rejected because it was about 1% slower end to end and produced a larger numerical delta without a speed benefit.

### Exact-parity fused GEGLU

The SD 1.x transformer feed-forward path normally stores a GELU result and then launches a separate multiply. On Apple Silicon, the fork combines the lookup and multiply into one Metal dispatch.

- The model's linear projection still runs normally, so active LoRAs and other projection hooks remain compatible.
- A one-time 65,536-entry FP16 table is generated with the installed PyTorch MPS GELU implementation. The table is 128 KB and maps every possible half-precision gate value to PyTorch's exact result.
- The fused output was byte-identical to PyTorch at all SD 1.x transformer shapes for batch one and batch two.
- CPU, FP32, training/autograd, incompatible layouts, disabled settings, and runtime failures use the original PyTorch implementation.
- Enabled by default through **Settings → Optimizations → Fuse GEGLU on Apple Silicon**.

### Unified-memory-aware attention

The fork treats system RAM and GPU memory as the same constrained resource instead of relying on a fixed attention threshold.

- Estimates scaled dot product attention's temporary memory from batch, heads, token counts, and element size.
- Limits native attention to a fraction of total and currently available unified memory.
- Dynamically reduces sub-quadratic query tiles for large self-attention workloads.
- Uses streaming online softmax when K/V attention is chunked, merging one tile at a time instead of stacking all partial outputs in memory.
- Keeps cross-attention on the fast path when its short key sequence remains inexpensive.

The adaptive path is especially useful for high resolutions and lower-memory Macs. The default Metal Flash Attention path remains the measured choice for normal SD 1.x shapes.

### Modern MPS runtime cleanup

Several workarounds needed by early PyTorch MPS releases are now gated by runtime version:

- Avoids cloning every `narrow()` result on PyTorch versions where the underlying MPS bug is fixed.
- Avoids unconditional FP32 LayerNorm conversion on modern runtimes.
- Keeps an environment switch for diagnosing regressions with legacy behavior.
- Prefers direct Metal matrix multiplication for the SD 1.x projection sizes measured on M1.
- Removes Automatic1111's default `--upcast-sampling` flag on Apple Silicon; it can be restored locally when exact upstream behavior is more important than speed.

### Coarse MPS stage profiler

An opt-in profiler measures the parts of a complete generation that are large enough to guide optimization decisions without adding synchronization to normal inference.

- Enable it with `A1111_MPS_PROFILE=1 ./webui.sh`.
- Reports conditioning, sampler, VAE decode/transfer, image processing, and request wall time.
- Records every UNet call shape and MPS allocation snapshots in a machine-readable `MPS_PROFILE_JSON` line.
- Adds no MPS synchronization points when disabled.

On the M1 reference workload, the profiler established that the sampler/UNet consumes roughly 87% of generation time after enabling the FP16 VAE. This is why current roadmap work targets whole-UNet execution rather than PNG conversion, conditioning, or more VAE micro-tuning.

### Apple Silicon dependency profile

The default Apple Silicon environment is pinned to the combination verified for this fork:

| Dependency | Version |
| --- | --- |
| Python | 3.10 recommended; 3.10.20 used during development |
| PyTorch | 2.3.1 |
| torchvision | 0.18.1 |
| SciPy | 1.13.1 |
| Native extension | `mps-flash-sdpa` 0.1.0 with local stream-safety and fusion patches |

SciPy is constrained on Apple Silicon because newer wheels encountered loader problems on the macOS beta used during development. Requirements parsing was also updated to understand platform markers and normal Python package specifiers correctly.

### Changed defaults

The following defaults intentionally differ from the upstream `dev` branch:

| Setting | Upstream | This fork | Effect |
| --- | --- | --- | --- |
| Negative Guidance minimum sigma (NGMS) | `0.0` | `1.0` | May skip unconditional guidance late in sampling |
| NGMS all steps | Off | On | Applies the configured NGMS rule on every eligible step |
| `--upcast-sampling` on macOS | On | Off | Keeps more sampling work in FP16 for speed |
| `--no-half-vae` on M1-family Macs | On | Off | Runs VAE encode/decode in FP16; Automatic1111 still retries in FP32 if VAE decode produces NaNs |
| Clip skip | `1` | `2` | Uses the common SD 1.x checkpoint default without depending on local `config.json` |
| Cross-attention Automatic choice on MPS | Sub-quadratic | Metal Flash Attention | Uses the measured native route when available |
| Fused GroupNorm + SiLU | Not present | On | Reduces compatible normalization/activation dispatches |
| Fused GEGLU | Not present | On | Preserves PyTorch FP16 output while reducing transformer activation dispatches |

NGMS is the largest user-visible behavioral change. It is recorded in PNG generation metadata when active. Set NGMS to `0` and disable **NGMS all steps** if a workflow expects upstream guidance behavior.

## Measured performance

One recorded Apple M1 Mac mini comparison during development used the same checkpoint hash and compute shape:

| Build | Workload | Time |
| --- | --- | ---: |
| Automatic1111 `v1.10.1-96-g1937682a` | 5 steps, DPM++ SDE, Karras, CFG 1.15, 384×640, SD 1.x checkpoint `8ecad70a19`, Clip skip 2, NGMS 1/all steps | 12.8 s |
| This fork `v1.10.1-99-g38ac556a` | Same sampler, schedule, dimensions, checkpoint hash, Clip skip, and NGMS settings | 8.7 s |

That observed run was approximately **32% lower latency**, or **1.47× as fast**. Later heads added fused GroupNorm + SiLU, the profiled FP16 VAE default, and exact-parity GEGLU after that recorded comparison.

The two recorded generations used different seeds. This makes the table a throughput comparison at matching tensor shapes, not an image-parity A/B.

### M1 fused GEGLU validation

A fixed-process API A/B used `hyperGlance` (`8ecad70a19`), prompt `a dog and a cat`, seed `158926638`, 5-step DPM++ SDE with Karras, CFG 1.15, Clip skip 2, NGMS 1/all steps, and 512×512 output. Three alternating warm pairs measured a positive saving in every pair: approximately 0.12–0.27 seconds, with a median paired saving of about 0.27 seconds. All fusion-on and fusion-off PNG files had the same SHA-256 hash.

An additional active-LoRA check used `a dog <lora:lcm:1>`, seed `784504668`, five Euler a steps, and the same CFG, size, Clip skip, and NGMS settings. Fusion-on and fusion-off output hashes were identical. The timing from that single LoRA pair is not reported as a speed result because its first run included LoRA activation overhead.

### M1 FP16 VAE validation

A later controlled A/B isolated VAE precision on a 16 GB Apple M1 Mac mini. Both paths used checkpoint `8ecad70a19`, prompt `a dog`, seed `3163229250`, 5-step DPM++ SDE with Karras, CFG 1.15, Clip skip 2, NGMS 1/all steps, and 384×640 output. Each result below is the median of five warm runs with coarse MPS stage profiling enabled.

| VAE path | End-to-end client time | Sampler stage | VAE decode + transfer |
| --- | ---: | ---: | ---: |
| FP32 (`--no-half-vae`) | 8.450 s | 6.715 s | 1.536 s |
| FP16 | 7.795 s | 6.666 s | 0.972 s |

FP16 reduced the measured VAE stage by about **37%** and end-to-end latency by about **7.8%**. The sampler time remained effectively unchanged, which is the expected result when only decode precision changes.

Output quality was checked across three fixed-seed generations at 384×640 and 512×512. Compared with FP32 VAE output, every changed 8-bit RGB channel differed by at most 1 value, PSNR was 64.0–64.6 dB, and 97.4–97.7% of channels were byte-identical. All FP16 runs were deterministic and free of NaN, green, or corrupted output. The default is therefore enabled only on the tested M1 family; other Apple Silicon generations retain FP32 VAE until separately validated.

### Current warm-run range

A user-facing run at `v1.10.1-102-g58e63e9f` used `fast-model` (`8ecad70a19`), prompt `a dog`, seed `4017012032`, five-step DPM++ SDE with Karras, CFG 1.15, Clip skip 2, NGMS 1/all steps, and 512×512 output. It completed in **8.3 seconds** on the 16 GB M1 Mac mini.

Later controlled development A/B runs of the same 512×512 shape typically clustered around 9.2–9.3 seconds after warm-up. Background load, extension startup activity, thermal state, and unified-memory pressure therefore matter at the sub-second scale. Report medians and the full generation settings rather than treating a single fastest run as a guarantee.

## Experiments that did not pass the gate

Failed experiments are documented to prevent attractive microbenchmarks from being repeated without new evidence.

| Experiment | Isolated result | End-to-end result | Decision |
| --- | --- | --- | --- |
| DPM++ 2M substitution | Fewer or cheaper operations in some paths | Did not reproduce the desired LCM/DPM++ SDE result | Rejected; preserve the requested sampler |
| Block-level MPSGraph | Working block prototype | About 1% slower with a larger numerical delta | Removed |
| Real-weight MPSGraph ResBlock/down stage | About 1.4% stage improvement versus the fused GroupNorm baseline | Too small to survive integration overhead | Removed |
| Fixed-shape TorchScript UNet | Some warm runs improved | Benefit was inconsistent and disappeared after cache loss while retaining extra unified memory | Removed |
| Native fused LayerNorm | Exact SD1 workload projection suggested about 110 ms potential savings | Baseline median 11.512 s versus 11.503 s enabled; paired median regressed by 0.026 s. Only 56.1% of RGB channels were identical, with PSNR 47.53 dB | Removed because there was no speed gain and output drifted |
| Cross-attention K/V reuse | Reused 112 of 144 projections | Baseline median 9.258 s versus 9.262 s cached; paired median regressed by 0.016 s while retaining 11 MiB. PNG hashes were identical | Removed because the projections were already too cheap |
| FP8 on M1 | Reduced theoretical weight storage | No matching M1 FP8 execution path; conversion/unpacking would dominate | Not implemented |

The LayerNorm and K/V probes were completely removed after testing. They are not hidden options and do not remain in the native extension. The repository returned to a clean state after each rejected sprint.

Treat these numbers as a development result, not a universal guarantee. Timing varies with:

- Apple Silicon generation and GPU core count
- Unified-memory capacity and pressure from other applications
- Model architecture and attention dimensions
- Resolution, batch size, sampler, and step count
- First-run shader compilation and warm-up
- VAE, LoRA, ControlNet, extensions, and live preview configuration

For a meaningful comparison, use the same model hash, prompt, negative prompt, seed, sampler, schedule, steps, CFG, dimensions, Clip skip, VAE, and optimization settings. Run at least two warm-ups, then compare the median of several generations.

## Installation

### Requirements

- An Apple Silicon Mac for the optimized native path
- macOS with Metal Performance Shaders support
- Python 3.10
- Git
- Xcode Command Line Tools, required to compile the Objective-C++/Metal extension

Install the command-line tools if needed:

```bash
xcode-select --install
```

### Fresh installation

```bash
git clone --branch dev https://github.com/dmikey/stable-diffusion-webui-metal.git
cd stable-diffusion-webui-metal
./webui.sh
```

The first launch creates the virtual environment, installs the pinned Apple Silicon dependencies, downloads and builds the native extension, runs its isolated self-test, and starts the normal Automatic1111 interface. Native extension compilation can make the first launch noticeably longer than later launches.

Put checkpoints in:

```text
models/Stable-diffusion/
```

LoRAs, VAEs, embeddings, extensions, and outputs use the usual Automatic1111 directories.

### Updating

The repository's default branch is `dev`:

```bash
git switch dev
git pull --ff-only origin dev
./webui.sh
```

Do not commit generated `config.json`, `ui-config.json`, `params.txt`, models, outputs, the virtual environment, or extension installations. They are local runtime state and are ignored by Git.

### Local launch options

`webui-macos-env.sh` contains the fork's tracked defaults. Put personal overrides in `webui-user.sh`, which is loaded afterward. For example, to restore sampling upcast while retaining the rest of the Metal work:

```bash
export COMMANDLINE_ARGS="--skip-torch-cuda-test --upcast-sampling --no-half-vae --use-cpu interrogate"
```

M1-family Macs use the validated FP16 VAE path by default. Intel and other Apple Silicon generations retain `--no-half-vae`. Add `--no-half-vae` to a local `COMMANDLINE_ARGS` override at any time to force the conservative FP32 VAE path. Automatic1111's enabled-by-default VAE precision recovery also converts the VAE to FP32 and retries if an FP16 decode produces NaNs.

## Startup messages and fallback behavior

A healthy optimized startup prints messages similar to:

```text
Metal self-test passed; deferred MFA, fused GroupNorm+SiLU, and fused GEGLU routing enabled.
Applying attention optimization: mps-flash... done.
```

The first compatible generation also reports the first native attention, GroupNorm, and GEGLU dispatch. These messages are informational and print only once per process.

If the extension cannot build or fails its isolated self-test, startup continues with native PyTorch MPS operations. If either fused activation kernel fails at runtime, that fusion is disabled for the process and PyTorch handles subsequent operations.

## Compatibility and output parity

### What should remain compatible

- Automatic1111's txt2img, img2img, inpainting, high-resolution pass, API, and metadata workflow
- Existing `.safetensors` and `.ckpt` checkpoints
- Standard LoRA, embedding, VAE, and extension directory layouts
- Existing sampler names and generation parameter syntax
- CPU and non-MPS fallback implementations

The optimized target is SD 1.x inference. Other architectures supported by this Automatic1111 base may run, but unsupported attention shapes fall back to PyTorch and may receive little or no speed benefit. Test model-specific extensions individually.

### Why the same seed may differ from upstream

Pixel-identical output is not guaranteed. Differences can come from:

- NGMS being enabled by default
- Sampling no longer being upcast by default
- Native Flash Attention and fused GroupNorm changing FP16 reduction/rounding order
- A different selected attention implementation

Small FP16 numerical differences can grow over multiple denoising evaluations even when both paths are deterministic.

### Closest upstream behavior

For an upstream-style comparison:

1. Set **Negative Guidance minimum sigma** to `0`.
2. Disable **Negative Guidance minimum sigma all steps**.
3. Disable **Fuse GroupNorm and SiLU on Apple Silicon**.
4. Disable **Fuse GEGLU on Apple Silicon**.
5. Select `sub-quadratic` under **Cross attention optimization**.
6. Add `--upcast-sampling` to `COMMANDLINE_ARGS` in `webui-user.sh`.
7. On M1, also add `--no-half-vae`.
8. Restart the WebUI after changing launch arguments.

For diagnostics only, `A1111_MPS_FORCE_LEGACY_OPS=1` restores version-gated MPS safety copies. `A1111_MPS_DISABLE_FUSED_GROUP_NORM_SILU=1` and `A1111_MPS_DISABLE_FUSED_GEGLU=1` disable the corresponding native fusion before startup.

## Troubleshooting

### Native extension does not build

Confirm that Xcode Command Line Tools and the local virtual environment are available:

```bash
xcode-select -p
./venv/bin/python scripts/install_mps_flash_attention.py
```

Then restart with `./webui.sh`. The installer intentionally rebuilds the package from source for the active Python and PyTorch environment.

### Metal self-test fails

The WebUI should continue on PyTorch MPS. Keep the final `Metal Flash Attention unavailable:` message when reporting the issue. Also include:

- Mac model and memory capacity
- macOS version
- `./venv/bin/python -c "import torch; print(torch.__version__)"`
- The selected cross-attention optimization
- Model family, resolution, and batch size

### Green, black, or corrupted output

Add `--no-half-vae` to the local launch options and restart first. Also compare with NGMS disabled, sampling upcast restored, `sub-quadratic` attention selected, and the fused GroupNorm option disabled. That separates model/VAE precision issues from the native Metal paths.

### High-resolution out-of-memory errors

Select `mps-adaptive - native Metal attention with a memory-safe fallback` or `sub-quadratic` in the optimization settings. Reduce batch size before reducing attention chunk limits manually.

## Benchmarks and tests

Two standalone benchmark scripts are included:

```bash
./venv/bin/python scripts/benchmark_mps_attention.py
./venv/bin/python scripts/benchmark_mps_unet_ops.py --batch 2
./venv/bin/python scripts/benchmark_mps_geglu_probe.py
```

The first compares PyTorch MPS scaled dot product attention with sliced attention. The second measures representative SD 1.x convolution, GroupNorm + SiLU, linear projection, and attention shapes.

For an end-to-end stage breakdown, launch with the opt-in profiler:

```bash
A1111_MPS_PROFILE=1 ./webui.sh
```

Each generation reports synchronized wall time for conditioning, sampling, VAE decode/transfer, and image processing; it also records UNet call shapes and MPS allocation snapshots in a machine-readable `MPS_PROFILE_JSON` line. Profiling is intentionally coarse because PyTorch 2.3 MPS timing events are unreliable on the tested runtime. When the environment variable is absent, the profiler adds no MPS synchronization points.

Focused tests cover:

- Metal Flash Attention routing and PyTorch fallback
- M1-specific FP16 VAE launch defaults with conservative Intel and newer-chip behavior
- Native fused GroupNorm + SiLU correctness
- Native fused GEGLU exact parity, fallback routing, and active-LoRA compatibility
- Opt-in MPS stage profiling and its zero-synchronization disabled path
- Unified-memory attention budgeting and dynamic query tiles
- Streaming online-softmax forward results and gradients

With `pytest` installed in the virtual environment:

```bash
./venv/bin/python -m pytest -q \
  test/test_macos_launch_defaults.py \
  test/test_mps_flash_attention.py \
  test/test_mps_fused_ops.py \
  test/test_mps_stage_profile.py \
  test/test_mps_utils.py \
  test/test_sub_quadratic_attention.py
```

## Future roadmap: native ggml/Metal UNet

The remaining material opportunity is engine-level work. The best incremental direction is inspired by [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) and ggml: execute the complete UNet as one planned Metal graph with a reusable memory arena instead of crossing the Python/PyTorch boundary for individual kernels.

This is not a plan to replace Automatic1111 wholesale. Prompt parsing, conditioning, the selected A1111/k-diffusion sampler, CFG and NGMS behavior, seed handling, extensions, VAE, image processing, metadata, API, and UI remain in the existing application. Only a compatible UNet evaluation may be delegated to the native backend.

```text
Automatic1111 prompt, LoRA, and conditioning setup
                         |
Existing DPM++ SDE / Karras sampler and NGMS logic
                         |
       Native ggml/Metal UNet evaluation
                         |
Existing CFG combination, VAE, image pipeline, API, and UI
```

stable-diffusion.cpp is relevant because its current implementation already provides a complete [SD 1.x UNet graph runner](https://github.com/leejet/stable-diffusion.cpp/blob/bcc7e29568b94a25f78e99d34a8fa048d77536b1/src/model/diffusion/unet.hpp#L748), a [reusable graph allocator](https://github.com/leejet/stable-diffusion.cpp/blob/bcc7e29568b94a25f78e99d34a8fa048d77536b1/src/core/ggml_extend.hpp#L2212), whole-graph [Metal command encoding](https://github.com/leejet/stable-diffusion.cpp/blob/bcc7e29568b94a25f78e99d34a8fa048d77536b1/ggml/src/ggml-metal/ggml-metal-context.m#L438), safetensors/GGUF loading, LoRA support, Flash Attention, and fused quantized matrix kernels. Its advantage comes from owning the graph, buffers, weights, and submission lifecycle together—not from one operator that can be dropped into PyTorch.

### Phase 0: reproducible captured-tensor corpus

Capture the inputs and reference outputs of every UNet evaluation from the existing M1 workload:

- Five calls with batch two and four calls with batch one under five-step DPM++ SDE plus NGMS.
- Both `512×512` and `384×640` latent shapes.
- Latent input, timestep, text conditioning, model hash, precision, and PyTorch output.
- A plain prompt, scheduled prompt, active LoRA, and a deliberately unsupported request for fallback testing.

The capture path must be diagnostic-only and must not alter normal timing or output when disabled.

### Phase 1: standalone native UNet shootout

Build a small C/C++ harness around stable-diffusion.cpp's `UNetModelRunner`. Load the same SD 1.x safetensors checkpoint and replay the captured calls outside WebUI.

Measure:

- Per-call and complete nine-call latency after warm-up.
- Batch-one and batch-two behavior separately.
- Peak and retained unified memory.
- Mean, maximum, and percentile tensor deviation from PyTorch MPS.
- Determinism across repeated runs.

Proceed only if the native nine-call workload is at least **20–25% faster** than the current PyTorch MPS UNet. A smaller isolated advantage is unlikely to survive framework-bridge synchronization and compatibility handling.

### Phase 2: copied-buffer A1111 prototype

Expose a minimal native interface that accepts latent, timestep, and conditioning buffers and returns the UNet prediction. Keep A1111's current sampler in control, initially accepting one synchronization and copy boundary per UNet evaluation.

The first supported route should be intentionally narrow:

- Apple M1 and SD 1.x only.
- FP16 inference.
- Txt2img, batch one, tested resolutions.
- No ControlNet, hypernetwork, training, or high-resolution pass.
- No active LoRA until mutation/invalidation is explicitly implemented.

Every unsupported request must automatically use the existing PyTorch UNet. The native backend should be an optional `SdUnetOption`, never a global monkey patch with no escape path.

### Phase 3: unified-memory zero-copy proof

If the copied prototype remains faster, investigate sharing the underlying Metal storage rather than copying through CPU memory. PyTorch MPS tensors and ggml Metal tensors ultimately reside in `MTLBuffer` objects, but safe sharing requires explicit work on:

- Buffer offsets, strides, dtype, and NCHW layout agreement.
- Ownership and lifetime across Python, PyTorch, and the native runner.
- Command-queue ordering and synchronization.
- Error recovery without leaving either backend in a poisoned state.

This phase should begin with one captured UNet call. Do not attempt full sampling until the shared-buffer output matches the copied native path.

### Phase 4: compatibility expansion

Add features one at a time, with a PyTorch fallback and an output test for each:

1. Dynamic SD 1.x resolutions and cached arenas per batch/shape regime.
2. Active LoRA application, model-mutation generation counters, and exact cache invalidation.
3. Img2img and inpainting conditioning.
4. High-resolution pass and model switching.
5. ControlNet where native semantics can match the installed A1111 extension.
6. Other model families only after SD 1.x is stable.

Extension compatibility is a routing problem: requests using unsupported hooks should remain fully functional on PyTorch rather than partially executing through native code.

### Phase 5: optional GGUF quantization

Quantization follows a successful FP16 engine; it is not the first step. ggml gains from quantized weights because dequantization is fused into its Metal matrix kernels. Merely storing quantized tensors in PyTorch would not reproduce that behavior.

Suggested order for the M1:

1. FP16 native backend establishes the execution-engine benefit and parity baseline.
2. Q8_0 evaluates memory reduction with the smallest expected quality risk.
3. Q6_K or Q5_K may become optional balanced modes.
4. Q4 remains an explicit low-memory choice, not the default.

Each format requires fixed-seed image comparisons, tensor statistics, LoRA checks, and end-to-end timing. A smaller model file alone is not a speed result.

### Roadmap acceptance gates

A native backend is eligible for default use only when it:

1. Improves multiple alternating warm end-to-end pairs, not just an operator microbenchmark.
2. Preserves all nine DPM++ SDE evaluations and the current sampler's result.
3. Reports deterministic output and quantified deviation from the PyTorch path.
4. Does not retain enough extra unified memory to erase warm-run stability.
5. Falls back cleanly for LoRA, ControlNet, dynamic shapes, training, and extensions it cannot reproduce.
6. Can be disabled without changing checkpoint files or local configuration.

The initial target is to determine whether native UNet execution can move the 16 GB M1 from the current roughly 8–9 second warm range toward 7–8 seconds. Phase 1 is deliberately a bounded proof: if the raw native UNet cannot clear its 20–25% gate, the integration project stops before modifying WebUI.

### What not to borrow incrementally

- Individual ggml convolutions or matrix kernels called from PyTorch. Repeated framework and command-queue boundaries would likely erase their benefit.
- Another standalone Flash Attention implementation. The fork already has a measured native MFA route.
- Required GGUF conversion. Existing Automatic1111 checkpoints remain the default input until an optional native backend proves itself.
- VAE tiling at ordinary 512-pixel resolutions. It reduces peak memory but normally increases latency.
- Sampler substitution. stable-diffusion.cpp supports related DPM++ samplers, but this project must retain the exact A1111 DPM++ SDE behavior already chosen for LCM output.

## Upstream features and documentation

This README focuses on the fork. For the complete WebUI feature set, usage documentation, and extension ecosystem, see:

- [Automatic1111 feature overview](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features)
- [Automatic1111 wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki)
- [Automatic1111 API documentation](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API)
- [Automatic1111 troubleshooting](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Troubleshooting)

## Contributing

Keep changes narrow, measurable, and safe to fall back from.

For performance work:

1. Record the exact model hash and generation settings.
2. Warm up both paths.
3. Compare multiple alternating runs rather than a single best time.
4. Verify deterministic behavior within each path.
5. Measure output deviation as well as latency and memory.
6. Retain the upstream PyTorch path for unsupported inputs and runtime failure.

Changes that improve an isolated operator but do not improve an end-to-end generation should not be enabled by default.

## License and credits

This fork retains Automatic1111's license and third-party notices. Licenses for bundled and borrowed components are available under **Settings → Licenses** and in `html/licenses.html`.

Primary credit remains with the [Automatic1111 project](https://github.com/AUTOMATIC1111/stable-diffusion-webui) and its contributors. The native attention work builds on [`mps-flash-sdpa`](https://pypi.org/project/mps-flash-sdpa/) and ideas explored by Draw Things, adapted here for Automatic1111's PyTorch MPS execution path.
