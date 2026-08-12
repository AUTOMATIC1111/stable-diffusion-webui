# Stable Diffusion WebUI Metal

An Apple Silicon performance fork of [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui), focused on faster and more memory-aware inference through PyTorch MPS and native Metal kernels.

The normal Automatic1111 interface, API, checkpoint layout, samplers, LoRA syntax, and extension structure are preserved. The fork adds a selective Metal attention path, fused GroupNorm + SiLU and exact-parity GEGLU kernels, unified-memory-aware attention fallback, an M1-validated FP16 VAE path, and tested macOS dependency defaults. Stable Diffusion 1.x inference—particularly short DPM++ SDE runs—is the primary optimization target.

> [!IMPORTANT]
> This is an experimental performance fork, not a new Stable Diffusion engine. It favors measured M1 inference performance and safe fallback behavior over broad hardware tuning. If a native Metal path is unavailable or fails its startup test, the WebUI falls back to the corresponding PyTorch implementation.

## How far is this from Automatic1111?

The Metal implementation at commit [`78c3fc98`](https://github.com/dmikey/stable-diffusion-webui-metal/commit/78c3fc988011add8f75dc66af259215d7fc56d2c) has an intentionally small, auditable delta from the official Automatic1111 `dev` branch. Documentation-only changes to this README are excluded from the implementation counts below.

| Measure | Value |
| --- | ---: |
| Automatic1111 base | [`1937682a`](https://github.com/AUTOMATIC1111/stable-diffusion-webui/commit/1937682a20f7f0442311a1ede68f9f0cb480163b) |
| Base version | `v1.10.1-96-g1937682a` |
| Metal implementation | [`78c3fc98`](https://github.com/dmikey/stable-diffusion-webui-metal/commit/78c3fc988011add8f75dc66af259215d7fc56d2c) |
| Implementation version | `v1.10.1-100-g78c3fc98` |
| Code relationship | 4 implementation commits ahead, 0 upstream commits behind official `dev` |
| Changed implementation paths | 20 of 329 tracked repository paths (6.1%) |
| New implementation files | 12 |
| Modified upstream implementation files | 8 |
| Implementation delta | 1,280 insertions, 46 deletions |

The four fork commits are:

1. Apple Silicon dependency, attention, memory, and benchmark foundation.
2. Removal of obsolete MPS safety copies on modern PyTorch.
3. Metal Flash Attention command-buffer coalescing.
4. Native fused GroupNorm + SiLU for compatible inference blocks.

Most of the added lines are isolated Metal code, benchmark utilities, and tests. The fork does **not** change checkpoint formats, prompt syntax, the REST API contract, or the core Gradio workflow.

<details>
<summary>Complete 20-path change surface</summary>

| Area | Added | Modified |
| --- | --- | --- |
| Metal runtime | `modules/mps_flash_attention.py`<br>`modules/mps_fused_ops.py`<br>`modules/mps_utils.py` | `modules/mac_specific.py`<br>`modules/sd_hijack_optimizations.py`<br>`modules/sd_hijack_unet.py`<br>`modules/sub_quadratic_attention.py` |
| Startup and defaults | `requirements_macos.txt` | `modules/launch_utils.py`<br>`modules/shared_options.py`<br>`requirements_versions.txt`<br>`webui-macos-env.sh` |
| Native build and benchmarks | `scripts/install_mps_flash_attention.py`<br>`scripts/mps_fused_group_norm.mm`<br>`scripts/benchmark_mps_attention.py`<br>`scripts/benchmark_mps_unet_ops.py`<br>`scripts/benchmark_mps_geglu_probe.py` | — |
| Tests | `test/test_mps_flash_attention.py`<br>`test/test_mps_fused_ops.py`<br>`test/test_mps_utils.py`<br>`test/test_sub_quadratic_attention.py` | — |

</details>

You can reproduce the comparison locally:

```bash
git rev-list --left-right --count 1937682a...78c3fc98
git diff --shortstat 1937682a..78c3fc98
git diff --name-status 1937682a..78c3fc98
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

That observed run was approximately **32% lower latency**, or **1.47× as fast**. The current head adds the fused GroupNorm + SiLU path after that recorded comparison.

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

## Deliberately not included

- Block-level MPSGraph execution: implemented and benchmarked, but rejected after a small regression.
- FP8 acceleration on M1: there is no matching M1 hardware fast path, so conversion would primarily add unpacking overhead.
- Core ML/ANE conversion: this would introduce a separate static execution engine and materially reduce Automatic1111 compatibility.
- Whole-UNet static graphs: potentially higher upside, but a much larger project with difficult LoRA, ControlNet, model-switching, and dynamic-resolution tradeoffs.
- Model-format changes or required quantization: existing Automatic1111 checkpoints are used directly.

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
