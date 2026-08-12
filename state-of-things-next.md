# State of Things and Native UNet Roadmap

Last updated: 2026-08-11

Target machine: 16 GB Apple M1 Mac mini

Branch: `dev`

Current committed head before this documentation update: `6eefbb402d177ec5166dbb364ea8e313d1bdb206`

Automatic1111 base: `1937682a20f7f0442311a1ede68f9f0cb480163b`

## Current state

This remains an Automatic1111 fork with targeted MPS and native Metal acceleration. It does not currently contain a separate diffusion engine.

The active optimization stack is:

1. Selective Draw Things-style Metal Flash Attention for measured SD 1.x shapes, encoded on PyTorch's current MPS command buffer.
2. Unified-memory-aware routing to native or sub-quadratic attention.
3. Native fused GroupNorm + SiLU for compatible FP16 UNet and VAE blocks.
4. Exact-parity fused GEGLU using a 65,536-entry, 128 KB PyTorch-generated FP16 GELU table.
5. Modern-PyTorch removal of obsolete MPS clones and FP32 LayerNorm workarounds.
6. FP16 VAE as the tracked default on the tested M1 family, with Automatic1111's FP32 NaN retry retained.
7. NGMS 1.0/all steps and Clip skip 2 as built-in defaults.
8. An opt-in coarse profiler enabled by `A1111_MPS_PROFILE=1` with no synchronization in the disabled path.

The native extension performs an isolated MPS startup test. Unsupported inputs and runtime failures retain PyTorch fallbacks.

## Reference workloads

Primary profiling workload:

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

The sampler makes nine UNet evaluations: five calls at batch two and four calls at batch one. At 384×640 the latent inputs are `2×4×80×48` and `1×4×80×48`. NGMS creates the batch-one regime.

The recurring 512×512 validation workload uses prompt `a dog`, seed `4017012032`, and the same model, sampler, schedule, CFG, Clip skip, and NGMS settings.

## Measured results

### Fork versus Automatic1111 baseline

At 384×640, the original Automatic1111 base recorded 12.8 seconds and an earlier fork head recorded 8.7 seconds at matching model hash and tensor shape: approximately 32% lower latency or 1.47× throughput. The paired runs used different seeds, so this is a throughput comparison rather than output parity.

### FP16 VAE

Five warm profiled 384×640 runs produced:

| VAE precision | End-to-end median | Sampler | VAE decode + transfer |
| --- | ---: | ---: | ---: |
| FP32 | 8.450 s | 6.715 s | 1.536 s |
| FP16 | 7.795 s | 6.666 s | 0.972 s |

FP16 saved about 0.65 seconds end to end and reduced the VAE stage by about 37%. Across three fixed-seed cases, PSNR versus FP32 was 64.0–64.6 dB, every changed RGB channel moved by at most one 8-bit value, and 97.4–97.7% of channels were identical.

### Exact GEGLU

A 512×512 alternating A/B produced identical PNG hashes and positive paired savings of approximately 0.12–0.27 seconds. An active LCM LoRA output hash was also identical with fusion enabled and disabled.

### Current range

A normal user run at 512×512 recorded 8.3 seconds. Later controlled warm A/B sessions commonly clustered around 9.2–9.3 seconds. Background work, extensions, thermal state, and unified-memory pressure are material at this scale.

## Rejected experiments

Do not repeat these without a new mechanism or new evidence:

| Experiment | Evidence | Result |
| --- | --- | --- |
| DPM++ 2M substitution | Changed the desired LCM/DPM++ SDE image behavior | Reject sampler substitution |
| FP8 on M1 | No M1 FP8 hardware execution path | Reject conversion/unpacking overhead |
| Per-operator MPS events on PyTorch 2.3 | Isolated synchronization hung | Use coarse profiling |
| Block-level MPSGraph | About 1% slower end to end with more numerical drift | Removed |
| Real-weight MPSGraph block | About 1.4% isolated stage gain versus fused GroupNorm | Failed integration gate |
| Fixed-shape TorchScript UNet | Inconsistent warm gain, lost after cache loss, increased retained memory | Removed |
| Fused LayerNorm | Projected 110 ms microbenchmark gain; paired end-to-end median regressed 0.026 s and only 56.1% of RGB channels matched | Removed |
| Cross-attention K/V reuse | Reused 112/144 projections and retained 11 MiB; paired end-to-end median regressed 0.016 s | Removed |

The LayerNorm and K/V code and saved probe settings were removed completely. The current repository and native extension contain neither path.

## What the profile says

After the FP16 VAE improvement, sampling/UNet consumes roughly 87% of measured generation time. Conditioning, image conversion, metadata, PNG creation, and additional VAE micro-tuning cannot provide the next material gain.

The failed LayerNorm and K/V experiments also show that transformer micro-operations are now below the useful granularity. The next work must reduce framework overhead across a large portion of the UNet or execute the complete UNet more efficiently.

## Chosen direction: native ggml/Metal UNet sidecar

Take architectural inspiration from stable-diffusion.cpp and ggml without replacing Automatic1111.

Keep in Automatic1111:

- Prompt parsing and conditioning.
- Existing DPM++ SDE/Karras sampler.
- CFG and NGMS decisions.
- Seed and RNG behavior.
- LoRA/extension activation and request routing.
- VAE, image pipeline, metadata, API, and UI.

Delegate only a supported UNet evaluation to a native graph runner. Unsupported requests continue through the current PyTorch MPS UNet.

stable-diffusion.cpp already demonstrates the relevant components: a complete SD 1.x UNet graph, graph-planned reusable buffers, whole-graph Metal encoding, safetensors/GGUF loading, LoRA support, Flash Attention, and fused quantized matrix kernels. The useful lesson is ownership of the complete graph and memory lifecycle, not copying individual kernels.

## Phase 0: capture the existing UNet contract

Create a diagnostic-only capture of all nine real calls for 384×640 and 512×512:

- Latent input and reference output.
- Timestep.
- Text conditioning.
- Shape, dtype, model hash, and request settings.
- Batch-two and batch-one regimes.

Include plain prompt, scheduled-prompt, active-LoRA, and unsupported/fallback fixtures. Disabled capture must add no synchronization or normal-path overhead.

Deliverable: a reproducible tensor corpus and a PyTorch replay test.

## Phase 1: standalone native shootout

Build a small harness around stable-diffusion.cpp's `UNetModelRunner`, load the same SD 1.x checkpoint, and replay the captures outside WebUI.

Measure complete nine-call warm latency, per-shape latency, retained/peak memory, determinism, and tensor deviation.

Gate: the native nine-call workload must be at least 20–25% faster than current PyTorch MPS. Stop the project here if it does not clear the gate; smaller gains will likely disappear behind bridge synchronization and compatibility work.

### M1 single-call probe result: stopped at the gate

On 2026-08-11, the first bounded probe captured the real first batch-two call from the 512×512 reference request. The fixture contains an FP16 `2×4×64×64` latent, timestep `[999, 999]`, FP16 cross-attention context `2×77×768`, and FP16 reference output. Replaying the captured inputs immediately through the existing PyTorch UNet produced a bit-for-bit identical output with zero mean and maximum absolute error.

Ten synchronized warm PyTorch MPS replays measured:

| Runner | Median | Minimum | Maximum |
| --- | ---: | ---: | ---: |
| Current PyTorch MPS UNet | 874.597 ms | 868.626 ms | 880.409 ms |

A fresh upstream stable-diffusion.cpp checkout at `bcc7e29` was built with its Metal backend and a temporary direct `UNetModelRunner` probe. With native Flash Attention enabled, three final measured calls produced:

| Runner | Median | Minimum | Maximum |
| --- | ---: | ---: | ---: |
| stable-diffusion.cpp Metal UNet | 2,192.908 ms | 2,187.225 ms | 2,203.125 ms |

The native call was approximately 2.51× slower than the current PyTorch MPS path before any Automatic1111 bridge or buffer-transfer overhead. It also returned 16,384 non-finite values out of 32,768 outputs—exactly one batch element—while the PyTorch reference contained none. Among finite values, mean absolute error was `0.0002314` and maximum absolute error was `0.0017264`.

Additional findings:

- Enabling mmap for the native Metal weights crashed in `ggml_metal_buffer_get_id`; disabling mmap allowed the probe to complete.
- Disabling native Flash Attention increased median latency to approximately 10.30 seconds per call and did not eliminate the non-finite output.
- The capture and immediate PyTorch replay were exact, so the input corpus itself passed its accuracy check.

Decision: do not proceed to a copied-buffer WebUI integration with this native runner. It misses the required speed gate by a wide margin and currently fails numerical validity. Retain the opt-in capture tooling as a small reusable test for a materially different future engine, but treat the stable-diffusion.cpp sidecar described below as rejected on the tested M1 implementation.

## Phase 2: copied-buffer WebUI prototype

Expose a minimal native interface for latent, timestep, conditioning, and UNet output. Keep A1111's sampler in control. A first implementation may synchronize and copy once per UNet call to prove integration.

Initial supported route:

- Apple M1.
- SD 1.x FP16.
- Txt2img, batch one.
- Tested 384×640 and 512×512 shapes.
- No active LoRA, ControlNet, hypernetwork, training, or high-resolution pass.

Implement it as an optional `SdUnetOption`. Every unsupported condition routes to PyTorch.

Gate: multiple alternating end-to-end pairs must remain materially faster, deterministic, and within an explicitly approved output-deviation envelope.

## Phase 3: zero-copy unified-memory proof

If the copied bridge wins, share the underlying Metal storage between PyTorch MPS and ggml.

Solve and test:

- `MTLBuffer` ownership and lifetime.
- Buffer offsets, strides, NCHW layout, and dtype agreement.
- PyTorch and ggml command-queue ordering.
- Error recovery and backend reset.

Start with one captured UNet call. Do not attempt full sampling until shared-buffer output matches the copied native implementation.

## Phase 4: compatibility expansion

Add independently gated support in this order:

1. Dynamic SD 1.x resolutions and reusable arenas per shape/batch regime.
2. LoRA weight application plus explicit model-mutation generation counters.
3. Img2img and inpainting.
4. High-resolution pass and model switching.
5. ControlNet where native semantics can match A1111.
6. Additional model families.

Never silently ignore an installed extension hook. Fall back to PyTorch for any request whose semantics the native backend cannot reproduce.

## Phase 5: optional quantization

Only after FP16 proves the native engine:

1. Q8_0 for the lowest-risk memory experiment.
2. Q6_K/Q5_K as optional balanced modes.
3. Q4 as an explicit low-memory mode, not a default.

Quantization must use native fused dequantization/matrix kernels. Do not add quantized PyTorch storage with per-call unpacking. Require tensor, fixed-seed image, LoRA, memory, and timing validation for every format.

## Global gates

1. Preserve A1111's exact DPM++ SDE evaluation sequence and both batch regimes.
2. Benchmark alternating warm pairs, never a single best run.
3. Report tensor and final-image deviation.
4. Measure retained unified memory and cache-loss behavior.
5. Preserve deterministic output within each path.
6. Keep automatic PyTorch fallback for unsupported features and runtime errors.
7. Never require checkpoint conversion for the normal PyTorch path.
8. Keep the native backend independently disableable.

## Explicit non-goals for the first sprint

- Replacing the A1111 UI, API, sampler, VAE, or extension ecosystem.
- Calling individual ggml convolutions or matrix kernels from PyTorch.
- Adding another Flash Attention implementation.
- Making GGUF mandatory.
- Supporting every model family or extension before the SD 1.x proof.
- Committing a backend before the standalone 20–25% gate passes.

## Immediate next task

Do not begin native WebUI integration. The single-call native shootout already failed the speed and validity gates. If a materially different engine becomes available, reuse the opt-in capture tooling and require it to beat the synchronized 874.597 ms batch-two PyTorch reference by at least 20–25% before expanding to batch one or the complete nine-call sequence.
