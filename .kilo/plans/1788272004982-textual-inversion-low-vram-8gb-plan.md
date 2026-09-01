# Plan: Textual Inversion on 8 GB VRAM + Targeted App Improvements

## 1. Goal & constraints

- **Primary:** Make textual-inversion (embedding) training run on a CUDA GPU with ~8 GB VRAM, reliably.
- **Constraints (hard):**
  - **SD 1.5 / 2.x only.** SDXL UNet alone (~4.6 GB fp16) + text encoders (~2 GB) + VAE (~1.6 GB) cannot fit 8 GB even with every offload. The pipeline must detect an oversized checkpoint and refuse to train with a clear message.
  - Target must be CUDA NVIDIA (AMP/GradScaler/fp16 are CUDA paths). Do not try to "fix" CPU/MPS training.
- **Secondary (scoped):** small, safe app-quality fixes directly tied to the training path; not a full rewrite.

## 2. Root-cause: what burns VRAM during TI training

File refs: `modules/textual_inversion/textual_inversion.py:515` (loop), `:533-551` (forward), `modules/models/diffusion/ddpm_edit.py:1022` (`p_losses`), `modules/sd_hijack_checkpoint.py:22` (grad checkpoint), `modules/textual_inversion/ui.py:17`, `modules/api/api.py:805`.

1. **Forward+backward of the UNet** is the dominant activation cost.
   - `p_losses` → `q_sample` (add noise) → `apply_model` (full UNet forward) → loss. Activations are kept for backward unless checkpointed.
   - **Mitigated already:** `sd_hijack_checkpoint.add()` wraps `BasicTransformerBlock`, `ResBlock`, `AttentionBlock` with `torch.utils.checkpoint.checkpoint` — recomputes activations during backward. Already ON for TI. Keep as-is.
2. **Memory-efficient attention is UNDONE during training by default.**
   - `ui.py:21-24` and `api.py:808-819`: `apply_optimizations = shared.opts.training_xattention_optimizations` (default `False`). When `False`, `sd_hijack.undo_optimizations()` is called, reverting to *vanilla* cross-attention (stores full attention matrix) — the opposite of memory saving.
   - **This is the single biggest missed lever for low VRAM.** sdp/xformers attention saves a large activation chunk on the 512×512 latent (64×64 tokens) attention maps.
3. **VAE (first_stage_model) kept on GPU.**
   - `unload_models_when_training` (default `False`) moves VAE to CPU at `:476` and restores it only for preview decode at `:595`, back to CPU at `:626`, final restore at `:684`. **This works** but is off by default.
   - The option description at `shared_options.py:156` says "Move VAE **and CLIP** to RAM" but the code only moves the VAE. **Inaccuracy to fix.**
4. **CLIP text encoder always on GPU.** `c = shared.sd_model.cond_stage_model(batch.cond_text)` at `:537` runs CLIP every step. ~0.3 GB. Cannot be precomputed because the trained embedding vector feeds into CLIP per step. Offload is possible but adds per-step transfer cost; treat as optional, not required for 8 GB SD 1.5.
5. **GradScaler created unconditionally:** `scaler = torch.cuda.amp.GradScaler()` at `:493`. Harmless on CUDA; produces a warning/no-op elsewhere. Guard it.
6. **Preview image decode spikes VRAM.** Every `create_image_every` the VAE is moved back to GPU (`torch.cuda.amp.GradScaler()`...). On 8 GB this + UNet + activations can OOM. Default `create_image_every` is 500 — fine, but a user lowering it will spike.

### Estimated VRAM budget (SD 1.5, fp16 weights, 512×512, batch=1, grad_step=1)
| Component | fp16 VRAM | On 8 GB? |
|---|---|---|
| UNET weights | ~1.7 GB | yes |
| CLIP text encoder | ~0.3 GB | yes (on GPU) |
| VAE (offloaded to CPU) | ~0  (GPU) | yes |
| UNET fwd+bwd activations (grad-ckpt + sdp) | ~1.0–1.8 GB | yes |
| Embedding optimizer state | ~0 | yes |
| System / Python / CUDA context | ~0.6 GB | — |
| **Total (VAE offloaded + sdp attn)** | **~3.6–4.0 GB** | **leaves ~4 GB headroom** |

So 8 GB is comfortable *provided*: VAE offloaded + memory-efficient attention active + SD 1.5. That's the target configuration.

## 3. Key design decisions (recommended)

- **D1 — Low-VRAM training auto-enables memory-efficient attention.**
  - Add `devices.is_low_vram_training()` (or inline check): returns True when `get_vram_optimization_mode()` in `{"saver","ultra"}` or `get_safe_mode()` is True.
  - In `modules/textual_inversion/ui.py` and `modules/api/api.py`: set `apply_optimizations = shared.opts.training_xattention_optimizations or devices.is_low_vram_training()`. Rationale: when the user already opted into a saver/ultra/safe profile, they want every memory win; the per-step undo of attention optimizations should not run.
  - Does **not** change the default for normal RAM users (reproducibility preserved).
- **D2 — Fix `unload_models_when_training` semantics.**
  - Keep current behavior (offload VAE only) as the safe default, because CLIP offload hurts throughput and isn't required for 8 GB SD 1.5.
  - Fix the option description at `shared_options.py:156` to say "Move VAE to RAM when training" (drop the CLIP claim), OR — preferred — add an explicit new option `training_unload_clip_when_training` (default `False`) and implement CLIP offload in the loop if enabled. Recommendation: **fix the description now** (low risk) and ship CLIP-offload as a separate optional item if needed.
- **D3 — Guard GradScaler (non-CUDA).** Replace `torch.cuda.amp.GradScaler()` with a CUDA-guarded scaler. If not CUDA, skip `scaler.scale`/`scaler.step`/`scaler.update` and call `loss.backward()` / `optimizer.step()` directly. (Purely defensive; CUDA is the target.)
- **D4 — Preview decode memory spike.** For 8 GB guidance: set `create_image_every` to 0 or a large value, OR ensure preview decode uses the approximate VAE when available (respect `sd_vae_decode_method == "TAESD"` for the preview `p`). Recommendation for the plan: document the `create_image_every` levers; optionally wire TAESD for previews (secondary).
- **D5 — Reject oversized models.** In `validate_train_inputs` (or `train_embedding`), when `get_vram_optimization_mode() in {"saver","ultra"}` or safe mode, and the loaded model is SDXL/SD3, raise a clear error instead of an opaque CUDA OOM. (Secondary hardening.)
- **D6 — Model version check helper.** Reuse existing checkpoint metadata (`sd_models.select_checkpoint()`) to detect SDXL/SD3 by architecture params; no new heuristic needed beyond a parameter-count or `is_sdxl` flag check. Confirm the right attribute in impl.

## 4. Implementation tasks (ordered)

### Task 1 — Auto-enable attention optimizations for low VRAM (required)
- `modules/devices.py`: add `def is_low_vram_training(): return get_vram_optimization_mode() in ("saver","ultra") or get_safe_mode()`.
- `modules/textual_inversion/ui.py:17-37`: change `apply_optimizations = shared.opts.training_xattention_optimizations` → `apply_optimizations = shared.opts.training_xattention_optimizations or devices.is_low_vram_training()`. Import `devices`.
- `modules/api/api.py:808`: same change in `train_embedding` API method. Import `devices`.
- Verify `sd_hijack.apply_optimizations()` (startup) actually selects sdp/xformers based on the user's `cross_attention_optimization` setting so the "not undone" path uses memory-efficient attention.

### Task 2 — Fix option description / clarify VAE-only offload (required, docs)
- `modules/shared_options.py:156`: correct description to "Move VAE to RAM when training if possible. Saves VRAM."
- (Optional follow-up item, not blocking 8 GB: add `training_unload_clip_when_training` and implement CLIP offload around the cond forward at `textual_inversion.py:537`.)

### Task 3 — Guard GradScaler (small, defensive)
- `modules/textual_inversion/textual_inversion.py:493`: create scaler conditionally on `torch.cuda.is_available()`. Wrap `scaler.scale(loss).backward()`, `scaler.step(optimizer)`, `scaler.update()` so non-CUDA skips scaling.
- Same guard in `modules/hypernetworks/hypernetwork.py` training loop (shares the pattern at `:613`-ish) for consistency. *(Optional if scope is TI-only.)*

### Task 4 — Preview decode spike (docs + optional TAESD)
- Document for 8 GB users: set `create_image_every` to 0 (or high) and `save_image_with_stored_embedding = False` to avoid VAE-on-GPU spikes.
- *(Optional)* Route the preview `StableDiffusionProcessingTxt2Img` through the approximate VAE when `sd_vae_decode_method == "TAESD"` so preview decode is cheap. Leave for follow-up if scope is TI-only.

### Task 5 — Refuse oversized checkpoints on low VRAM (secondary hardening)
- In `train_embedding`/`validate_train_inputs`, when low-VRAM mode active, detect SDXL/SD3 checkpoint (via `checkpoint.sd_checkpoint` metadata or a `model.is_sdxl`-style attribute) and raise: `raise RuntimeError("Textual inversion training on 8 GB is only supported for SD 1.5/2.x; SDXL/SD3 will not fit.")`.

### Task 6 — 8 GB quick-start guidance (docs, no code)
- Add a short section to `README.md` (or a `docs/training-low-memory.md`) named "Training on low VRAM (8 GB / SD 1.5)" listing the exact settings: `--vram-optimization-mode ultra` (or `--medvram`), enable `training_xattention_optimizations`, set `unload_models_when_training = True`, `batch_size=1`, `gradient_step=1`, `nvpt=1`, 512×512, `create_image_every` small or 0.

## 5. Out of scope
- SDXL / SD3 training on 8 GB (infeasible) — explicitly unsupported, not engineered.
- Full CLIP/CPU offload of cond model for TI (not required; optional follow-up).
- LoRA training (different code path) — only textual inversion is in scope.
- Rewriting the dataset or switching to a different training framework.

## 6. Validation plan
- **Static:** `python -c "import ast; ast.parse(open('modules/textual_inversion/textual_inversion.py').read())"` and lint (`ruff check modules/textual_inversion modules/api/api.py modules/devices.py modules/shared_options.py`).
- **Unit:** run existing test suite `pytest test/` (TI-related tests if present) to confirm imports/CLI parse for `train_embedding`.
- **Behavioral (CUDA dev box, if available):**
  1. `--vram-optimization-mode saver` then start TI training; assert `sd_hijack.undo_optimizations()` is NOT called (i.e., attention optimizations remain) — check by spying on `BasicTransformerBlock.forward` identity vs. vanilla.
  2. Assert GradScaler path works on CUDA (no warning) and doesn't crash on a CPU-only import path.
  3. With `unload_models_when_training=True`, confirm VAE is on CPU during a training step (log device of `first_stage_model`).
  4. OOM-guard test: force safe-mode + SDXL checkpoint → confirm the clear error, not a raw CUDA OOM.
- If no GPU available, validate the decision logic with a CPU smoke test: import modules, instantiate optimizer path logic, confirm `is_low_vram_training()` returns expected boolean from opts.

## 7. Rollout / risk
- Risk is low: Task 1 only flips behavior when the user *already* requested saver/ultra/safe mode — a strict subset of users who are explicitly asking for low-VRAM behavior. Default reproducibility path is unchanged.
- GradScaler guard is purely defensive.
- Description fix is text-only.
- The hard "reject SDXL on 8 GB" should be gated to low-VRAM mode only so normal high-VRAM users are unaffected.
