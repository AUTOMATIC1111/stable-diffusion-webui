# Prompt engineering

Structure prompts: **subject → action → composition → camera → lighting → environment → style**.

## Example 1 — portrait
**Prompt:** `portrait of a woman, three-quarter view, soft window light, neutral background, 85mm lens, shallow depth of field, natural skin texture, photorealistic`

**Settings:** 512×512, steps 25, CFG 7, euler

**Why it works:** Specifies framing, lens, and light before style token.

**Failure mode:** `beautiful woman` alone → generic, oversaturated faces.

**Refined:** Add `subtle catchlight in eyes, muted color palette` and negative `plastic skin, oversharpened`.

## Example 2 — product
**Prompt:** `matte black headphones on white marble surface, top-down flat lay, soft studio softbox, minimal shadows, commercial product photography`

**Negative:** `text, logo, watermark, cluttered background`

## Token weighting
ComfyUI CLIP encode accepts `(keyword:1.2)` in prompt text for emphasis.

## Iteration loop
1. Generate at 512 fast
2. Fix composition in prompt
3. Increase steps only if detail lacking
4. Raise CFG slightly for prompt adherence — watch for burned colors above 10
