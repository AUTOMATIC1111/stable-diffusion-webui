# Samplers, schedulers, steps, and CFG

## Steps
- 15–25 often sufficient for SD1.5 euler/dpm variants
- Diminishing returns after ~30 for many samplers

## CFG
- Low (3–5): creative, loose
- Mid (6–8): balanced default
- High (10+): harsh contrast, artifact risk

## Fair A/B
Change one knob at a time; lock seed.

## Oversaturation
Reduce CFG or add negative `oversaturated, hdr`.
