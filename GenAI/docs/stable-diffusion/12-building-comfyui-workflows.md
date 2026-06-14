# Building ComfyUI workflows

1. Start from `beginner-text-to-image.json`
2. Add LoraLoader after checkpoint if needed
3. Insert ControlNetApply for guided generation
4. Use **Preview Image** before Save for fast iteration
5. Save incremental versions with descriptive names

Export JSON to `workflows/comfyui/` for Git tracking (no embedded private images).
