# Changelog

## [0.1.0] - 2026-06-14

### Added
- Initial GenAI project scaffold
- ComfyUI v0.9.2 and FaceFusion 3.6.1 upstream pins
- Windows PowerShell and macOS shell scripts (setup, launch, doctor, repair, reset, update, smoke-test)
- Separate venv environments under `runtime/environments/`
- Starter ComfyUI workflow and tutorial documentation
- Configuration templates and validation tests
- Architecture decision records (ADR-001 through ADR-004)

### Not tested in CI
- Full ComfyUI image generation (requires user checkpoint)
- FaceFusion face swap (requires authorized local media)
- Apple Silicon MPS / CoreML hardware acceleration
- NVIDIA RTX 4090 CUDA acceleration
