# macOS troubleshooting

| Symptom | Action |
|---------|--------|
| `Permission denied` on scripts | `chmod +x scripts/macos/*.sh scripts/lib/common.sh` |
| MPS False on M-series | Reinstall torch in comfyui venv via setup-comfyui.sh |
| CoreML unavailable | Set `FACEFUSION_EXECUTION_PROVIDER=cpu` |
| FFmpeg missing | `brew install ffmpeg` |

Run `./scripts/macos/doctor.sh` first.
