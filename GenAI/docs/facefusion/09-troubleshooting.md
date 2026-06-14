# FaceFusion troubleshooting

| Symptom | Cause | Diagnostic | Fix |
|---------|-------|------------|-----|
| No face detected | Poor source | doctor + preview | Better source photo |
| CUDA provider missing | Wrong onnxruntime | doctor ONNX line | Re-run setup-facefusion |
| FFmpeg error | Not installed | `ffmpeg -version` | Install FFmpeg |
| Port occupied | Other Gradio app | doctor port check | Change port / stop app |
| Temp disk full | Long video | Check facefusion-temp | Clear temp, shorten clip |

Logs: `logs/facefusion/setup.log`, terminal output.
