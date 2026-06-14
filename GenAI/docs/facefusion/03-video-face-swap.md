# Video face swap

## Before long videos
Process **5–10 second clip** first.

## Pipeline
1. FFmpeg extracts frames
2. FaceFusion processes each frame
3. Re-encode video; audio copied when configured

## Disk space
Temp frames in `runtime/facefusion-temp/` — ensure 2× video size free.

## Flicker
Caused by detection inconsistency — use consistent source angle; enable enhancement cautiously.

## Audio missing
Verify FFmpeg on PATH; check FaceFusion output encoder settings.
