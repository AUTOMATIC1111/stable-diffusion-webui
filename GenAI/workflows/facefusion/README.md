# FaceFusion workflows and jobs

FaceFusion **3.6.1** is launched from the cloned repo via `facefusion.py`.

## Interactive
```bash
python facefusion.py run --execution-providers cuda
```
(macOS: `coreml` or `cpu`)

## Headless image (verify flags on your install)
```bash
python facefusion.py headless-run \
  -s inputs/facefusion/source/your-source.jpg \
  -t inputs/facefusion/target/your-target.jpg \
  -o outputs/facefusion/result.jpg
```

## Model download
```bash
python facefusion.py force-download
```

## Example config
See [example-image-job.ini](example-image-job.ini) for placeholder paths.

Official docs: https://docs.facefusion.io/
