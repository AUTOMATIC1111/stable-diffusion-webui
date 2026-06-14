# Batch and headless operation

Verify flags on pinned 3.6.1:
```bash
python facefusion.py --help
python facefusion.py run --help
python facefusion.py headless-run --help
```

Typical patterns:
```bash
python facefusion.py force-download
python facefusion.py headless-run -s SOURCE -t TARGET -o OUTPUT
```

Use placeholder paths from `workflows/facefusion/example-image-job.ini`.

Job system commands (if available in 3.6.1): check `job-list`, `job-create` via --help output.
