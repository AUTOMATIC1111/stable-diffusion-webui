# TODO

## Issues

Stuff to be fixed...

- `mat1 and mat2 must have the same dtype` error (half mode)
- Some samplers won't work (test later)

## Something needs discussion

- About memory optimization.

Basically, we cannot get detailed vram information from `torch-directml`.

It has `gpu_memory` method which returns an array contains used memory size, but it is almostly useless without any other information.

What should we do?

1. Use any fixed value as the available memory capacity.
2. Use `atiadlxx`(AMD/ATI GPU driver library) to infer vram information as similar as possible to the actual value. (works for AMDGPUs)
3. or another better way.
