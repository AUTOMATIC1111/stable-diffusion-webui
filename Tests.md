You can run tests to validate your modifications to weubi.

There is a `basic test` that we used to verify the basic image creation works vi API, you can create other test for different scenarios.

To run tests, add `--tests TESTS_DIR` as a commandline argument to `launch.py` along with your other command line arguments.

for the `basic test`, pass `--tests test` to `launch.py`:
```sh
python launch.py --tests test
```
for CPU only test:
```sh
python launch.py --tests test --use-cpu all --no-half
```
the test the arguments used for the [automated GitHub actions upon Pull Requests](https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/.github/workflows/run_tests.yaml) is:
```sh
python launch.py --tests test --no-half --disable-opt-split-attention --use-cpu all --skip-torch-cuda-test
```

You'll find outputs of main program in `test/stdout.txt` and `test/stderr.txt`.