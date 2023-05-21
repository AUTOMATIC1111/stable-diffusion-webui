You can run tests to validate your modifications.


## dev

Post [PR #10291](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/10291), [py.test](https://docs.pytest.org/en/7.3.x/) is used as the test runner. Testing dependencies are in `requirements-test.txt`, so `pip install -r requirements-test.txt` first.

Most of the tests run against a live instance of the WebUI. You can start the WebUI server with a suitable baseline configuration with the `--test-server` argument, but you may want to add e.g. `--use-cpu all --no-half` depending on your system.

Once the server is running, you can run tests with just `py.test`.

## master

To run tests, add `--tests TESTS_DIR` as a commandline argument to `launch.py` along with your other command line arguments.

There is a `basic test` that we used to verify the basic image creation works vi API, you can create other test for different scenarios.

To run the `basic test`, pass the argument `--tests test` to `launch.py`:
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