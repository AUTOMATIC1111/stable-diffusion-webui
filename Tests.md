You can run tests to validate your modifications.

Post [PR #10291](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/10291), [py.test](https://docs.pytest.org/en/7.3.x/) is used as the test runner. Testing dependencies are in `requirements-test.txt`, so `pip install -r requirements-test.txt` first.

Most of the tests run against a live instance of the WebUI. You can start the WebUI server with a suitable baseline configuration with the `--test-server` argument, but you may want to add e.g. `--use-cpu all --no-half` depending on your system.

The command to run webui tests is: `python -m pytest -vv --verify-base-url test`