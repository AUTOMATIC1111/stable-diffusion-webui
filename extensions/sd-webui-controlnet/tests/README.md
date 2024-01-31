# Tests
There are 2 types of tests:
- unittest: backend based tests that directly import A1111 shared modules
- api test: test functionality through A1111 web API

# Run tests locally
Make sure the current working directory is A1111 root.

## Install test dependencies
`pip install -r requirements-test.txt`

## Start test server
```shell
python -m coverage run
          --data-file=.coverage.server
          launch.py
          --skip-prepare-environment
          --skip-torch-cuda-test
          --test-server
          --do-not-download-clip
          --no-half
          --disable-opt-split-attention
          --use-cpu all
          --api-server-stop
```

## Setting environment variables
Setting `CONTROLNET_TEST_SD_VERSION` for stable diffusion model family used during testing.
- 1 for SD1.x
- 2 for SD2.x
- 3 for SDXL

## Run test
```shell
python -m pytest -vv --junitxml=test/results.xml --cov ./extensions/sd-webui-controlnet --cov-report=xml --verify-base-url ./extensions/sd-webui-controlnet/tests
```

## Check code coverage
Text report
```shell
python -m coverage report -i
```

HTML report
```shell
python -m coverage html -i
```