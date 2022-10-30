There are tests that just verify that basic image creation works vi API.

To run tests, add `--tests` as a commandline argument to `launch.py` along with your other command line arguments:

```
python launch.py --skip-torch-cuda-test --deepdanbooru --no-half-vae --tests
```

You'll find outputs of main program in `test/stdout.txt` and `test/stderr.txt`.