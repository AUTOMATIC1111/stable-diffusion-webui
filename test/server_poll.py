import unittest
import requests
import time


def run_tests(proc):
    timeout_threshold = 240
    start_time = time.time()
    while time.time()-start_time < timeout_threshold:
        try:
            requests.head("http://localhost:7860/")
            break
        except requests.exceptions.ConnectionError:
            if proc.poll() is not None:
                break
    if proc.poll() is None:
        suite = unittest.TestLoader().discover('', pattern='*_test.py')
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    else:
        print("Launch unsuccessful")
