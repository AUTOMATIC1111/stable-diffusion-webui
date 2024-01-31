import unittest
import numpy as np

import importlib
utils = importlib.import_module('extensions.sd-webui-controlnet.tests.utils', 'utils')


from annotator.openpose.util import faceDetect, handDetect
from annotator.openpose.body import Keypoint, BodyResult

class TestFaceDetect(unittest.TestCase):
    def test_no_faces(self):
        oriImg = np.zeros((100, 100, 3), dtype=np.uint8)
        body = BodyResult([None] * 18, total_score=3, total_parts=0)
        expected_result = None
        result = faceDetect(body, oriImg)

        self.assertEqual(result, expected_result)

    def test_single_face(self):
        body = BodyResult([
            Keypoint(50, 50),
            *([None] * 13),
            Keypoint(30, 40),
            Keypoint(70, 40),
            Keypoint(20, 50),
            Keypoint(80, 50),
        ], total_score=2, total_parts=5)

        oriImg = np.zeros((100, 100, 3), dtype=np.uint8)

        expected_result = (0, 0, 120)
        result = faceDetect(body, oriImg)

        self.assertEqual(result, expected_result)

class TestHandDetect(unittest.TestCase):
    def test_no_hands(self):
        oriImg = np.zeros((100, 100, 3), dtype=np.uint8)
        body = BodyResult([None] * 18, total_score=3, total_parts=0)
        expected_result = []
        result = handDetect(body, oriImg)

        self.assertEqual(result, expected_result)

    def test_single_left_hand(self):
        oriImg = np.zeros((100, 100, 3), dtype=np.uint8)

        body = BodyResult([
            None, None, None, None, None,
            Keypoint(20, 20),
            Keypoint(40, 30),
            Keypoint(60, 40),
            *([None] * 8),
            Keypoint(20, 60),
            Keypoint(40, 70),
            Keypoint(60, 80)
        ], total_score=3, total_parts=0.5)

        expected_result = [(49, 26, 33, True)]
        result = handDetect(body, oriImg)

        self.assertEqual(result, expected_result)

    def test_single_right_hand(self):
        oriImg = np.zeros((100, 100, 3), dtype=np.uint8)

        body = BodyResult([
            None, None,
            Keypoint(20, 20),
            Keypoint(40, 30),
            Keypoint(60, 40),
            *([None] * 11),
            Keypoint(20, 60),
            Keypoint(40, 70),
            Keypoint(60, 80)
        ], total_score=3, total_parts=0.5)

        expected_result = [(49, 26, 33, False)]
        result = handDetect(body, oriImg)

        self.assertEqual(result, expected_result)

    def test_multiple_hands(self):
        body = BodyResult([
            Keypoint(20, 20),
            Keypoint(40, 30),
            Keypoint(60, 40),
            Keypoint(20, 60),
            Keypoint(40, 70),
            Keypoint(60, 80),
            Keypoint(10, 10),
            Keypoint(30, 20),
            Keypoint(50, 30),
            Keypoint(10, 50),
            Keypoint(30, 60),
            Keypoint(50, 70),
            *([None] * 6),
        ], total_score=3, total_parts=0.5)

        oriImg = np.zeros((100, 100, 3), dtype=np.uint8)

        expected_result = [(0, 0, 100, True), (16, 43, 56, False)]
        result = handDetect(body, oriImg)
        self.assertEqual(result, expected_result)


if __name__ == '__main__':
    unittest.main()
