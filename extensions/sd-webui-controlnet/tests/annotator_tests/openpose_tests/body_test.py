import unittest
import numpy as np

import importlib
utils = importlib.import_module('extensions.sd-webui-controlnet.tests.utils', 'utils')


from annotator.openpose.body import Body, Keypoint, BodyResult

class TestFormatBodyResult(unittest.TestCase):
    def setUp(self):
        self.candidate = np.array([
            [10, 20, 0.9, 0],
            [30, 40, 0.8, 1],
            [50, 60, 0.7, 2],
            [70, 80, 0.6, 3]
        ])

        self.subset = np.array([
            [-1,  0,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1.7, 2],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  3, 0.6, 1]
        ])

    def test_format_body_result(self):
        expected_result = [
            BodyResult(
                keypoints=[
                    None,
                    Keypoint(x=10, y=20, score=0.9, id=0),
                    Keypoint(x=30, y=40, score=0.8, id=1),
                    None
                ] + [None] * 14,
                total_score=1.7,
                total_parts=2
            ),
            BodyResult(
                keypoints=[None] * 17 + [
                    Keypoint(x=70, y=80, score=0.6, id=3)
                ],
                total_score=0.6,
                total_parts=1
            )
        ]
        
        result = Body.format_body_result(self.candidate, self.subset)

        self.assertEqual(result, expected_result)

if __name__ == '__main__':
    unittest.main()        