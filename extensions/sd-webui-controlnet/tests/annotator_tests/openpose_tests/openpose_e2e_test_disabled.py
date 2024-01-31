"""
Disabled because unloading openpose detection models is flaky.
See https://github.com/Mikubill/sd-webui-controlnet/actions/runs/6758718106/job/18370634881
for CI report of an example flaky run.
"""

import unittest
import cv2
import numpy as np
from pathlib import Path
from typing import Dict


import importlib
utils = importlib.import_module('extensions.sd-webui-controlnet.tests.utils', 'utils')


from annotator.openpose import OpenposeDetector

class TestOpenposeDetector(unittest.TestCase):
    image_path = str(Path(__file__).parent.parent.parent / 'images')
    def setUp(self) -> None:
        self.detector = OpenposeDetector()
        self.detector.load_model()

    def tearDown(self) -> None:
        self.detector.unload_model()

    def expect_same_image(self, img1, img2, diff_img_path: str):
        # Calculate the difference between the two images
        diff = cv2.absdiff(img1, img2)
        
        # Set a threshold to highlight the different pixels
        threshold = 30
        diff_highlighted = np.where(diff > threshold, 255, 0).astype(np.uint8)

        # Assert that the two images are similar within a tolerance
        similar = np.allclose(img1, img2, rtol=1e-05, atol=1e-08)
        if not similar:
            # Save the diff_highlighted image to inspect the differences
            cv2.imwrite(diff_img_path, cv2.cvtColor(diff_highlighted, cv2.COLOR_RGB2BGR))

        self.assertTrue(similar)

    # Save expectation image as png so that no compression issue happens.
    def template(self, test_image: str, expected_image: str, detector_config: Dict, overwrite_expectation: bool = False):
        oriImg = cv2.cvtColor(cv2.imread(test_image), cv2.COLOR_BGR2RGB)
        canvas = self.detector(oriImg, **detector_config)

        # Create expectation file
        if overwrite_expectation:
            cv2.imwrite(expected_image, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        else:
            expected_canvas = cv2.cvtColor(cv2.imread(expected_image), cv2.COLOR_BGR2RGB)
            self.expect_same_image(canvas, expected_canvas, diff_img_path=expected_image.replace('.png', '_diff.png'))

    def test_body(self):
        self.template(
            test_image = f'{TestOpenposeDetector.image_path}/ski.jpg',
            expected_image = f'{TestOpenposeDetector.image_path}/expected_ski_output.png',
            detector_config=dict(),
            overwrite_expectation=False
        )

    def test_hand(self):
        self.template(
            test_image = f'{TestOpenposeDetector.image_path}/woman.jpeg',
            expected_image = f'{TestOpenposeDetector.image_path}/expected_woman_hand_output.png',
            detector_config=dict(
                include_body=False,
                include_face=False,
                include_hand=True,
            ),
            overwrite_expectation=False
        )

    def test_face(self):
        self.template(
            test_image = f'{TestOpenposeDetector.image_path}/woman.jpeg',
            expected_image = f'{TestOpenposeDetector.image_path}/expected_woman_face_output.png',
            detector_config=dict(
                include_body=False,
                include_face=True,
                include_hand=False,                
            ),
            overwrite_expectation=False
        )
    
    def test_all(self):
        self.template(
            test_image = f'{TestOpenposeDetector.image_path}/woman.jpeg',
            expected_image = f'{TestOpenposeDetector.image_path}/expected_woman_all_output.png',
            detector_config=dict(
                include_body=True,
                include_face=True,
                include_hand=True,
            ),
            overwrite_expectation=False
        )

    def test_dw(self):
        self.template(
            test_image = f'{TestOpenposeDetector.image_path}/woman.jpeg',
            expected_image = f'{TestOpenposeDetector.image_path}/expected_woman_dw_all_output.png',
            detector_config=dict(
                include_body=True,
                include_face=True,
                include_hand=True,
                use_dw_pose=True,
            ),
            overwrite_expectation=False,
        )
        
if __name__ == '__main__':
    unittest.main()