import os
import unittest
import requests
from gradio.processing_utils import encode_pil_to_base64
from PIL import Image
from modules.paths import script_path

class TestExtrasWorking(unittest.TestCase):
    def setUp(self):
        self.url_extras_single = "http://localhost:7860/sdapi/v1/extra-single-image"
        self.extras_single = {
            "resize_mode": 0,
            "show_extras_results": True,
            "gfpgan_visibility": 0,
            "codeformer_visibility": 0,
            "codeformer_weight": 0,
            "upscaling_resize": 2,
            "upscaling_resize_w": 128,
            "upscaling_resize_h": 128,
            "upscaling_crop": True,
            "upscaler_1": "None",
            "upscaler_2": "None",
            "extras_upscaler_2_visibility": 0,
            "image": encode_pil_to_base64(Image.open(os.path.join(script_path, r"test/test_files/img2img_basic.png")))
            }

    def test_simple_upscaling_performed(self):
        self.extras_single["upscaler_1"] = "Lanczos"
        self.assertEqual(requests.post(self.url_extras_single, json=self.extras_single).status_code, 200)


class TestPngInfoWorking(unittest.TestCase):
    def setUp(self):
        self.url_png_info = "http://localhost:7860/sdapi/v1/extra-single-image"
        self.png_info = {
            "image": encode_pil_to_base64(Image.open(os.path.join(script_path, r"test/test_files/img2img_basic.png")))
        }

    def test_png_info_performed(self):
        self.assertEqual(requests.post(self.url_png_info, json=self.png_info).status_code, 200)


class TestInterrogateWorking(unittest.TestCase):
    def setUp(self):
        self.url_interrogate = "http://localhost:7860/sdapi/v1/extra-single-image"
        self.interrogate = {
            "image": encode_pil_to_base64(Image.open(os.path.join(script_path, r"test/test_files/img2img_basic.png"))),
            "model": "clip"
        }

    def test_interrogate_performed(self):
        self.assertEqual(requests.post(self.url_interrogate, json=self.interrogate).status_code, 200)


if __name__ == "__main__":
    unittest.main()
