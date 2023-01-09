import unittest


class TestExtrasWorking(unittest.TestCase):
    def setUp(self):
        self.url_img2img = "http://localhost:7860/sdapi/v1/extra-single-image"
        self.simple_extras = {
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
            "image": ""
            }


class TestExtrasCorrectness(unittest.TestCase):
    pass


if __name__ == "__main__":
    unittest.main()
