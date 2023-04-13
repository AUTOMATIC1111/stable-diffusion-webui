import os
import unittest
import requests
from gradio.processing_utils import encode_pil_to_base64
from PIL import Image
from modules.paths import script_path


class TestImg2ImgWorking(unittest.TestCase):
    def setUp(self):
        self.url_img2img = "http://localhost:7860/sdapi/v1/img2img"
        self.simple_img2img = {
            "init_images": [encode_pil_to_base64(Image.open(os.path.join(script_path, r"test/test_files/img2img_basic.png")))],
            "resize_mode": 0,
            "denoising_strength": 0.75,
            "mask": None,
            "mask_blur": 4,
            "inpainting_fill": 0,
            "inpaint_full_res": False,
            "inpaint_full_res_padding": 0,
            "inpainting_mask_invert": False,
            "prompt": "example prompt",
            "styles": [],
            "seed": -1,
            "subseed": -1,
            "subseed_strength": 0,
            "seed_resize_from_h": -1,
            "seed_resize_from_w": -1,
            "batch_size": 1,
            "n_iter": 1,
            "steps": 3,
            "cfg_scale": 7,
            "width": 64,
            "height": 64,
            "restore_faces": False,
            "tiling": False,
            "negative_prompt": "",
            "eta": 0,
            "s_churn": 0,
            "s_tmax": 0,
            "s_tmin": 0,
            "s_noise": 1,
            "override_settings": {},
            "sampler_index": "Euler a",
            "include_init_images": False
            }

    def test_img2img_simple_performed(self):
        self.assertEqual(requests.post(self.url_img2img, json=self.simple_img2img).status_code, 200)

    def test_inpainting_masked_performed(self):
        self.simple_img2img["mask"] = encode_pil_to_base64(Image.open(os.path.join(script_path, r"test/test_files/img2img_basic.png")))
        self.assertEqual(requests.post(self.url_img2img, json=self.simple_img2img).status_code, 200)

    def test_inpainting_with_inverted_masked_performed(self):
        self.simple_img2img["mask"] = encode_pil_to_base64(Image.open(os.path.join(script_path, r"test/test_files/img2img_basic.png")))
        self.simple_img2img["inpainting_mask_invert"] = True
        self.assertEqual(requests.post(self.url_img2img, json=self.simple_img2img).status_code, 200)

    def test_img2img_sd_upscale_performed(self):
        self.simple_img2img["script_name"] = "sd upscale"
        self.simple_img2img["script_args"] = ["", 8, "Lanczos", 2.0]

        self.assertEqual(requests.post(self.url_img2img, json=self.simple_img2img).status_code, 200)


if __name__ == "__main__":
    unittest.main()
