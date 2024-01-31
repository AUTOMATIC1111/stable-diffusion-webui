import os
import unittest
import importlib
utils = importlib.import_module('extensions.sd-webui-controlnet.tests.utils', 'utils')
import requests
from scripts.enums import StableDiffusionVersion


class TestImg2ImgWorkingBase(unittest.TestCase):
    def setUp(self):
        sd_version = StableDiffusionVersion(int(
            os.environ.get("CONTROLNET_TEST_SD_VERSION", StableDiffusionVersion.SD1x.value)))
        self.model = utils.get_model("canny", sd_version)

        controlnet_unit = {
            "module": "none",
            "model": self.model,
            "weight": 1.0,
            "input_image": utils.readImage("test/test_files/img2img_basic.png"),
            "mask": utils.readImage("test/test_files/img2img_basic.png"),
            "resize_mode": 1,
            "lowvram": False,
            "processor_res": 64,
            "threshold_a": 64,
            "threshold_b": 64,
            "guidance_start": 0.0,
            "guidance_end": 1.0,
            "control_mode": 0,
        }
        setup_args = {"alwayson_scripts":{"ControlNet":{"args": ([controlnet_unit] * getattr(self, 'units_count', 1))}}}
        self.setup_route(setup_args)

    def setup_route(self, setup_args):
        self.url_img2img = "http://localhost:7860/sdapi/v1/img2img"
        self.simple_img2img = {
            "init_images": [utils.readImage("test/test_files/img2img_basic.png")],
            "resize_mode": 0,
            "denoising_strength": 0.75,
            "image_cfg_scale": 0,
            "mask_blur": 4,
            "inpainting_fill": 0,
            "inpaint_full_res": True,
            "inpaint_full_res_padding": 0,
            "inpainting_mask_invert": 0,
            "initial_noise_multiplier": 0,
            "prompt": "example prompt",
            "styles": [],
            "seed": -1,
            "subseed": -1,
            "subseed_strength": 0,
            "seed_resize_from_h": -1,
            "seed_resize_from_w": -1,
            "sampler_name": "Euler a",
            "batch_size": 1,
            "n_iter": 1,
            "steps": 3,
            "cfg_scale": 7,
            "width": 64,
            "height": 64,
            "restore_faces": False,
            "tiling": False,
            "do_not_save_samples": False,
            "do_not_save_grid": False,
            "negative_prompt": "",
            "eta": 0,
            "s_churn": 0,
            "s_tmax": 0,
            "s_tmin": 0,
            "s_noise": 1,
            "override_settings": {},
            "override_settings_restore_afterwards": True,
            "sampler_index": "Euler a",
            "include_init_images": False,
            "send_images": True,
            "save_images": False,
            "alwayson_scripts": {}
        }
        self.simple_img2img.update(setup_args)

    def assert_status_ok(self):
        self.assertEqual(requests.post(self.url_img2img, json=self.simple_img2img).status_code, 200)

    def test_img2img_simple_performed(self):
        self.assert_status_ok()

    def test_img2img_alwayson_scripts_default_units(self):
        self.units_count = 0
        self.setUp()
        self.assert_status_ok()

    def test_img2img_default_params(self):
        self.simple_img2img["alwayson_scripts"]["ControlNet"]["args"] = [{
            "input_image": utils.readImage("test/test_files/img2img_basic.png"),
            "model": self.model,
        }]
        self.assert_status_ok()

if __name__ == "__main__":
    unittest.main()