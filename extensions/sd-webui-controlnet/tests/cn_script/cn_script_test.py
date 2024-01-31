from typing import Any, Dict, List
import unittest
from PIL import Image
import numpy as np

import importlib

utils = importlib.import_module("extensions.sd-webui-controlnet.tests.utils", "utils")


from scripts import external_code, processor
from scripts.controlnet import prepare_mask, Script, set_numpy_seed
from modules import processing


class TestPrepareMask(unittest.TestCase):
    def test_prepare_mask(self):
        p = processing.StableDiffusionProcessing()
        p.inpainting_mask_invert = True
        p.mask_blur = 5

        mask = Image.new("RGB", (10, 10), color="white")

        processed_mask = prepare_mask(mask, p)

        # Check that mask is correctly converted to grayscale
        self.assertTrue(processed_mask.mode, "L")

        # Check that mask colors are correctly inverted
        self.assertEqual(
            processed_mask.getpixel((0, 0)), 0
        )  # inverted white should be black

        p.inpainting_mask_invert = False
        processed_mask = prepare_mask(mask, p)

        # Check that mask colors are not inverted when 'inpainting_mask_invert' is False
        self.assertEqual(
            processed_mask.getpixel((0, 0)), 255
        )  # white should remain white

        p.mask_blur = 0
        mask = Image.new("RGB", (10, 10), color="black")
        processed_mask = prepare_mask(mask, p)

        # Check that mask is not blurred when 'mask_blur' is 0
        self.assertEqual(
            processed_mask.getpixel((0, 0)), 0
        )  # black should remain black


class TestSetNumpySeed(unittest.TestCase):
    def test_seed_subseed_minus_one(self):
        p = processing.StableDiffusionProcessing()
        p.seed = -1
        p.subseed = -1
        p.all_seeds = [123, 456]
        expected_seed = (123 + 123) & 0xFFFFFFFF
        self.assertEqual(set_numpy_seed(p), expected_seed)

    def test_valid_seed_subseed(self):
        p = processing.StableDiffusionProcessing()
        p.seed = 50
        p.subseed = 100
        p.all_seeds = [123, 456]
        expected_seed = (50 + 100) & 0xFFFFFFFF
        self.assertEqual(set_numpy_seed(p), expected_seed)

    def test_invalid_seed_subseed(self):
        p = processing.StableDiffusionProcessing()
        p.seed = "invalid"
        p.subseed = 2.5
        p.all_seeds = [123, 456]
        self.assertEqual(set_numpy_seed(p), None)

    def test_empty_all_seeds(self):
        p = processing.StableDiffusionProcessing()
        p.seed = -1
        p.subseed = 2
        p.all_seeds = []
        self.assertEqual(set_numpy_seed(p), None)

    def test_random_state_change(self):
        p = processing.StableDiffusionProcessing()
        p.seed = 50
        p.subseed = 100
        p.all_seeds = [123, 456]
        expected_seed = (50 + 100) & 0xFFFFFFFF

        np.random.seed(0)  # set a known seed
        before_random = np.random.randint(0, 1000)  # get a random integer

        seed = set_numpy_seed(p)
        self.assertEqual(seed, expected_seed)

        after_random = np.random.randint(0, 1000)  # get another random integer

        self.assertNotEqual(before_random, after_random)


class MockImg2ImgProcessing(processing.StableDiffusionProcessing):
    """Mock the Img2Img processing as the WebUI version have dependency on
    `sd_model`."""

    def __init__(self, init_images, resize_mode, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_images = init_images
        self.resize_mode = resize_mode


class TestScript(unittest.TestCase):
    sample_base64_image = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAARMAAAC3CAIAAAC+MS2jAAAAqUlEQVR4nO3BAQ"
        "0AAADCoPdPbQ8HFAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAA/wZOlAAB5tU+nAAAAABJRU5ErkJggg=="
    )

    sample_np_image = np.array(
        [[100, 200, 50], [150, 75, 225], [30, 120, 180]], dtype=np.uint8
    )

    def test_bound_check_params(self):
        def param_required(module: str, param: str) -> bool:
            configs = processor.preprocessor_sliders_config[module]
            config_index = ("processor_res", "threshold_a", "threshold_b").index(param)
            return config_index < len(configs) and configs[config_index] is not None

        for module in processor.preprocessor_sliders_config.keys():
            for param in ("processor_res", "threshold_a", "threshold_b"):
                with self.subTest(param=param, module=module):
                    unit = external_code.ControlNetUnit(
                        module=module,
                        **{param: -100},
                    )
                    Script.bound_check_params(unit)
                    if param_required(module, param):
                        self.assertGreaterEqual(getattr(unit, param), 0)
                    else:
                        self.assertEqual(getattr(unit, param), -100)

    def test_choose_input_image(self):
        with self.subTest(name="no image"):
            with self.assertRaises(ValueError):
                Script.choose_input_image(
                    p=processing.StableDiffusionProcessing(),
                    unit=external_code.ControlNetUnit(),
                    idx=0,
                )

        with self.subTest(name="control net input"):
            _, resize_mode = Script.choose_input_image(
                p=MockImg2ImgProcessing(
                    init_images=[TestScript.sample_np_image],
                    resize_mode=external_code.ResizeMode.OUTER_FIT,
                ),
                unit=external_code.ControlNetUnit(
                    image=TestScript.sample_base64_image,
                    module="none",
                    resize_mode=external_code.ResizeMode.INNER_FIT,
                ),
                idx=0,
            )
            self.assertEqual(resize_mode, external_code.ResizeMode.INNER_FIT)

        with self.subTest(name="A1111 input"):
            _, resize_mode = Script.choose_input_image(
                p=MockImg2ImgProcessing(
                    init_images=[TestScript.sample_np_image],
                    resize_mode=external_code.ResizeMode.OUTER_FIT,
                ),
                unit=external_code.ControlNetUnit(
                    module="none",
                    resize_mode=external_code.ResizeMode.INNER_FIT,
                ),
                idx=0,
            )
            self.assertEqual(resize_mode, external_code.ResizeMode.OUTER_FIT)


if __name__ == "__main__":
    unittest.main()
