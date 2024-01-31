import unittest
import pytest
from typing import NamedTuple, Optional

from .template import (
    sd_version,
    StableDiffusionVersion,
    is_full_coverage,
    APITestTemplate,
    living_room_img,
    general_negative_prompt,
)

base_prompt = "A modern living room"

general_depth_modules = [
    "depth",
    "depth_leres",
    "depth_leres++",
    "depth_anything",
]
hand_refiner_module = "depth_hand_refiner"

general_depth_models = [
    "control_sd15_depth_anything [48a4bc3a]",
    "control_v11f1p_sd15_depth [cfd03158]",
    "t2iadapter_depth_sd15v2 [3489cd37]",
]
hand_refiner_model = "control_sd15_inpaint_depth_hand_fp16 [09456e54]"


class TestDepthFullCoverage(unittest.TestCase):
    def setUp(self):
        if not is_full_coverage:
            pytest.skip()
        # TODO test SDXL.
        if sd_version == StableDiffusionVersion.SDXL:
            pytest.skip()

    def test_depth(self):
        for module in general_depth_modules:
            for model in general_depth_models:
                name = f"depth_txt2img_{module}_{model}"
                with self.subTest(name=name):
                    self.assertTrue(
                        APITestTemplate(
                            name,
                            "txt2img",
                            payload_overrides={
                                "prompt": base_prompt,
                                "negative_prompt": general_negative_prompt,
                                "steps": 20,
                                "width": 768,
                                "height": 512,
                            },
                            unit_overrides={
                                "module": module,
                                "model": model,
                                "image": living_room_img,
                            },
                        ).exec(result_only=False)
                    )


if __name__ == "__main__":
    unittest.main()
