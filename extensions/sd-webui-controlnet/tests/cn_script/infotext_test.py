import unittest
import importlib

utils = importlib.import_module("extensions.sd-webui-controlnet.tests.utils", "utils")

from scripts.infotext import parse_unit
from scripts.external_code import ControlNetUnit


class TestInfotext(unittest.TestCase):
    def test_parsing(self):
        infotext = (
            "Module: inpaint_only+lama, Model: control_v11p_sd15_inpaint [ebff9138], Weight: 1, "
            "Resize Mode: Resize and Fill, Low Vram: False, Guidance Start: 0, Guidance End: 1, "
            "Pixel Perfect: True, Control Mode: Balanced, Hr Option: Both, Save Detected Map: True"
        )
        self.assertEqual(
            vars(
                ControlNetUnit(
                    module="inpaint_only+lama",
                    model="control_v11p_sd15_inpaint [ebff9138]",
                    weight=1,
                    resize_mode="Resize and Fill",
                    low_vram=False,
                    guidance_start=0,
                    guidance_end=1,
                    pixel_perfect=True,
                    control_mode="Balanced",
                    hr_option="Both",
                    save_detected_map=True,
                )
            ),
            vars(parse_unit(infotext)),
        )
