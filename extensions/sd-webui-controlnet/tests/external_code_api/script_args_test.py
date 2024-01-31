import unittest
import importlib
utils = importlib.import_module('extensions.sd-webui-controlnet.tests.utils', 'utils')


from scripts import external_code


class TestGetAllUnitsFrom(unittest.TestCase):
    def setUp(self):
        self.control_unit = {
            "module": "none",
            "model": utils.get_model("canny"),
            "image": utils.readImage("test/test_files/img2img_basic.png"),
            "resize_mode": 1,
            "low_vram": False,
            "processor_res": 64,
            "control_mode": external_code.ControlMode.BALANCED.value,
        }
        self.object_unit = external_code.ControlNetUnit(**self.control_unit)

    def test_empty_converts(self):
        script_args = []
        units = external_code.get_all_units_from(script_args)
        self.assertListEqual(units, [])

    def test_object_forwards(self):
        script_args = [self.object_unit]
        units = external_code.get_all_units_from(script_args)
        self.assertListEqual(units, [self.object_unit])


if __name__ == '__main__':
    unittest.main()