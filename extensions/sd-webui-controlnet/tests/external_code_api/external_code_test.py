import unittest
import importlib

import numpy as np

utils = importlib.import_module('extensions.sd-webui-controlnet.tests.utils', 'utils')


from copy import copy
from scripts import external_code
from scripts import controlnet
from modules import scripts, ui, shared


class TestExternalCodeWorking(unittest.TestCase):
    max_models = 6
    args_offset = 10

    def setUp(self):
        self.scripts = copy(scripts.scripts_txt2img)
        self.scripts.initialize_scripts(False)
        ui.create_ui()
        self.cn_script = controlnet.Script()
        self.cn_script.args_from = self.args_offset
        self.cn_script.args_to = self.args_offset + self.max_models
        self.scripts.alwayson_scripts = [self.cn_script]
        self.script_args = [None] * self.cn_script.args_from

        self.initial_max_models = shared.opts.data.get("control_net_unit_count", 3)
        shared.opts.data.update(control_net_unit_count=self.max_models)

        self.extra_models = 0

    def tearDown(self):
        shared.opts.data.update(control_net_unit_count=self.initial_max_models)

    def get_expected_args_to(self):
        args_len = max(self.max_models, len(self.cn_units))
        return self.args_offset + args_len

    def assert_update_in_place_ok(self):
        external_code.update_cn_script_in_place(self.scripts, self.script_args, self.cn_units)
        self.assertEqual(self.cn_script.args_to, self.get_expected_args_to())

    def test_empty_resizes_min_args(self):
        self.cn_units = []
        self.assert_update_in_place_ok()

    def test_empty_resizes_extra_args(self):
        extra_models = 1
        self.cn_units = [external_code.ControlNetUnit()] * (self.max_models + extra_models)
        self.assert_update_in_place_ok()


class TestControlNetUnitConversion(unittest.TestCase):
    def setUp(self):
        self.dummy_image = 'base64...'
        self.input = {}
        self.expected = external_code.ControlNetUnit()

    def assert_converts_to_expected(self):
        self.assertEqual(vars(external_code.to_processing_unit(self.input)), vars(self.expected))

    def test_empty_dict_works(self):
        self.assert_converts_to_expected()

    def test_image_works(self):
        self.input = {
            'image': self.dummy_image
        }
        self.expected = external_code.ControlNetUnit(image=self.dummy_image)
        self.assert_converts_to_expected()

    def test_image_alias_works(self):
        self.input = {
            'input_image': self.dummy_image
        }
        self.expected = external_code.ControlNetUnit(image=self.dummy_image)
        self.assert_converts_to_expected()

    def test_masked_image_works(self):
        self.input = {
            'image': self.dummy_image,
            'mask': self.dummy_image,
        }
        self.expected = external_code.ControlNetUnit(image={'image': self.dummy_image, 'mask': self.dummy_image})
        self.assert_converts_to_expected()


class TestControlNetUnitImageToDict(unittest.TestCase):
    def setUp(self):
        self.dummy_image = utils.readImage("test/test_files/img2img_basic.png")
        self.input = external_code.ControlNetUnit()
        self.expected_image = external_code.to_base64_nparray(self.dummy_image)
        self.expected_mask = external_code.to_base64_nparray(self.dummy_image)

    def assert_dict_is_valid(self):
        actual_dict = controlnet.image_dict_from_any(self.input.image)
        self.assertEqual(actual_dict['image'].tolist(), self.expected_image.tolist())
        self.assertEqual(actual_dict['mask'].tolist(), self.expected_mask.tolist())

    def test_none(self):
        self.assertEqual(controlnet.image_dict_from_any(self.input.image), None)

    def test_image_without_mask(self):
        self.input.image = self.dummy_image
        self.expected_mask = np.zeros_like(self.expected_image, dtype=np.uint8)
        self.assert_dict_is_valid()

    def test_masked_image_tuple(self):
        self.input.image = (self.dummy_image, self.dummy_image,)
        self.assert_dict_is_valid()

    def test_masked_image_dict(self):
        self.input.image = {'image': self.dummy_image, 'mask': self.dummy_image}
        self.assert_dict_is_valid()


class TestPixelPerfectResolution(unittest.TestCase):
    def test_outer_fit(self):
        image = np.zeros((100, 100, 3))
        target_H, target_W = 50, 100
        resize_mode = external_code.ResizeMode.OUTER_FIT
        result = external_code.pixel_perfect_resolution(image, target_H, target_W, resize_mode)
        expected = 50  # manually computed expected result
        self.assertEqual(result, expected)

    def test_inner_fit(self):
        image = np.zeros((100, 100, 3))
        target_H, target_W = 50, 100
        resize_mode = external_code.ResizeMode.INNER_FIT
        result = external_code.pixel_perfect_resolution(image, target_H, target_W, resize_mode)
        expected = 100  # manually computed expected result
        self.assertEqual(result, expected)


class TestGetAllUnitsFrom(unittest.TestCase):
    def test_none(self):
        self.assertListEqual(external_code.get_all_units_from([None]), [])

    def test_bool(self):
        self.assertListEqual(external_code.get_all_units_from([True]), [])

    def test_inheritance(self):
        class Foo(external_code.ControlNetUnit):
            def __init__(self):
                super().__init__(self)
                self.bar = 'a'
        
        foo = Foo()
        self.assertListEqual(external_code.get_all_units_from([foo]), [foo])

    def test_dict(self):
        units = external_code.get_all_units_from([{}])
        self.assertGreater(len(units), 0)
        self.assertIsInstance(units[0], external_code.ControlNetUnit)

    def test_unitlike(self):
        class Foo(object):
            """ bar """

        foo = Foo()
        for key in vars(external_code.ControlNetUnit()).keys():
            setattr(foo, key, True)
        setattr(foo, 'bar', False)
        self.assertListEqual(external_code.get_all_units_from([foo]), [foo])


if __name__ == '__main__':
    unittest.main()