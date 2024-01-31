import unittest
import importlib
utils = importlib.import_module('extensions.sd-webui-controlnet.tests.utils', 'utils')


from scripts import external_code


class TestImportlibReload(unittest.TestCase):
    def setUp(self):
        self.ControlNetUnit = external_code.ControlNetUnit

    def test_reload_does_not_redefine(self):
        importlib.reload(external_code)
        NewControlNetUnit = external_code.ControlNetUnit
        self.assertEqual(self.ControlNetUnit, NewControlNetUnit)

    def test_force_import_does_not_redefine(self):
        external_code_copy = importlib.import_module('extensions.sd-webui-controlnet.scripts.external_code', 'external_code')
        self.assertEqual(self.ControlNetUnit, external_code_copy.ControlNetUnit)


if __name__ == '__main__':
    unittest.main()
