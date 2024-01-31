import unittest
import importlib
import requests

utils = importlib.import_module(
    'extensions.sd-webui-controlnet.tests.utils', 'utils')


from scripts.processor import preprocessor_filters


class TestControlTypes(unittest.TestCase):
    def test_fetching_control_types(self):
        response = requests.get(utils.BASE_URL + "/controlnet/control_types")
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn('control_types', result)

        for control_type in preprocessor_filters:
            self.assertIn(control_type, result['control_types'])


if __name__ == "__main__":
    unittest.main()
