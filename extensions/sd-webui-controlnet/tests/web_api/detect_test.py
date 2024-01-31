import requests
import unittest
import importlib
utils = importlib.import_module(
    'extensions.sd-webui-controlnet.tests.utils', 'utils')


class TestDetectEndpointWorking(unittest.TestCase):
    def setUp(self):
        self.base_detect_args = {
            "controlnet_module": "canny",
            "controlnet_input_images": [utils.readImage("test/test_files/img2img_basic.png")],
            "controlnet_processor_res": 512,
            "controlnet_threshold_a": 0,
            "controlnet_threshold_b": 0,
        }

    def test_detect_with_invalid_module_performed(self):
        detect_args = self.base_detect_args.copy()
        detect_args.update({
            "controlnet_module": "INVALID",
        })
        self.assertEqual(utils.detect(detect_args).status_code, 422)

    def test_detect_with_no_input_images_performed(self):
        detect_args = self.base_detect_args.copy()
        detect_args.update({
            "controlnet_input_images": [],
        })
        self.assertEqual(utils.detect(detect_args).status_code, 422)

    def test_detect_with_valid_args_performed(self):
        detect_args = self.base_detect_args
        response = utils.detect(detect_args)

        self.assertEqual(response.status_code, 200)
        
    def test_detect_invert(self):
        detect_args = self.base_detect_args.copy()
        detect_args["controlnet_module"] = "invert"
        response = utils.detect(detect_args)
        self.assertEqual(response.status_code, 200)
        self.assertNotEqual(response.json()['images'], [""])


if __name__ == "__main__":
    unittest.main()
