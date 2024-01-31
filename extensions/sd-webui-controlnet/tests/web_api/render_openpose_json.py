import requests
import unittest
import importlib
import json
from pathlib import Path

utils = importlib.import_module("extensions.sd-webui-controlnet.tests.utils", "utils")


def render(poses):
    return requests.post(
        utils.BASE_URL + "/controlnet/render_openpose_json", json=poses
    ).json()


with open(Path(__file__).parent / "pose.json", "r") as f:
    pose = json.load(f)


with open(Path(__file__).parent / "animal_pose.json", "r") as f:
    animal_pose = json.load(f)


class TestDetectEndpointWorking(unittest.TestCase):
    def test_render_single(self):
        res = render([pose])
        self.assertEqual(res["info"], "Success")
        self.assertEqual(len(res["images"]), 1)

    def test_render_multiple(self):
        res = render([pose, pose])
        self.assertEqual(res["info"], "Success")
        self.assertEqual(len(res["images"]), 2)

    def test_render_no_pose(self):
        res = render([])
        self.assertNotEqual(res["info"], "Success")

    def test_render_invalid_pose(self):
        res = render([{"foo": 10, "bar": 100}])
        self.assertNotIn("info", res)
        self.assertNotIn("images", res)

    def test_render_animals(self):
        res = render([animal_pose])
        self.assertEqual(res["info"], "Success")
        self.assertEqual(len(res["images"]), 1)


if __name__ == "__main__":
    unittest.main()
