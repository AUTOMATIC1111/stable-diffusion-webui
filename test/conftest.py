import os

import pytest
import base64


test_files_path = os.path.dirname(__file__) + "/test_files"


def file_to_base64(filename):
    with open(filename, "rb") as file:
        data = file.read()

    base64_str = str(base64.b64encode(data), "utf-8")
    return "data:image/png;base64," + base64_str


@pytest.fixture(scope="session")  # session so we don't read this over and over
def img2img_basic_image_base64() -> str:
    return file_to_base64(os.path.join(test_files_path, "img2img_basic.png"))


@pytest.fixture(scope="session")  # session so we don't read this over and over
def mask_basic_image_base64() -> str:
    return file_to_base64(os.path.join(test_files_path, "mask_basic.png"))
