import os

import pytest
from PIL import Image
from gradio.processing_utils import encode_pil_to_base64

test_files_path = os.path.dirname(__file__) + "/test_files"


@pytest.fixture(scope="session")  # session so we don't read this over and over
def img2img_basic_image_base64() -> str:
    return encode_pil_to_base64(Image.open(os.path.join(test_files_path, "img2img_basic.png")))


@pytest.fixture(scope="session")  # session so we don't read this over and over
def mask_basic_image_base64() -> str:
    return encode_pil_to_base64(Image.open(os.path.join(test_files_path, "mask_basic.png")))
