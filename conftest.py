import os

import pytest
from PIL import Image
from gradio.processing_utils import encode_pil_to_base64

test_files_path = os.path.join(os.path.dirname(__file__), "test", "test_files")


def pytest_configure(config):
    # Importing `modules.shared` attempts to parse command line arguments, which we don't want to do in tests,
    # since command line arguments are py.test's.  Ignore those.
    os.environ.setdefault('IGNORE_CMD_ARGS_ERRORS', '1')
    # We'll import `webui.py` early here to let it load all modules, etc. in the correct order,
    # and to have `modules.paths` set up `sys.path` correctly.
    import webui  # noqa: F401


@pytest.fixture(scope="session")  # session so we don't read this over and over
def img2img_basic_image_base64() -> str:
    return encode_pil_to_base64(Image.open(os.path.join(test_files_path, "img2img_basic.png")))


@pytest.fixture(scope="session")  # session so we don't read this over and over
def mask_basic_image_base64() -> str:
    return encode_pil_to_base64(Image.open(os.path.join(test_files_path, "mask_basic.png")))
