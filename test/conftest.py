import base64
import os
import socket

import pytest

test_files_path = os.path.dirname(__file__) + "/test_files"
test_outputs_path = os.path.dirname(__file__) + "/test_outputs"
api_integration_test_files = {
    "test_extras.py",
    "test_img2img.py",
    "test_txt2img.py",
    "test_utils.py",
}
_server_available = None


def pytest_configure(config):
    # We don't want to fail on Py.test command line arguments being
    # parsed by webui:
    os.environ.setdefault("IGNORE_CMD_ARGS_ERRORS", "1")


def webui_server_available():
    global _server_available

    if _server_available is not None:
        return _server_available

    host = "127.0.0.1"
    port = 7860
    server_available = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_available.settimeout(0.25)
    try:
        server_available.connect((host, port))
        _server_available = True
    except OSError:
        _server_available = False
    finally:
        server_available.close()

    return _server_available


def pytest_ignore_collect(path, config):
    return path.basename in api_integration_test_files and not webui_server_available()


def pytest_collection_modifyitems(config, items):
    skip_api = None
    if not webui_server_available():
        skip_api = pytest.mark.skip(reason="WebUI server is required for API integration tests")

    if skip_api is None:
        return

    for item in items:
        if item.path.name in api_integration_test_files:
            item.add_marker(skip_api)


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


@pytest.fixture(scope="session")
def initialize() -> None:
    import webui  # noqa: F401
