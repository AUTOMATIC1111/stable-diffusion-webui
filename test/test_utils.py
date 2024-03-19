import pytest
import requests


def test_options_write(base_url):
    url_options = f"{base_url}/sdapi/v1/options"
    response = requests.get(url_options)
    assert response.status_code == 200

    pre_value = response.json()["send_seed"]

    assert requests.post(url_options, json={'send_seed': (not pre_value)}).status_code == 200

    response = requests.get(url_options)
    assert response.status_code == 200
    assert response.json()['send_seed'] == (not pre_value)

    requests.post(url_options, json={"send_seed": pre_value})


@pytest.mark.parametrize("url", [
    "sdapi/v1/cmd-flags",
    "sdapi/v1/samplers",
    "sdapi/v1/upscalers",
    "sdapi/v1/sd-models",
    "sdapi/v1/hypernetworks",
    "sdapi/v1/face-restorers",
    "sdapi/v1/realesrgan-models",
    "sdapi/v1/prompt-styles",
    "sdapi/v1/embeddings",
])
def test_get_api_url(base_url, url):
    assert requests.get(f"{base_url}/{url}").status_code == 200
