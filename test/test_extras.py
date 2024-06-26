import requests


def test_simple_upscaling_performed(base_url, img2img_basic_image_base64):
    payload = {
        "resize_mode": 0,
        "show_extras_results": True,
        "gfpgan_visibility": 0,
        "codeformer_visibility": 0,
        "codeformer_weight": 0,
        "upscaling_resize": 2,
        "upscaling_resize_w": 128,
        "upscaling_resize_h": 128,
        "upscaling_crop": True,
        "upscaler_1": "Lanczos",
        "upscaler_2": "None",
        "extras_upscaler_2_visibility": 0,
        "image": img2img_basic_image_base64,
    }
    assert requests.post(f"{base_url}/sdapi/v1/extra-single-image", json=payload).status_code == 200


def test_png_info_performed(base_url, img2img_basic_image_base64):
    payload = {
        "image": img2img_basic_image_base64,
    }
    assert requests.post(f"{base_url}/sdapi/v1/extra-single-image", json=payload).status_code == 200


def test_interrogate_performed(base_url, img2img_basic_image_base64):
    payload = {
        "image": img2img_basic_image_base64,
        "model": "clip",
    }
    assert requests.post(f"{base_url}/sdapi/v1/extra-single-image", json=payload).status_code == 200
