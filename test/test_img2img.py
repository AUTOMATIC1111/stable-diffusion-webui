
import pytest
import requests


@pytest.fixture()
def url_img2img(base_url):
    return f"{base_url}/sdapi/v1/img2img"


@pytest.fixture()
def simple_img2img_request(img2img_basic_image_base64):
    return {
        "batch_size": 1,
        "cfg_scale": 7,
        "denoising_strength": 0.75,
        "eta": 0,
        "height": 64,
        "include_init_images": False,
        "init_images": [img2img_basic_image_base64],
        "inpaint_full_res": False,
        "inpaint_full_res_padding": 0,
        "inpainting_fill": 0,
        "inpainting_mask_invert": False,
        "mask": None,
        "mask_blur": 4,
        "n_iter": 1,
        "negative_prompt": "",
        "override_settings": {},
        "prompt": "example prompt",
        "resize_mode": 0,
        "restore_faces": False,
        "s_churn": 0,
        "s_noise": 1,
        "s_tmax": 0,
        "s_tmin": 0,
        "sampler_index": "Euler a",
        "seed": -1,
        "seed_resize_from_h": -1,
        "seed_resize_from_w": -1,
        "steps": 3,
        "styles": [],
        "subseed": -1,
        "subseed_strength": 0,
        "tiling": False,
        "width": 64,
    }


def test_img2img_simple_performed(url_img2img, simple_img2img_request):
    assert requests.post(url_img2img, json=simple_img2img_request).status_code == 200


def test_inpainting_masked_performed(url_img2img, simple_img2img_request, mask_basic_image_base64):
    simple_img2img_request["mask"] = mask_basic_image_base64
    assert requests.post(url_img2img, json=simple_img2img_request).status_code == 200


def test_inpainting_with_inverted_masked_performed(url_img2img, simple_img2img_request, mask_basic_image_base64):
    simple_img2img_request["mask"] = mask_basic_image_base64
    simple_img2img_request["inpainting_mask_invert"] = True
    assert requests.post(url_img2img, json=simple_img2img_request).status_code == 200


def test_img2img_sd_upscale_performed(url_img2img, simple_img2img_request):
    simple_img2img_request["script_name"] = "sd upscale"
    simple_img2img_request["script_args"] = ["", 8, "Lanczos", 2.0]
    assert requests.post(url_img2img, json=simple_img2img_request).status_code == 200
