import io
import cv2
import base64
import requests
from PIL import Image

"""
    To use this example make sure you've done the following steps before executing:
    1. Ensure automatic1111 is running in api mode with the controlnet extension. 
       Use the following command in your terminal to activate:
            ./webui.sh --no-half --api
    2. Validate python environment meet package dependencies.
       If running in a local repo you'll likely need to pip install cv2, requests and PIL 
"""


def generate(url: str, payload: dict, file_suffix: str = ""):
    response = requests.post(url=url, json=payload).json()
    if "images" not in response:
        print(response)
    else:
        for i, base64image in enumerate(response["images"]):
            Image.open(io.BytesIO(base64.b64decode(base64image.split(",", 1)[0]))).save(
                f"{url.split('/')[-1]}-{i}{file_suffix}.png"
            )


def read_image(img_path: str) -> str:
    img = cv2.imread(img_path)
    _, bytes = cv2.imencode(".png", img)
    encoded_image = base64.b64encode(bytes).decode("utf-8")
    return encoded_image


input_image = read_image("stock_mountain.png")

txt2img_payload = {
    "alwayson_scripts": {
        "ControlNet": {
            "args": [
                {
                    "batch_images": "",
                    "control_mode": "Balanced",
                    "enabled": True,
                    "guidance_end": 1,
                    "guidance_start": 0,
                    "image": input_image,
                    "low_vram": False,
                    "model": "control_v11p_sd15_canny [d14c016b]",
                    "module": "canny",
                    "pixel_perfect": False,
                    "processor_res": -1,
                    "resize_mode": "Crop and Resize",
                    "save_detected_map": True,
                    "threshold_a": -1,
                    "threshold_b": -1,
                    "weight": 1,
                }
            ]
        }
    },
    "batch_size": 1,
    "cfg_scale": 7,
    "comments": {},
    "disable_extra_networks": False,
    "do_not_save_grid": False,
    "do_not_save_samples": False,
    "enable_hr": False,
    "height": 512,
    "width": 768,
    "hr_negative_prompt": "",
    "hr_prompt": "",
    "hr_resize_x": 0,
    "hr_resize_y": 0,
    "hr_scale": 2,
    "hr_second_pass_steps": 0,
    "hr_upscaler": "Latent",
    "n_iter": 1,
    "negative_prompt": "",
    "override_settings": {},
    "override_settings_restore_afterwards": True,
    "prompt": "(masterpiece: 1.3), (highres: 1.3), best quality, a large avalanche",
    "restore_faces": False,
    "s_churn": 0.0,
    "s_min_uncond": 0,
    "s_noise": 1.0,
    "s_tmax": None,
    "s_tmin": 0.0,
    "sampler_name": "DPM++ 2M Karras",
    "script_args": [],
    "script_name": None,
    "seed": 42,
    "seed_enable_extras": True,
    "seed_resize_from_h": -1,
    "seed_resize_from_w": -1,
    "steps": 30,
    "styles": [],
    "subseed": -1,
    "subseed_strength": 0,
    "tiling": False,
}


if __name__ == "__main__":
    url = "http://localhost:7860/sdapi/v1/"
    for weight_factor in (0.3, 0.5, 0.8):
        advanced_weighting = [weight_factor ** float(12 - i) for i in range(13)]
        txt2img_payload["alwayson_scripts"]["ControlNet"]["args"][0][
            "advanced_weighting"
        ] = advanced_weighting
        generate(url + "txt2img", txt2img_payload, file_suffix=f"fac{weight_factor}")

    for linear_start in (0.3, 0.5, 0.8):
        step = (1.0 - linear_start) / 12
        advanced_weighting = [linear_start + i * step for i in range(13)]
        txt2img_payload["alwayson_scripts"]["ControlNet"]["args"][0][
            "advanced_weighting"
        ] = advanced_weighting
        generate(url + "txt2img", txt2img_payload, file_suffix=f"linear{linear_start}")
