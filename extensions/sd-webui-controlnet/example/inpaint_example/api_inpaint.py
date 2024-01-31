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


def generate(url: str, payload: dict):
    response = requests.post(url=url, json=payload).json()
    if "images" not in response:
        print(response)
    else:
        for i, base64image in enumerate(response["images"]):
            Image.open(io.BytesIO(base64.b64decode(base64image.split(",", 1)[0]))).save(
                f"{url.split('/')[-1]}-{i}.png"
            )


def read_image(img_path: str) -> str:
    img = cv2.imread(img_path)
    _, bytes = cv2.imencode(".png", img)
    encoded_image = base64.b64encode(bytes).decode("utf-8")
    return encoded_image


input_image = read_image("1girl.png")
mask_image = read_image("mask.png")

img2img_payload = {
    "batch_size": 1,
    "cfg_scale": 7,
    "height": 768,
    "width": 512,
    "n_iter": 1,
    "steps": 30,
    "sampler_name": "DPM++ 2M Karras",
    "prompt": "(masterpiece: 1.3), (highres: 1.3), best quality,",
    "negative_prompt": "(worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, backlight,(ugly:1.331), (duplicate:1.331), (morbid:1.21), (mutilated:1.21), (tranny:1.331), mutated hands, (poorly drawn hands:1.331), blurry, (bad anatomy:1.21), (bad proportions:1.331), extra limbs, (more than 2 nipples:1.331), (missing arms:1.331), (extra legs:1.331), (fused fingers:1.61051), (too many fingers:1.61051), (unclear eyes:1.331), bad hands, missing fingers, extra digit, (futa:1.1), bad body, pubic hair, glans, easynegative,more than 2 tits, ng_deepnegative_v1_75t,(big fee:1),more than 2 feet,incorrect feet",
    "seed": 42,
    "seed_enable_extras": False,
    "seed_resize_from_h": 0,
    "seed_resize_from_w": 0,
    "subseed": -1,
    "subseed_strength": 0,
    "override_settings": {},
    "override_settings_restore_afterwards": False,
    "do_not_save_grid": False,
    "do_not_save_samples": False,
    "s_churn": 0,
    "s_min_uncond": 0,
    "s_noise": 1,
    "s_tmax": None,
    "s_tmin": 0,
    "script_args": [],
    "script_name": None,
    "styles": [],
    "alwayson_scripts": {
        "ControlNet": {
            "args": [
                {
                    "control_mode": 0,
                    "enabled": True,
                    "guidance_end": 1,
                    "guidance_start": 0,
                    "low_vram": False,
                    "model": "control_v11p_sd15_inpaint [ebff9138]",
                    "module": "inpaint_only",
                    "pixel_perfect": True,
                    "processor_res": 512,
                    "resize_mode": 1,
                    "threshold_a": 64,
                    "threshold_b": 64,
                    "weight": 1,
                }
            ]
        }
    },
    "denoising_strength": 0.75,
    "initial_noise_multiplier": 1,
    "inpaint_full_res": 0,
    "inpaint_full_res_padding": 32,
    "inpainting_fill": 1,
    "inpainting_mask_invert": 0,
    "mask_blur_x": 4,
    "mask_blur_y": 4,
    "mask_blur": 4,
    "resize_mode": 0,
    "init_images": [input_image],
    "mask": mask_image,
}

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
                    "image": {
                        "image": input_image,
                        "mask": mask_image,
                    },
                    "low_vram": False,
                    "model": "control_v11p_sd15_inpaint [ebff9138]",
                    "module": "inpaint_only",
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
    "height": 768,
    "hr_negative_prompt": "",
    "hr_prompt": "",
    "hr_resize_x": 0,
    "hr_resize_y": 0,
    "hr_scale": 2,
    "hr_second_pass_steps": 0,
    "hr_upscaler": "Latent",
    "n_iter": 1,
    "negative_prompt": "(worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, backlight,(ugly:1.331), (duplicate:1.331), (morbid:1.21), (mutilated:1.21), (tranny:1.331), mutated hands, (poorly drawn hands:1.331), blurry, (bad anatomy:1.21), (bad proportions:1.331), extra limbs, (more than 2 nipples:1.331), (missing arms:1.331), (extra legs:1.331), (fused fingers:1.61051), (too many fingers:1.61051), (unclear eyes:1.331), bad hands, missing fingers, extra digit, (futa:1.1), bad body, pubic hair, glans, easynegative,more than 2 tits, ng_deepnegative_v1_75t,(big fee:1),more than 2 feet,incorrect feet",
    "override_settings": {},
    "override_settings_restore_afterwards": True,
    "prompt": "(masterpiece: 1.3), (highres: 1.3), best quality,",
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
    "width": 512,
}


if __name__ == "__main__":
    url = "http://localhost:7860/sdapi/v1/"
    generate(url + "img2img", img2img_payload)
    generate(url + "txt2img", txt2img_payload)
