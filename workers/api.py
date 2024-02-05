from datetime import datetime
import urllib.request
import base64
import json
import time
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--port", default=7860, help="Input file path")
args = parser.parse_args()

webui_server_url = f"http://127.0.0.1:{args.port}"

image_file = "./tunong.jpg"
prompt = "<lora:ip-adapter-faceid-plus_sd15_lora:0.7>, <lora:blindbox_v1_mix:0.7>, (masterpiece), (best quality), (ultra-detailed), (full body:1.25), 1boy, chibi, toy figurine, spider man, (beautiful detailed face), (beautiful detailed eyes), standing straight, gym rat, gym background"
negative_prompt = "(low quality:1.3), (worst quality:1.3)"

out_dir = "api_out"
out_dir_t2i = os.path.join(out_dir, "txt2img")
out_dir_i2i = os.path.join(out_dir, "img2img")

os.makedirs(out_dir_t2i, exist_ok=True)
os.makedirs(out_dir_i2i, exist_ok=True)


def timestamp():
    return datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")


def encode_file_to_base64(path):
    with open(path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


def decode_and_save_base64(base64_str, save_path):
    with open(save_path, "wb") as file:
        file.write(base64.b64decode(base64_str))


def call_api(webui_server_url, api_endpoint, **payload):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{webui_server_url}/{api_endpoint}",
        headers={"Content-Type": "application/json"},
        data=data,
    )
    response = urllib.request.urlopen(request)
    return json.loads(response.read().decode("utf-8"))


def call_txt2img_api(webui_server_url, **payload):
    response = call_api(webui_server_url, "sdapi/v1/txt2img", **payload)
    for index, image in enumerate(response.get("images")):
        save_path = os.path.join(out_dir_t2i, f"txt2img-{timestamp()}-{index}.png")
        decode_and_save_base64(image, save_path)


def call_img2img_api(**payload):
    response = call_api("sdapi/v1/img2img", **payload)
    for index, image in enumerate(response.get("images")):
        save_path = os.path.join(out_dir_i2i, f"img2img-{timestamp()}-{index}.png")
        decode_and_save_base64(image, save_path)


def payloader(
    image_file,
    prompt,
    negative_prompt,
    batch_size=1,
    steps=20,
    n_iter=1,
    seed=-1,
    sd_model_checkpoint="revAnimated_v122EOL.safetensors",
    module="ip-adapter_face_id_plus",
    model="ip-adapter-faceid-plus_sd15 [d86a490f]",
    control_mode="Balanced",
    sampler_name="DPM++ 2M Karras",
    resize_mode="Crop and Resize",
    width=512,
    height=512,
    cfg_scale=7,
):
    """
    This function takes in parameters and return prompt for the server.
    """
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "steps": steps,
        "width": width,
        "height": height,
        "cfg_scale": cfg_scale,
        "sampler_name": sampler_name,
        "n_iter": n_iter,
        "batch_size": batch_size,
        # "init_images": init_images,
        # example args for Refiner and ControlNet
        "alwayson_scripts": {
            "ControlNet": {
                "args": [
                    {
                        "batch_images": "",
                        "control_mode": control_mode,
                        "enabled": True,
                        "guidance_end": 1,
                        "guidance_start": 0,
                        "image": {
                            "image": encode_file_to_base64(image_file),
                            "mask": None,  # base64
                        },
                        "input_mode": "simple",
                        "is_ui": True,
                        "loopback": False,
                        "model": model,
                        "module": module,
                        "output_dir": "",
                        "pixel_perfect": False,
                        "processor_res": 512,
                        "resize_mode": resize_mode,
                        "threshold_a": 100,
                        "threshold_b": 200,
                        "weight": 1,
                    }
                ]
            },
        },
        "override_settings": {
            "sd_model_checkpoint": sd_model_checkpoint,
        },
    }
    return payload


if __name__ == "__main__":
    init_images = [
        encode_file_to_base64(image_file),
    ]

    prompt = "<lora:ip-adapter-faceid-plus_sd15_lora:0.7>, <lora:blindbox_v1_mix:0.7>, (masterpiece), (best quality), (ultra-detailed), (full body:1.25), 1boy, chibi, toy figurine, spider man, (beautiful detailed face), (beautiful detailed eyes), standing straight, gym rat, gym background"
    negative_prompt = "(low quality:1.3), (worst quality:1.3)"
    payload = payloader(
        image_file=image_file,
        prompt=prompt,
        negative_prompt=negative_prompt,
        batch_size=4,
    )

    reponse = call_txt2img_api(webui_server_url, **payload)
