from datetime import datetime
import urllib.request
import base64
import json
import time
import os

webui_server_url = 'http://127.0.0.1:7860'

image_file = "./IMG_9887.jpg"
prompt = "<lora:ip-adapter-faceid-plus_sd15_lora:0.7>, <lora:blindbox_v1_mix:0.7>, (masterpiece), (best quality), (ultra-detailed), (full body:1.25), 1boy, chibi, toy figurine, spider man, (beautiful detailed face), (beautiful detailed eyes), standing straight, gym rat, gym background"
negative_prompt = "(low quality:1.3), (worst quality:1.3)"

out_dir = "api_out"
out_dir_t2i = os.path.join(out_dir, 'txt2img')
out_dir_i2i = os.path.join(out_dir, 'img2img')

os.makedirs(out_dir_t2i, exist_ok=True)
os.makedirs(out_dir_i2i, exist_ok=True)


def timestamp():
    return datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")


def encode_file_to_base64(path):
    with open(path, 'rb') as file:
        return base64.b64encode(file.read()).decode('utf-8')


def decode_and_save_base64(base64_str, save_path):
    with open(save_path, "wb") as file:
        file.write(base64.b64decode(base64_str))


def call_api(api_endpoint, **payload):
    data = json.dumps(payload).encode('utf-8')
    request = urllib.request.Request(
        f'{webui_server_url}/{api_endpoint}',
        headers={'Content-Type': 'application/json'},
        data=data,
    )
    response = urllib.request.urlopen(request)
    return json.loads(response.read().decode('utf-8'))


def call_txt2img_api(**payload):
    response = call_api('sdapi/v1/txt2img', **payload)
    for index, image in enumerate(response.get('images')):
        save_path = os.path.join(out_dir_t2i, f'txt2img-{timestamp()}-{index}.png')
        decode_and_save_base64(image, save_path)


def call_img2img_api(**payload):
    response = call_api('sdapi/v1/img2img', **payload)
    for index, image in enumerate(response.get('images')):
        save_path = os.path.join(out_dir_i2i, f'img2img-{timestamp()}-{index}.png')
        decode_and_save_base64(image, save_path)


if __name__ == '__main__':
    init_images = [
        encode_file_to_base64(image_file),
    ]

    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": -1,
        "steps": 20,
        "width": 512,
        "height": 512,
        "cfg_scale": 7,
        "sampler_name": "DPM++ 2M Karras",
        "n_iter": 1,
        "batch_size": 1,
        # "init_images": init_images,
        # example args for Refiner and ControlNet
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
                            "image": encode_file_to_base64(image_file),
                            "mask": None  # base64, None when not need
                        },
                        "input_mode": "simple",
                        "is_ui": True,
                        "loopback": False,
                        "model": "ip-adapter-faceid-plus_sd15 [d86a490f]",
                        "module": "ip-adapter_face_id_plus",
                        "output_dir": "",
                        "pixel_perfect": False,
                        "processor_res": 512,
                        "resize_mode": "Crop and Resize",
                        "threshold_a": 100,
                        "threshold_b": 200,
                        "weight": 1
                    }
                ]
            },
        },
        "override_settings": {
            'sd_model_checkpoint': "revAnimated_v122EOL.safetensors",  # this can use to switch sd model
        },
    }
    call_txt2img_api(**payload)
