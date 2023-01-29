import os
import json
import requests
import io
import base64
from PIL import Image, PngImagePlugin


def call_depth2img_endpoint(source_image, depth_image, url, prompt, output_path):
    assert os.path.exists(source_image)
    assert os.path.exists(depth_image)

    # Open the image file
    with open(source_image, "rb") as image_file:
        img_encoded_string = base64.b64encode(image_file.read())
    print(f"Source image b64: {img_encoded_string[:20]}...{img_encoded_string[-10:]}")

    # Open the depth file
    with open(depth_image, "rb") as image_file:
        depth_encoded_string = base64.b64encode(image_file.read())
    print(f"Depth image b64: {depth_encoded_string[:20]}...{depth_encoded_string[-10:]}")


    payload = {
        "init_images": [
            img_encoded_string.decode()
        ],
        "depth_images": [
            depth_encoded_string.decode()
        ],
        "prompt": prompt,
        "steps": 5
    }

    response = requests.post(url=f'{url}/sdapi/v1/depth2img', json=payload)

    r = response.json()

    for i in r['images']:
        image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))

        png_payload = {
            "image": "data:image/png;base64," + i
        }
        response2 = requests.post(url=f'{url}/sdapi/v1/png-info', json=png_payload)

        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("parameters", response2.json().get("info"))
        image.save(output_path, pnginfo=pnginfo)

        print(f"Done. Output saved to {output_path}")
    return

if __name__ == "__main__":
    url = "http://127.0.0.1:7860"
    source_image = "test/test_files/nike.jpeg"
    depth_image = "test/test_files/nike_depth.jpeg"
    prompt = "Original Nike Air Jordan 4 sneaker, ((orange and black)), product photograph, ultra realistic, professional product photography, photograph, center camera angle, vray render, 4K, 8K, product lighting, extremely detailed, unreal engine, artstation"
    output_path = "out.png"
    call_depth2img_endpoint(source_image, depth_image, url, prompt, output_path)
