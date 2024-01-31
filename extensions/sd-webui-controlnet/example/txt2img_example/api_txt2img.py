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


class ControlnetRequest:
    def __init__(self, prompt, path):
        self.url = "http://localhost:7860/sdapi/v1/txt2img"
        self.prompt = prompt
        self.img_path = path
        self.body = None

    def build_body(self):
        self.body = {
            "prompt": self.prompt,
            "negative_prompt": "",
            "batch_size": 1,
            "steps": 20,
            "cfg_scale": 7,
            "alwayson_scripts": {
                "controlnet": {
                    "args": [
                        {
                            "enabled": True,
                            "module": "none",
                            "model": "canny",
                            "weight": 1.0,
                            "image": self.read_image(),
                            "resize_mode": 1,
                            "lowvram": False,
                            "processor_res": 64,
                            "threshold_a": 64,
                            "threshold_b": 64,
                            "guidance_start": 0.0,
                            "guidance_end": 1.0,
                            "control_mode": 0,
                            "pixel_perfect": False
                        }
                    ]
                }
            }
        }

    def send_request(self):
        response = requests.post(url=self.url, json=self.body)
        return response.json()

    def read_image(self):
        img = cv2.imread(self.img_path)
        retval, bytes = cv2.imencode('.png', img)
        encoded_image = base64.b64encode(bytes).decode('utf-8')
        return encoded_image


if __name__ == '__main__':
    path = 'stock_mountain.png'
    prompt = 'a large avalanche'

    control_net = ControlnetRequest(prompt, path)
    control_net.build_body()
    output = control_net.send_request()

    result = output['images'][0]

    image = Image.open(io.BytesIO(base64.b64decode(result.split(",", 1)[0])))
    image.show()
