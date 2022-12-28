import requests
import base64
import io
from PIL import Image
import time

payload = {
    'enable_hr': False, 
    'denoising_strength': 0, 
    'firstphase_width': 0, 
    'firstphase_height': 0, 
    'prompt': 'a cute girl', 
    'styles': None, 
    'seed': -1, 
    'subseed': -1, 
    'subseed_strength': 0, 
    'seed_resize_from_h': -1, 
    'seed_resize_from_w': -1, 
    'sampler_name': None, 
    'batch_size': 4, 
    'n_iter': 1, 
    'steps': 15, 
    'cfg_scale': 17.0, 
    'width': 512, 
    'height': 512, 
    'restore_faces': False, 
    'tiling': False, 
    'negative_prompt': None, 
    'eta': None, 
    's_churn': 0.0, 
    's_tmax': None, 
    's_tmin': 0.0, 
    's_noise': 1.0, 
    'override_settings': None, 
    'sampler_index': 'Euler' 
}

start = time.time()

def query_api(payload):
    response = requests.post(url=f'https://ac9c0ab35af08308.gradio.app/sdapi/v1/txt2img', json=payload)    
    return response.json()

for i in range(10):
    respose_json = query_api(payload)

print(f"Time taken: {time.time() - start}")    

for index, i in enumerate(respose_json['images']):
    image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))
    image.save(f'test_api_{index}.png')