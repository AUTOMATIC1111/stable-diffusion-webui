from pathlib import Path
import os
import sys
from lollms.paths import LollmsPaths
from lollms.config import TypedConfig, ConfigTemplate, BaseConfig
import time
import sys
import requests
import os
import base64
import subprocess
import time
import platform

class SD:
    def __init__(self, lollms_path:LollmsPaths, personality_config: TypedConfig, wm = "Artbot", max_retries=30):
        # Get the current directory
        root_dir = lollms_path.personal_path
        current_dir = Path(__file__).resolve().parent
        self.wm = wm

        # Store the path to the script
        self.auto_sd_url = "http://127.0.0.1:7860"
        shared_folder = root_dir/"shared"
        self.sd_folder = shared_folder / "auto_sd"
        self.output_dir = root_dir / "outputs/sd"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.wait_for_service(1):
            # Launch the Flask service using the appropriate script for the platform
            if platform.system() == "Windows":
                script_path = self.sd_folder / "lollms_webui.bat"
            else:
                script_path = self.sd_folder / "lollms_webui.sh"

            subprocess.Popen(script_path, cwd=self.sd_folder)

        # Wait until the service is available at http://127.0.0.1:7860/
        self.wait_for_service(max_retries=max_retries)

    def wait_for_service(self, max_retries = 30):
        url = f"{self.auto_sd_url}/internal/ping"
        # Adjust this value as needed
        retries = 0

        while retries < max_retries or max_retries<0:
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    print("Service is available.")
                    return True
            except requests.exceptions.RequestException:
                pass

            retries += 1
            time.sleep(1)

        print("Service did not become available within the given time.")
        return False
    
    def get_available_image_name(self, save_folder, base_name):
        index = 0
        while True:
            image_name = f"{base_name}_{index}.png"
            image_path = os.path.join(save_folder, image_name)
            if not os.path.exists(image_path):
                return image_name
            index += 1        

    def txt_to_img(
                    self, 
                    positive_prompt,
                    negative_prompt="",
                    denoising_strength=0,
                    seed=-1,
                    cfg_scale=7,
                    steps=50,
                    width=512,
                    height=512,
                    tiling=False,
                    sampler_name="Euler", 
                    upscaler_name="", 
                    styles=None, 
                    restore_faces=False, 
                    save_folder=None, 
                    script_name=""
                    ):
        url = f"{self.auto_sd_url}/sdapi/v1/txt2img"
        
        if styles is None:
            styles = []
            
        if save_folder is None:
            save_folder = self.output_dir

        data = {
            "enable_hr": False,
            "denoising_strength": denoising_strength,
            "firstphase_width": 0,
            "firstphase_height": 0,
            "hr_scale": 2,
            "hr_upscaler": upscaler_name,
            "hr_second_pass_steps": 0,
            "hr_resize_x": 0,
            "hr_resize_y": 0,
            "hr_sampler_name": sampler_name,
            "hr_prompt": "",
            "hr_negative_prompt": "",
            "prompt": positive_prompt,
            "styles": styles,
            "seed": seed,
            "subseed": -1,
            "subseed_strength": 0,
            "seed_resize_from_h": -1,
            "seed_resize_from_w": -1,
            "sampler_name": sampler_name,
            "batch_size": 1,
            "n_iter": 1,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "width": width,
            "height": height,
            "restore_faces": restore_faces,
            "tiling": tiling,
            "do_not_save_samples": False,
            "do_not_save_grid": True,
            "negative_prompt": negative_prompt,
            "eta": 0,
            "s_min_uncond": 0,
            "s_churn": 0,
            "s_tmax": 0,
            "s_tmin": 0,
            "s_noise": 1,
            "override_settings": {},
            "override_settings_restore_afterwards": True,
            "script_args": [],
            "sampler_index": sampler_name,
            "script_name": script_name,
            "send_images": True,
            "save_images": False,
            "alwayson_scripts": {}
        }

        headers = {
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(url, json=data, headers=headers)

            # Check if the request was successful (status code 200)
            if response.status_code == 200:
                response_data = response.json()
                if save_folder:
                    image_paths = []
                    for i, image_base64 in enumerate(response_data['images']):
                        image_name = self.get_available_image_name(save_folder, self.wm)
                        image_path = os.path.join(save_folder, image_name)
                        image_data = base64.b64decode(image_base64)
                        with open(image_path, 'wb') as f:
                            f.write(image_data)
                        image_paths.append(image_path)
                    response_data['image_paths'] = image_paths
                return response_data
            else:
                # If the request was not successful, print the status code and response text
                print(f"Error: {response.status_code} - {response.text}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            return None

    def img_to_img(self, 
                   init_images, 
                   prompt="", 
                   sampler_name="Euler",
                   seed=-1,
                   cfg_scale=7,
                   steps=50,
                   width=512,
                   height=512,
                   tiling=False,
                   restore_faces=False,
                   styles=None, 
                   save_folder=None, 
                   script_name=""
                ):
        url = f"{self.auto_sd_url}/sdapi/v1/img2img"

        if styles is None:
            styles = []

        data = {
            "init_images": init_images,
            "resize_mode": 0,
            "denoising_strength": 0.75,
            "image_cfg_scale": 0,
            "mask": "string",
            "mask_blur": 0,
            "mask_blur_x": 4,
            "mask_blur_y": 4,
            "inpainting_fill": 0,
            "inpaint_full_res": True,
            "inpaint_full_res_padding": 0,
            "inpainting_mask_invert": 0,
            "initial_noise_multiplier": 0,
            "prompt": prompt,
            "styles": styles,
            "seed": seed,
            "subseed": -1,
            "subseed_strength": 0,
            "seed_resize_from_h": -1,
            "seed_resize_from_w": -1,
            "sampler_name": sampler_name,
            "batch_size": 1,
            "n_iter": 1,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "width": width,
            "height": height,
            "restore_faces": restore_faces,
            "tiling": tiling,
            "do_not_save_samples": False,
            "do_not_save_grid": True,
            "negative_prompt": "string",
            "eta": 0,
            "s_min_uncond": 0,
            "s_churn": 0,
            "s_tmax": 0,
            "s_tmin": 0,
            "s_noise": 1,
            "override_settings": {},
            "override_settings_restore_afterwards": True,
            "script_args": [],
            "sampler_index": "Euler",
            "include_init_images": False,
            "script_name": script_name,
            "send_images": True,
            "save_images": False,
            "alwayson_scripts": {}
        }

        headers = {
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(url, json=data, headers=headers)

            # Check if the request was successful (status code 200)
            if response.status_code == 200:
                response_data = response.json()
                if save_folder:
                    image_paths = []
                    for i, image_base64 in enumerate(response_data['images']):
                        image_name = self.get_available_image_name(save_folder, self.wm)
                        image_path = os.path.join(save_folder, image_name)
                        image_data = base64.b64decode(image_base64)
                        with open(image_path, 'wb') as f:
                            f.write(image_data)
                        image_paths.append(image_path)
                    response_data['image_paths'] = image_paths
                return response_data
            else:
                # If the request was not successful, print the status code and response text
                print(f"Error: {response.status_code} - {response.text}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            return None
