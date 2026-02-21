#!/usr/bin/env python3
"""
Stable Diffusion WebUI Automation Script

This script provides programmatic access to the Stable Diffusion WebUI API
for automated image generation without needing the GUI.

Usage:
    python automate_sd.py --prompt "a beautiful landscape" --output output.png
    python automate_sd.py --prompt "portrait of a person" --steps 50 --cfg 7.5
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import requests


class StableDiffusionAPI:
    """Python wrapper for Stable Diffusion WebUI API"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:7860"):
        self.base_url = base_url
        self.api_endpoint = f"{base_url}/sdapi/v1"
        
    def wait_for_api(self, timeout: int = 60) -> bool:
        """Wait for the WebUI API to be ready"""
        print(f"Waiting for WebUI at {self.base_url}...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.api_endpoint}/options", timeout=5)
                if response.status_code == 200:
                    print("WebUI API is ready!")
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(2)
        print("Timeout waiting for WebUI")
        return False
    
    def txt2img(
        self,
        prompt: str,
        negative_prompt: str = "",
        steps: int = 20,
        cfg_scale: float = 7.0,
        width: int = 512,
        height: int = 512,
        seed: int = -1,
        batch_size: int = 1,
        sampler_name: str = "Euler a"
    ) -> list:
        """
        Generate images from text prompt
        
        Args:
            prompt: Positive prompt describing what to generate
            negative_prompt: What to avoid in the image
            steps: Number of denoising steps (higher = better quality, slower)
            cfg_scale: Classifier free guidance scale (7 is good default)
            width: Image width
            height: Image height
            seed: Random seed (-1 for random)
            batch_size: Number of images to generate
            sampler_name: Sampling method
            
        Returns:
            List of base64 encoded images
        """
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "width": width,
            "height": height,
            "seed": seed,
            "batch_size": batch_size,
            "sampler_name": sampler_name,
        }
        
        print(f"Generating: '{prompt}' (steps={steps}, cfg={cfg_scale})")
        response = requests.post(
            f"{self.api_endpoint}/txt2img",
            json=payload,
            timeout=300  # 5 minute timeout for generation
        )
        
        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code} - {response.text}")
            
        result = response.json()
        return result.get("images", [])
    
    def img2img(
        self,
        prompt: str,
        image_path: str,
        negative_prompt: str = "",
        steps: int = 20,
        cfg_scale: float = 7.0,
        denoising_strength: float = 0.75,
        seed: int = -1,
    ) -> list:
        """Generate images from image + text prompt"""
        # Read and encode image
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "init_images": [image_b64],
            "steps": steps,
            "cfg_scale": cfg_scale,
            "denoising_strength": denoising_strength,
            "seed": seed,
        }
        
        print(f"Img2Img: '{prompt}' from {image_path}")
        response = requests.post(
            f"{self.api_endpoint}/img2img",
            json=payload,
            timeout=300
        )
        
        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code}")
            
        result = response.json()
        return result.get("images", [])
    
    def save_image(self, base64_data: str, output_path: str) -> str:
        """Save base64 image to file"""
        image_bytes = base64.b64decode(base64_data)
        with open(output_path, "wb") as f:
            f.write(image_bytes)
        return output_path
    
    def get_models(self) -> list:
        """Get available models"""
        response = requests.get(f"{self.api_endpoint}/sd-models")
        return response.json()
    
    def get_samplers(self) -> list:
        """Get available samplers"""
        response = requests.get(f"{self.api_endpoint}/samplers")
        return response.json()


def generate_batch(prompts: list, output_dir: str = "outputs", **kwargs):
    """Generate multiple images from a list of prompts"""
    os.makedirs(output_dir, exist_ok=True)
    api = StableDiffusionAPI()
    
    if not api.wait_for_api():
        print("Failed to connect to WebUI")
        return
    
    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] {prompt}")
        try:
            images = api.txt2img(prompt, **kwargs)
            for j, img_data in enumerate(images):
                output_path = os.path.join(output_dir, f"gen_{i}_{j}.png")
                api.save_image(img_data, output_path)
                print(f"  Saved: {output_path}")
        except Exception as e:
            print(f"  Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Stable Diffusion WebUI Automation")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--negative", type=str, default="", help="Negative prompt")
    parser.add_argument("--output", type=str, default="output.png", help="Output file")
    parser.add_argument("--steps", type=int, default=20, help="Number of steps")
    parser.add_argument("--cfg", type=float, default=7.0, help="CFG scale")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--seed", type=int, default=-1, help="Seed (-1 for random)")
    parser.add_argument("--sampler", type=str, default="Euler a", help="Sampler name")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:7860", help="WebUI URL")
    parser.add_argument("--img2img", type=str, help="Input image for img2img")
    parser.add_argument("--denoise", type=float, default=0.75, help="Denoising strength for img2img")
    
    args = parser.parse_args()
    
    api = StableDiffusionAPI(args.url)
    
    if not api.wait_for_api():
        print("Error: Could not connect to WebUI. Is it running?")
        sys.exit(1)
    
    try:
        if args.img2img:
            images = api.img2img(
                prompt=args.prompt,
                image_path=args.img2img,
                negative_prompt=args.negative,
                steps=args.steps,
                cfg_scale=args.cfg,
                denoising_strength=args.denoise,
                seed=args.seed
            )
        else:
            images = api.txt2img(
                prompt=args.prompt,
                negative_prompt=args.negative,
                steps=args.steps,
                cfg_scale=args.cfg,
                width=args.width,
                height=args.height,
                seed=args.seed,
                sampler_name=args.sampler
            )
        
        for i, img_data in enumerate(images):
            output_path = args.output if i == 0 else args.output.replace(".png", f"_{i}.png")
            api.save_image(img_data, output_path)
            print(f"Saved: {output_path}")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
