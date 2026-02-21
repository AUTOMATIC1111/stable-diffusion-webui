#!/usr/bin/env python3
"""
Portfolio Image Generation Examples

This script demonstrates how to use the Stable Diffusion automation
to generate images for your portfolio projects.

Usage:
    python3.10 examples/portfolio_examples.py
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from automate_sd import StableDiffusionAPI


def generate_portfolio_hero(api: StableDiffusionAPI, output_dir: str = "outputs/portfolio"):
    """Generate hero images for portfolio projects"""
    os.makedirs(output_dir, exist_ok=True)
    
    prompts = [
        ("futuristic tech dashboard", "hero_tech.png"),
        ("minimalist workspace", "hero_minimal.png"),
        ("creative design studio", "hero_creative.png"),
    ]
    
    for prompt, filename in prompts:
        print(f"\nGenerating: {filename}")
        images = api.txt2img(
            prompt=prompt,
            steps=30,
            width=1024,
            height=576,  # 16:9 aspect ratio
            cfg_scale=8.0
        )
        api.save_image(images[0], os.path.join(output_dir, filename))
        print(f"Saved: {output_dir}/{filename}")


def generate_project_thumbnails(api: StableDiffusionAPI, output_dir: str = "outputs/thumbnails"):
    """Generate project thumbnails"""
    os.makedirs(output_dir, exist_ok=True)
    
    prompts = [
        "app icon mobile design",
        "website landing page mockup",
        "logo design modern minimalist",
        "social media post template",
    ]
    
    for i, prompt in enumerate(prompts):
        print(f"\nGenerating thumbnail {i+1}: {prompt}")
        images = api.txt2img(
            prompt=prompt,
            steps=25,
            width=512,
            height=512,
            cfg_scale=7.5
        )
        api.save_image(images[0], os.path.join(output_dir, f"thumb_{i+1}.png"))


def generate_placeholder_images(api: StableDiffusionAPI, output_dir: str = "outputs/placeholders"):
    """Generate placeholder images for development"""
    os.makedirs(output_dir, exist_ok=True)
    
    placeholders = [
        ("abstract geometric shapes", "abstract_1.png"),
        ("gradient background purple blue", "gradient_bg.png"),
        ("wireframe UI design", "wireframe.png"),
    ]
    
    for prompt, filename in placeholders:
        print(f"\nGenerating placeholder: {filename}")
        images = api.txt2img(
            prompt=prompt,
            steps=15,  # Faster for placeholders
            width=800,
            height=600,
            cfg_scale=6.0
        )
        api.save_image(images[0], os.path.join(output_dir, filename))


def batch_generate(api: StableDiffusionAPI, prompts: list, output_dir: str = "outputs/batch"):
    """Generate multiple images from a list of prompts"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] {prompt}")
        images = api.txt2img(
            prompt=prompt,
            steps=20,
            width=512,
            height=512
        )
        api.save_image(images[0], os.path.join(output_dir, f"gen_{i+1}.png"))


def main():
    print("=" * 60)
    print("Portfolio Image Generator")
    print("=" * 60)
    
    # Connect to WebUI
    api = StableDiffusionAPI("http://127.0.0.1:7860")
    
    # Wait for WebUI to be ready
    if not api.wait_for_api(timeout=30):
        print("Error: Could not connect to WebUI")
        print("Make sure WebUI is running: python3.10 launch.py --xformers --api")
        sys.exit(1)
    
    print("\n✓ Connected to WebUI")
    
    # Example 1: Generate hero images
    print("\n--- Generating Hero Images ---")
    generate_portfolio_hero(api)
    
    # Example 2: Generate thumbnails
    print("\n--- Generating Thumbnails ---")
    generate_project_thumbnails(api)
    
    # Example 3: Custom batch
    print("\n--- Custom Batch Generation ---")
    custom_prompts = [
        "modern office interior",
        "coffee shop aesthetic",
        "mountain landscape at dawn",
        "cyberpunk city night",
    ]
    batch_generate(api, custom_prompts)
    
    print("\n" + "=" * 60)
    print("All images generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
