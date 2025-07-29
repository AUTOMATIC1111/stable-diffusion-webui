#!/usr/bin/env python3
"""
Test script to verify Stable Diffusion WebUI API connectivity
This simulates what your Rust tool will do when calling the API
"""

import json
import requests
import time
import base64
from pathlib import Path

def test_api_connection():
    """Test if the API is running and responsive"""
    try:
        response = requests.get("http://localhost:7860/sdapi/v1/options", timeout=5)
        if response.status_code == 200:
            print("✅ API is running and responsive")
            return True
        else:
            print(f"❌ API returned status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to API: {e}")
        return False

def test_food_image_generation():
    """Test generating a food image like your Rust tool will do"""
    
    # This matches the JSON structure your Rust tool sends
    payload = {
        "prompt": "professional food photography of grilled chicken breast, high quality, appetizing presentation, clean white background, studio lighting, commercial food photography style, gourmet, fresh, cooked perfectly, restaurant quality, high resolution, detailed, 4K, professional lighting",
        "negative_prompt": "blurry, low quality, amateur, dirty, unappetizing, dark, shadowy, cluttered background",
        "width": 1024,
        "height": 1024,
        "steps": 30,
        "cfg_scale": 7.5,
        "sampler_name": "DPM++ 2M",
        "batch_size": 1,
        "n_iter": 1,
        "restore_faces": False,
        "tiling": False,
        "do_not_save_samples": True,
        "do_not_save_grid": True
    }
    
    print("🎨 Testing food image generation...")
    print(f"📝 Prompt: {payload['prompt'][:80]}...")
    
    try:
        response = requests.post(
            "http://localhost:7860/sdapi/v1/txt2img",
            json=payload,
            timeout=120  # 2 minutes timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            
            if "images" in result and len(result["images"]) > 0:
                print("✅ Image generation successful!")
                
                # Save the test image
                image_data = base64.b64decode(result["images"][0])
                test_image_path = Path("test_generated_food.png")
                test_image_path.write_bytes(image_data)
                
                print(f"💾 Test image saved as: {test_image_path}")
                print(f"📏 Image size: {len(image_data)} bytes")
                
                # Show some info about the generation
                if "info" in result:
                    info = json.loads(result["info"])
                    print(f"⏱️  Generation time: ~{info.get('time_taken', 'unknown')} seconds")
                
                return True
            else:
                print("❌ No images in response")
                return False
        else:
            print(f"❌ Generation failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False

def main():
    print("🧪 Testing Stable Diffusion WebUI API for Food Image Generation")
    print("=" * 60)
    
    print("\n1️⃣ Testing API connection...")
    if not test_api_connection():
        print("\n💡 Make sure the WebUI is running with: ./start-with-uv.sh")
        return
    
    print("\n2️⃣ Testing food image generation...")
    if test_food_image_generation():
        print("\n🎉 All tests passed! Your Rust tool should work correctly.")
        print("\n🔗 API Documentation: http://localhost:7860/docs")
        print("🌐 WebUI Interface: http://localhost:7860")
    else:
        print("\n💥 Image generation test failed.")
        print("Check the WebUI logs for more details.")

if __name__ == "__main__":
    main()