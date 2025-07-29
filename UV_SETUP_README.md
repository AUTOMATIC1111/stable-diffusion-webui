# UV Setup for Stable Diffusion WebUI

## ✅ Setup Complete!

UV has been successfully configured for Stable Diffusion WebUI with Python 3.10.6.

## 🚀 Quick Start

### 1. Start the WebUI with API (Simplified!)
```bash
cd /Users/bost/git/stable-diffusion-webui
./start-with-uv.sh
```

**Note**: The first run will automatically:
- Download the default Stable Diffusion model (~4GB)
- Install any missing dependencies
- Set up the API endpoints

### 2. Test the API
```bash
# In a new terminal
uv run python test-api.py
```

### 3. Use with your Rust tool
Update your `.env` file in the Rust project:
```bash
AI_API_KEY=not_required_for_local
AI_API_ENDPOINT=http://localhost:7860/sdapi/v1/txt2img
AI_MODEL=stable-diffusion-v1-5
```

## 📋 What Was Set Up

### Files Created/Modified:
- ✅ `.python-version` → Forces Python 3.10.6
- ✅ `pyproject.toml` → Updated for UV compatibility
- ✅ `start-with-uv.sh` → UV-powered startup script
- ✅ `test-api.py` → API connectivity test
- ✅ `.venv/` → Virtual environment with all dependencies

### Dependencies Installed:
- 🐍 Python 3.10.6 (exactly as required)
- 🎨 All Stable Diffusion WebUI requirements
- 🚀 Optimized for food image generation

## 🔧 Available Commands

### Start WebUI
```bash
./start-with-uv.sh
```

### Test API
```bash
uv run python test-api.py
```

### Run any Python command
```bash
uv run python <your-script.py>
```

### Install additional packages
```bash
uv add <package-name>
```

### Check Python version
```bash
uv run python --version
```

## 🌐 API Endpoints

Once running, these will be available:

- **WebUI Interface**: http://localhost:7860
- **API Documentation**: http://localhost:7860/docs
- **Text-to-Image API**: http://localhost:7860/sdapi/v1/txt2img
- **API Options**: http://localhost:7860/sdapi/v1/options

## 🎯 Integration with Rust Tool

Your Rust food image generation tool is now ready to connect! The API endpoint configuration is:

```bash
# In your Rust project's .env file:
AI_API_KEY=not_required_for_local
AI_API_ENDPOINT=http://localhost:7860/sdapi/v1/txt2img
AI_MODEL=stable-diffusion-v1-5
AI_TIMEOUT_SECONDS=120
AI_MAX_RETRIES=3
AI_RETRY_DELAY_MS=2000
```

## 🐛 Troubleshooting

### If the WebUI won't start:
```bash
# Check Python version
uv run python --version  # Should show 3.10.6

# Reinstall dependencies
uv sync --reinstall

# Check for errors
./start-with-uv.sh
```

### If API calls fail:
```bash
# Test connectivity
curl http://localhost:7860/sdapi/v1/options

# Run the test script
uv run python test-api.py
```

### If images are poor quality:
1. Download better models to `models/Stable-diffusion/`
2. Adjust the prompt templates in your Rust code
3. Increase the `steps` parameter (30-50 for better quality)

## 📦 Model Management

### Download recommended models:
```bash
cd models/Stable-diffusion/

# For general use (good for food):
wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors

# For photorealistic food:
wget https://civitai.com/api/download/models/130072 -O realistic-vision-v5.safetensors
```

## 🔥 Performance Tips

### For faster generation:
- Use `--xformers` in the startup script (if installed)
- Reduce image size to 512x512 for testing
- Use fewer steps (20-30) for development

### For better quality:
- Use higher resolution (1024x1024)
- More steps (30-50)
- Better models (see model recommendations above)

## ✨ Success Indicators

You'll know everything is working when:

1. ✅ `./start-with-uv.sh` starts without errors
2. ✅ WebUI loads at http://localhost:7860
3. ✅ `uv run python test-api.py` generates a test image
4. ✅ Your Rust tool can connect and generate food images

## 🎉 Next Steps

1. **Test the setup**: Run `./start-with-uv.sh` and `uv run python test-api.py`
2. **Update your Rust tool**: Set the API endpoint in your `.env` file
3. **Generate test images**: Use your Rust tool with the sample CSV
4. **Download better models**: For higher quality food images
5. **Optimize prompts**: Tweak the food-specific prompts in your Rust code

Your local AI food image generation pipeline is now ready! 🍕🥗🍗