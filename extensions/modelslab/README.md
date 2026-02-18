# ModelsLab Extension for AUTOMATIC1111

> Transform your AUTOMATIC1111 interface with ModelsLab's superior AI models and cost-effective cloud generation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![AUTOMATIC1111](https://img.shields.io/badge/AUTOMATIC1111-Compatible-green)](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
[![ModelsLab](https://img.shields.io/badge/ModelsLab-API-blue)](https://modelslab.com)

**Generate images in the cloud while keeping your favorite AUTOMATIC1111 interface.** Access Flux, Stable Diffusion XL, Playground v2.5, and more professional models without needing expensive GPUs.

## 🎯 Why Choose ModelsLab?

### vs Local Generation
✅ **No GPU Required** - Works on any machine, even laptops  
✅ **Latest Models** - Always up-to-date Flux, SDXL, Playground versions  
✅ **Cost Effective** - Pay per generation vs GPU purchase/electricity  
✅ **Instant Start** - No model downloads or setup time  
✅ **Scalable** - Generate multiple images simultaneously  

### vs Other Cloud APIs
✅ **13+ Models** - More variety than Replicate, fal.ai, or OpenAI  
✅ **Better Pricing** - More competitive rates than competitors  
✅ **Professional Tools** - Background removal, headshots, inpainting  
✅ **Native Integration** - Purpose-built for AUTOMATIC1111 interface  

## ✨ Features

### 🎨 **Multiple Generation Modes**
- **Text-to-Image** - Generate from detailed prompts using Flux, SDXL, Playground
- **Image-to-Image** - Transform existing images with AI control
- **Professional Quality** - Access to ModelsLab's optimized model implementations

### 🔧 **Smart Integration**
- **Familiar Interface** - Same AUTOMATIC1111 UI you already know
- **Parameter Mapping** - All your favorite settings work seamlessly
- **Batch Support** - Generate multiple images just like local generation
- **Progress Tracking** - Real-time generation status and progress bars

### 💡 **Enhanced Experience**
- **Model Tooltips** - Learn about each model's strengths and capabilities
- **Cost Estimation** - See estimated costs before generating
- **API Key Validation** - Instant feedback on configuration
- **Error Handling** - Clear, helpful error messages

## 🚀 Quick Start

### Prerequisites
- [AUTOMATIC1111 Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) installed
- [ModelsLab API Key](https://modelslab.com/dashboard/api-keys) (free to get)

### Installation

#### Option 1: Git Clone (Recommended)
```bash
cd /path/to/stable-diffusion-webui
git clone https://github.com/modelslab/automatic1111-extension.git extensions/modelslab-api
```

#### Option 2: Manual Download
1. Download this repository as ZIP
2. Extract to `extensions/modelslab-api/` in your AUTOMATIC1111 directory
3. Restart AUTOMATIC1111

#### Option 3: Extension Manager
1. Open AUTOMATIC1111 WebUI
2. Go to **Extensions** tab
3. Click **Install from URL**
4. Paste: `https://github.com/modelslab/automatic1111-extension.git`
5. Click **Install**
6. Restart WebUI

### Configuration

1. **Get API Key**
   - Visit [ModelsLab Dashboard](https://modelslab.com/dashboard/api-keys)
   - Create account if needed (it's free!)
   - Generate new API key

2. **Configure Extension**
   - Launch AUTOMATIC1111
   - Go to txt2img or img2img tab
   - Scroll down to find **"ModelsLab API"** section
   - Enter your API key
   - Select your preferred model (Flux recommended)

3. **Start Generating**
   - Check "Use ModelsLab API" checkbox
   - Enter your prompt as usual
   - Click Generate
   - Enjoy cloud-powered results!

## 📖 Usage Guide

### Basic Generation

1. **Enable ModelsLab**: Check the "Use ModelsLab API" checkbox
2. **Choose Model**: Select from Flux, SDXL, or Playground v2.5
3. **Enter Prompt**: Use the same prompts you'd use locally
4. **Configure Settings**: All your usual parameters work (CFG, steps, etc.)
5. **Generate**: Click generate and wait for cloud processing

### Available Models

| Model | Best For | Speed | Cost |
|-------|----------|--------|------|
| **Flux** ⭐ | High-quality, prompt adherence | Fast (2-5s) | ~$0.02/image |
| **Stable Diffusion XL** | Artistic, open-source style | Fast (3-6s) | ~$0.015/image |
| **Playground v2.5** | Creative, stylized images | Fast (2-4s) | ~$0.018/image |

### img2img Transformation

1. Upload your base image as usual in img2img tab
2. Enable ModelsLab and choose model
3. Set transformation strength (0.1 = minimal change, 1.0 = completely new)
4. Enter transformation prompt
5. Generate to see AI-powered transformations

### Advanced Tips

- **Prompt Enhancement**: Enable "Enhance Prompt" for automatic prompt improvement
- **Batch Generation**: Use batch count for multiple variations
- **Cost Control**: Monitor the cost estimate display
- **API Management**: Set API key via environment variable `MODELSLAB_API_KEY`

## ⚙️ Configuration Options

### Environment Variables
```bash
# API Configuration
export MODELSLAB_API_KEY="your-api-key"
export MODELSLAB_DEFAULT_MODEL="flux"
export MODELSLAB_BASE_URL="https://modelslab.com/api/v6"

# Performance
export MODELSLAB_TIMEOUT="30"
export MODELSLAB_MAX_RETRIES="3"

# Debugging
export MODELSLAB_DEBUG="false"
```

### Command Line Arguments
```bash
# Launch with ModelsLab configuration
python launch.py \
  --modelslab-api-key "your-key" \
  --modelslab-default-model "flux" \
  --modelslab-debug
```

### UI Settings
- **API Key**: Password-protected input with validation
- **Model Selection**: Dropdown with descriptions and cost estimates
- **Enhance Prompt**: Auto-improve prompts for better results
- **Safety Checker**: Content filtering toggle
- **Strength Control**: img2img transformation intensity

## 🎨 Use Cases

### Content Creators
```
Use Case: Blog post illustrations
Process: Write article → Generate featured image with Flux → Download and publish
Benefit: Professional images without stock photo costs
```

### Digital Artists
```
Use Case: Concept art and iterations
Process: Rough sketch → img2img with Playground → Multiple variations
Benefit: Rapid iteration without local GPU requirements
```

### Small Businesses
```
Use Case: Marketing materials
Process: Product description → Generate with SDXL → Social media posts
Benefit: Professional marketing without design costs
```

### Students & Researchers
```
Use Case: Academic presentations
Process: Research topic → Generate illustrations → Enhanced presentations
Benefit: Access to latest AI models without hardware investment
```

## 🛠️ Troubleshooting

### Common Issues

#### Extension Not Showing
- **Check Installation**: Ensure files are in `extensions/modelslab-api/`
- **Restart WebUI**: Extensions load at startup
- **Check Logs**: Look for errors in console output

#### API Key Issues
```
❌ Problem: "API key required" error
✅ Solution: Get key from https://modelslab.com/dashboard/api-keys

❌ Problem: "Invalid API key" error  
✅ Solution: Check key is copied correctly, no extra spaces

❌ Problem: "Quota exceeded" error
✅ Solution: Check your ModelsLab account balance and limits
```

#### Generation Failures
```
❌ Problem: "Request timeout" error
✅ Solution: Check internet connection, try again

❌ Problem: "Model not found" error
✅ Solution: Verify model name, update extension

❌ Problem: Images not displaying
✅ Solution: Check firewall/proxy settings for image URLs
```

### Debug Mode

Enable detailed logging:
```bash
python launch.py --modelslab-debug
```

This will show:
- API request details
- Response processing
- Error stack traces
- Performance metrics

### Getting Help

1. **Check Logs**: Enable debug mode and check console output
2. **Test API Key**: Verify it works at [ModelsLab Dashboard](https://modelslab.com/dashboard)
3. **Update Extension**: Pull latest changes from repository
4. **Report Issues**: Open GitHub issue with logs and steps to reproduce

## 📊 Performance Comparison

### Generation Times
| Method | Flux | SDXL | Playground |
|--------|------|------|------------|
| **ModelsLab Cloud** | 2-5s | 3-6s | 2-4s |
| **Local RTX 4090** | 8-12s | 15-20s | 10-15s |
| **Local RTX 3080** | 15-25s | 30-40s | 20-30s |

### Cost Analysis (Per Image)
| Method | Initial Cost | Per Image | 100 Images | 1000 Images |
|--------|-------------|-----------|------------|-------------|
| **ModelsLab** | $0 | ~$0.02 | $2 | $20 |
| **RTX 4090** | $1600 | ~$0.05* | $5* | $50* |
| **RTX 3080** | $800 | ~$0.08* | $8* | $80* |

*Includes electricity costs

## 🔧 Development

### Extension Structure
```
extensions/modelslab-api/
├── install.py              # Dependency installation
├── preload.py              # Command line arguments
├── scripts/
│   └── modelslab_generation.py  # Main integration script
├── javascript/
│   └── modelslab.js        # UI enhancements
├── style.css               # Visual styling
└── README.md               # This file
```

### Adding New Models

To add support for new ModelsLab models:

1. Update `MODELSLAB_MODELS` in `modelslab_generation.py`
2. Add model info to JavaScript configuration
3. Update documentation and UI tooltips

### Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Test thoroughly with real AUTOMATIC1111 installation
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **ModelsLab Dashboard**: [modelslab.com/dashboard](https://modelslab.com/dashboard)
- **API Documentation**: [docs.modelslab.com](https://docs.modelslab.com)
- **AUTOMATIC1111**: [github.com/AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
- **Extension Repository**: [github.com/modelslab/automatic1111-extension](https://github.com/modelslab/automatic1111-extension)

## 🙏 Acknowledgments

- **AUTOMATIC1111** - For creating the most popular Stable Diffusion interface
- **ModelsLab Team** - For providing superior AI models and API
- **Community** - For testing, feedback, and feature requests

---

**Transform your image generation workflow today!** 
🚀 [Get your ModelsLab API key](https://modelslab.com/dashboard/api-keys) and start generating with superior models in your familiar AUTOMATIC1111 interface.