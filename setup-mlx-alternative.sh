#!/bin/bash

# MLX-based Stable Diffusion Alternative Setup
# This provides a high-performance alternative using Apple's MLX framework

echo "🍎 Setting up MLX-based Stable Diffusion for Apple Silicon..."
echo "📍 This will create a parallel MLX installation alongside your existing WebUI"
echo ""

# Create MLX environment
echo "📦 Setting up MLX environment..."
cd /Users/bost/git
git clone https://github.com/ml-explore/mlx-examples.git
cd mlx-examples/stable_diffusion

# Install MLX dependencies
echo "🔧 Installing MLX dependencies..."
uv init --python 3.10.6
uv add mlx
uv add huggingface-hub
uv add regex
uv add tqdm
uv add pillow
uv add numpy
uv add fastapi
uv add uvicorn

# Create MLX API server
echo "🌐 Creating MLX API server..."
cat > mlx_api_server.py << 'EOF'
#!/usr/bin/env python3
"""
MLX-based Stable Diffusion API Server
High-performance alternative for Apple Silicon
"""

import base64
import io
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlx.core as mx
from stable_diffusion import StableDiffusion
from PIL import Image
import numpy as np

app = FastAPI(title="MLX Stable Diffusion API")

# Global model instance
sd_model = None

class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    width: int = 1024
    height: int = 1024
    steps: int = 30
    cfg_scale: float = 7.5
    seed: Optional[int] = None

class GenerationResponse(BaseModel):
    images: list[str]  # Base64 encoded
    info: Optional[str] = None

@app.on_event("startup")
async def load_model():
    global sd_model
    print("🍎 Loading MLX Stable Diffusion model...")
    sd_model = StableDiffusion()
    print("✅ MLX model loaded successfully!")

@app.post("/sdapi/v1/txt2img")
async def txt2img(request: GenerationRequest):
    global sd_model
    
    if sd_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        print(f"🎨 Generating: {request.prompt[:50]}...")
        
        # Generate image using MLX
        image = sd_model.generate_image(
            request.prompt,
            n_images=1,
            steps=request.steps,
            cfg_weight=request.cfg_scale,
            negative_text=request.negative_prompt or "",
            seed=request.seed
        )
        
        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return GenerationResponse(
            images=[img_base64],
            info=f"Generated with MLX on Apple Silicon"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sdapi/v1/options")
async def get_options():
    return {"status": "MLX Stable Diffusion API Ready"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7861)  # Different port
EOF

# Create startup script
echo "🚀 Creating MLX startup script..."
cat > start-mlx-api.sh << 'EOF'
#!/bin/bash
echo "🍎 Starting MLX Stable Diffusion API on port 7861..."
echo "⚡ Optimized for Apple Silicon"
echo ""
echo "🌐 API will be available at: http://localhost:7861/docs"
echo "🔗 txt2img endpoint: http://localhost:7861/sdapi/v1/txt2img"
echo ""

uv run python mlx_api_server.py
EOF

chmod +x start-mlx-api.sh

echo ""
echo "✅ MLX setup complete!"
echo ""
echo "🚀 To use MLX (recommended for Apple Silicon):"
echo "   cd /Users/bost/git/mlx-examples/stable_diffusion"
echo "   ./start-mlx-api.sh"
echo ""
echo "📝 Update your Rust .env file to use MLX:"
echo "   AI_API_ENDPOINT=http://localhost:7861/sdapi/v1/txt2img"
echo ""
echo "🔥 Expected performance improvement: 3-5x faster on Apple Silicon!"