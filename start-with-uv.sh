#!/bin/bash

# Start Stable Diffusion WebUI with UV and Python 3.10.6
# This script uses the official launcher with UV environment

echo "🚀 Starting Stable Diffusion WebUI with UV and Python 3.10.6..."
echo "📍 Working directory: $(pwd)"
echo "🐍 Python version: $(uv run python --version)"
echo "⚡ UV environment: Active"
echo ""

echo "🌐 Starting WebUI with API enabled..."
echo "   API will be available at: http://localhost:7860/docs"
echo "   WebUI will be available at: http://localhost:7860"
echo ""

# Use the official launcher with UV - optimized for Apple Silicon
COMMANDLINE_ARGS="--api --listen --skip-torch-cuda-test --api-log --cors-allow-origins=* --opt-split-attention-v1 --medvram --no-half-vae" uv run python launch.py