#!/usr/bin/env bash

declare -A arr

# models
arr["https://huggingface.co/prompthero/openjourney/resolve/main/mdjrny-v4.safetensors"]="models/Stable-diffusion"
arr+=(["https://huggingface.co/prompthero/openjourney-v2/resolve/main/openjourney-v2.ckpt"]="models/Stable-diffusion")
arr+=(["https://huggingface.co/ckpt/anything-v4.5-vae-swapped/resolve/main/anything-v4.5-vae-swapped.safetensors"]="models/Stable-diffusion")
arr+=(["https://huggingface.co/dreamlike-art/dreamlike-photoreal-2.0/resolve/main/dreamlike-photoreal-2.0.safetensors"]="models/Stable-diffusion")
arr+=(["https://huggingface.co/nuigurumi/basil_mix/resolve/main/Basil_mix_fixed.safetensors"]="models/Stable-diffusion")
arr+=(["https://huggingface.co/swl-models/chilloutmix-ni/resolve/main/chilloutmix-Ni-ema-fp32.safetensors"]="models/Stable-diffusion")
# vae
arr+=(["https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors"]="models/VAE")
# embeddings
arr+=(["https://huggingface.co/datasets/gsdf/EasyNegative/resolve/main/EasyNegative.safetensors"]="embeddings")

for key in ${!arr[@]}; do
  mkdir -p "${arr[${key}]}"
  download_to="${arr[${key}]}"/$(basename "${key}")
  if [ ! -f "$download_to" ]; then
    echo "Download ${key} to ${arr[${key}]}"
    curl -Lo "$download_to" "${key}"
  fi
done