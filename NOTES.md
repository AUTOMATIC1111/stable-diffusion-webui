# Launch nix cuda dev shell
```nix shell cuda-shell.nix```
# Copy models as described in README
cp ~/Downloads/model.ckpt .
cp ~/Downloads/GFPGANv1.3.pth .
# Clone other repos as mentioned in README
```mkdir repositories```

```git clone https://github.com/CompVis/stable-diffusion.git repositories/stable-diffusion```

```git clone https://github.com/CompVis/taming-transformers.git repositories/taming-transformers```

```git clone https://github.com/sczhou/CodeFormer.git repositories/CodeFormer```

```git clone https://github.com/crowsonkb/k-diffusion.git repositories/k-diffusion```

# Create and activate conda env
```conda env create -f environment.yaml```

```conda activate webui```

# Install python packages via pip
```pip install -r requirements.txt --no-cache-dir --prefer-binary```

```pip install -r repositories/CodeFormer/requirements.txt --prefer-binary```

```pip install torch --extra-index-url https://download.pytorch.org/whl/cu113```