Sometimes it might be useful to move your models to another location. Reasons for this could be:
- Main disk has low disk space
- You are using models in multiple tools and don't want to store them twice

The default model folder is `stable-diffusion-webui/models`

## macOS Finder
- Open in Finder two windows e.g. `stable-diffusion-webui/models/Stable-diffusion` and the folder where your models are located.
- Press <kbd>option ⌥</kbd> + <kbd>command ⌘</kbd> while dragging your model from the model folder to the target folder
- This will make an alias instead of moving the models

## Command line
- Let's assume your model `openjourney-v4.ckpt` is stored in `~/ai/models/`
- Now we make a symbolic link (i.e. alias) to this model
- Open your terminal and navigate to your Stable Diffusion model folder e.g. `cd ~/stable-diffusion-webui/models/Stable-diffusion`
- Make a symbolic link to your model with `ln -sf  ~/ai/models/openjourney-v4.ckpt`