# Aesthetic Scorer extension for SD Automatic WebUI

Uses existing CLiP model with an additional small pretrained to calculate perceived aesthetic score of an image  

This is an *"invisible"* extension, it runs in the background before any image save and  
appends **`score`** as *PNG info section* and/or *EXIF comments* field

## Notes

- Configuration via **Settings** &rarr; **Aesthetic scorer**  
  ![screenshot](aesthetic-scorer.jpg)
- Extension obeys existing **Move VAE and CLiP to RAM** settings
- Models will be auto-downloaded upon first usage (small)
- Score values are `0..10`  
- Supports both `CLiP-ViT-L/14` and `CLiP-ViT-B/16`

This extension uses different method than [Aesthetic Image Scorer](https://github.com/tsngo/stable-diffusion-webui-aesthetic-image-scorer) extension which:
- Uses modified [SD Chad scorer](https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/1831) implementation
- Windows-only!
- Executes as to replace `image.save` so limited compatibity with other *non-txt2img* use-cases

## Credits

- Based on: [simulacra-aesthetic-models](https://github.com/crowsonkb/simulacra-aesthetic-models)  
- Training data set: [simulacra-aesthetic-captions](https://github.com/JD-P/simulacra-aesthetic-captions)  
