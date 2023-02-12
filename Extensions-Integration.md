# Integration Extensions

## auto-sd-paint-ext

https://github.com/Interpause/auto-sd-paint-ext
>Extension for AUTOMATIC1111's webUI with Krita Plugin (other drawing studios soon?)
![image](https://user-images.githubusercontent.com/98228077/217986983-d23a334d-50bc-4bb1-ac8b-60f99a3b07b9.png)
- Optimized workflow (txt2img, img2img, inpaint, upscale) & UI design.
- Only drawing studio plugin that exposes the Script API.
See https://github.com/Interpause/auto-sd-paint-ext/issues/41 for planned developments.
See [CHANGELOG.md](https://github.com/Interpause/auto-sd-paint-ext/blob/main/CHANGELOG.md) for the full changelog.

## openOutpaint extension
https://github.com/zero01101/openOutpaint-webUI-extension

A tab with the full openOutpaint UI. Run with the --api flag.

![image](https://user-images.githubusercontent.com/98228077/209453250-8bd2d766-56b5-4887-97bc-cb0e22f98dde.png)

## DAAM
https://github.com/kousw/stable-diffusion-webui-daam

DAAM stands for Diffusion Attentive Attribution Maps. Enter the attention text (must be a string contained in the prompt) and run. An overlapping image with a heatmap for each attention will be generated along with the original image.

![image](https://user-images.githubusercontent.com/98228077/208332173-ffb92131-bd02-4a07-9531-136822d06c86.png)

## Visualize Cross-Attention
https://github.com/benkyoujouzu/stable-diffusion-webui-visualize-cross-attention-extension

![image](https://user-images.githubusercontent.com/98228077/208332131-6acdae9a-2b25-4e71-8ab0-6e375c7c0419.png)

Generates highlighted sectors of a submitted input image, based on input prompts. Use with tokenizer extension. See the readme for more info.

## Deforum
https://github.com/deforum-art/deforum-for-automatic1111-webui


The official port of Deforum, an extensive script for 2D and 3D animations, supporting keyframable sequences, dynamic math parameters (even inside the prompts), dynamic masking, depth estimation and warping.

![image](https://user-images.githubusercontent.com/98228077/217986819-67fd1e3c-b007-475c-b8c9-3bf28ee3aa67.png)

## Push to 🤗 Hugging Face

https://github.com/camenduru/stable-diffusion-webui-huggingface

![Push Folder to Hugging Face](https://user-images.githubusercontent.com/54370274/206897701-9e86ce7c-af06-4d95-b9ea-385276c99d3a.jpg)

To install it, clone the repo into the `extensions` directory and restart the web ui:

`git clone https://github.com/camenduru/stable-diffusion-webui-huggingface`

`pip install huggingface-hub`

## Auto TLS-HTTPS
https://github.com/papuSpartan/stable-diffusion-webui-auto-tls-https

Allows you to easily, or even completely automatically start using HTTPS.

## NSFW checker
https://github.com/AUTOMATIC1111/stable-diffusion-webui-nsfw-censor

Replaces NSFW images with black.

## DH Patch
https://github.com/d8ahazard/sd_auto_fix

Random patches by D8ahazard. Auto-load config YAML files for v2, 2.1 models; patch latent-diffusion to fix attention on 2.1 models (black boxes without no-half), whatever else I come up with.

## Stable Horde

### Stable Horde Client
https://github.com/natanjunges/stable-diffusion-webui-stable-horde

Generate pictures using other user's PC. You should be able to recieve images from the stable horde with anonymous `0000000000` api key, however it is recommended to get your own - https://stablehorde.net/register

Note: Retrieving Images may take 2 minutes or more, especially if you have no kudos. 

## Stable Horde Worker
https://github.com/sdwebui-w-horde/sd-webui-stable-horde-worker

An unofficial [Stable Horde](https://stablehorde.net/) worker bridge as a [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) extension.

### Features

**This extension is still WORKING IN PROGRESS**, and is not ready for production use.

- Get jobs from Stable Horde, generate images and submit generations
- Configurable interval between every jobs
- Enable and disable extension whenever
- Detect current model and fetch corresponding jobs on the fly
- Show generation images in the Stable Diffusion WebUI
- Save generation images with png info text to local

### Install

- Run the following command in the root directory of your Stable Diffusion WebUI installation:

  ```bash
  git clone https://github.com/sdwebui-w-horde/sd-webui-stable-horde-worker.git extensions/stable-horde-worker
  ```

- Launch the Stable Diffusion WebUI, You would see the `Stable Horde Worker` tab page.

  ![settings](./screenshots/settings.png)

- Register an account on [Stable Horde](https://stablehorde.net/) and get your `API key` if you don't have one.

  **Note**: the default anonymous key `00000000` is not working for a worker, you need to register an account and get your own key.

- Setup your `API key` here.
- Setup `Worker name` here with a proper name.
- Make sure `Enable` is checked.
- Click the `Apply settings` buttons.

## Discord Rich Presence
https://github.com/kabachuha/discord-rpc-for-automatic1111-webui

Provides connection to Discord RPC, showing a fancy table in the user profile.

## Aesthetic Image Scorer
https://github.com/tsngo/stable-diffusion-webui-aesthetic-image-scorer

Extension for https://github.com/AUTOMATIC1111/stable-diffusion-webui

Calculates aesthetic score for generated images using [CLIP+MLP Aesthetic Score Predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor) based on [Chad Scorer](https://github.com/grexzen/SD-Chad/blob/main/chad_scorer.py)

See [Discussions](https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/1831)

Saves score to windows tags with other options planned

![picture](https://github.com/tsngo/stable-diffusion-webui-aesthetic-image-scorer/blob/main/tag_group_by.png)


## Aesthetic Scorer
https://github.com/vladmandic/sd-extension-aesthetic-scorer

Uses existing CLiP model with an additional small pretrained to calculate perceived aesthetic score of an image  

Enable or disable via `Settings` -> `Aesthetic scorer`  

This is an *"invisible"* extension, it runs in the background before any image save and  
appends **`score`** as *PNG info section* and/or *EXIF comments* field

### Notes

- Configuration via **Settings** &rarr; **Aesthetic scorer**  
  ![screenshot](https://raw.githubusercontent.com/vladmandic/sd-extension-aesthetic-scorer/main/aesthetic-scorer.jpg)
- Extension obeys existing **Move VAE and CLiP to RAM** settings
- Models will be auto-downloaded upon first usage (small)
- Score values are `0..10`  
- Supports both `CLiP-ViT-L/14` and `CLiP-ViT-B/16`
- Cross-platform!

## cafe-aesthetic
https://github.com/p1atdev/stable-diffusion-webui-cafe-aesthetic

Pre-trained model, determines if aesthetic/non-aesthetic, does 5 different style recognition modes, and Waifu confirmation. Also has a tab with Batch processing.

## Stable Diffusion Aesthetic Scorer
https://github.com/grexzen/SD-Chad

Rates your images.

## mine-diffusion
https://github.com/fropych/mine-diffusion

This extension converts images into blocks and creates schematics for easy importing into Minecraft using the Litematica mod.

![](https://github.com/fropych/mine-diffusion/blob/master/README_images/demo.gif)

