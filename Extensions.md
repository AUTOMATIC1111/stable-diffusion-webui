# General Info

Extensions are a more convenient form of user scripts.

Extensions all exist in their own folder inside the extensions folder of webui. You can use git to install an extension like this:

    git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui-aesthetic-gradients extensions/aesthetic-gradients

This installs an extension from https://github.com/AUTOMATIC1111/stable-diffusion-webui-aesthetic-gradients into the extensions/aesthetic-gradients directory.

Alternatively you can just copy-paste a directory into extensions.

For developing extensions, see [Developing extensions](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Extensions/Developing-extensions).

# Security

As extensions allow the user to install and run arbitrary code, this can be used maliciously, and is disabled by default when running with options that allow remote users to connect to the server (`--share` or `--listen`) - you'll still have the UI, but trying to install anything will result in error. If you want to use those options and still be able to install extensions, use `--enable-insecure-extension-access` command line flag.

# Extensions

## MultiDiffusion with Tiled VAE
https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111

### MultiDiffusion

- txt2img panorama generation, as mentioned in MultiDiffusion.
- It can cooperate with ControlNet to produce wide images with control.

Panorama Example:
Before: [click for the raw image](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/blob/docs/imgs/ancient_city_origin.jpeg)
After: [click for the raw image](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/blob/docs/imgs/ancient_city.jpeg)

ControlNet Canny Output: https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/raw/docs/imgs/yourname.jpeg?raw=true

### Tiled Vae

The `vae_optimize.py` script is a wild hack that splits the image into tiles, encodes each tile separately, and merges the result back together. This process allows the VAE to work with giant images on limited VRAM (~10 GB for 8K images!).

Remove --lowvram and --medvram to enjoy!

## VRAM Estimator
https://github.com/space-nuko/a1111-stable-diffusion-webui-vram-estimator

Runs txt2img, img2img, highres-fix at increasing dimensions and batch sizes until OOM, and outputs data to graph.

![image](https://user-images.githubusercontent.com/98228077/223624383-545aeb31-c001-4ba6-bdb8-23e688130b8f.png)

## Dump U-Net
https://github.com/hnmr293/stable-diffusion-webui-dumpunet

View different layers, observe U-Net feature maps. Allows Image generation by giving different prompts for each block of the unet: https://note.com/kohya_ss/n/n93b7c01b0547

![image](https://user-images.githubusercontent.com/98228077/223624012-2df926d5-d4c4-44bc-a04f-bed15d43b88f.png)

## posex
https://github.com/hnmr293/posex

Estimated Image Generator for Pose2Image. This extension allows moving the openpose figure in 3d space.

![image](https://user-images.githubusercontent.com/98228077/223622234-26907947-a723-4671-ae42-60a0011bfda2.png)

	
## LLuL
https://github.com/hnmr293/sd-webui-llul

Local Latent Upscaler. Target an area to selectively enhance details.

https://user-images.githubusercontent.com/120772120/221390831-9fbccdf8-5898-4515-b988-d6733e8af3f1.mp4


## CFG-Schedule-for-Automatic1111-SD
https://github.com/guzuligo/CFG-Schedule-for-Automatic1111-SD

These 2 scripts allow for dynamic CFG control during generation steps. With the right settings, this could help get the details of high CFG without damaging the generated image even with low denoising in img2img.

See their [wiki](https://github.com/guzuligo/CFG-Schedule-for-Automatic1111-SD/wiki/CFG-Auto-script) on how to use.

## a1111-sd-webui-locon
https://github.com/KohakuBlueleaf/a1111-sd-webui-locon
An extension for loading LoCon networks in webui.

## ebsynth_utility
https://github.com/s9roll7/ebsynth_utility

Extension for creating videos using img2img and ebsynth. Output edited videos using ebsynth. Works with ControlNet extension.

![image](https://user-images.githubusercontent.com/98228077/223622872-0575abe9-9a53-4614-b9a5-1333f0b34733.png)


## Lora Block Weight

Lora is a powerful tool, but it is sometimes difficult to use and can affect areas that you do not want it to affect. This script allows you to set the weights block-by-block. Using this script, you may be able to get the image you want.

Used in conjunction with the XY plot, it is possible to examine the impact of each level of the hierarchy.

![image](https://user-images.githubusercontent.com/98228077/223573538-d8fdb00d-6c49-47ec-af63-cea691f515d4.png)

Included Presets:

```
NOT:0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 
ALL:1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 
INS:1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0
IND:1,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0
INALL:1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0
MIDD:1,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0
OUTD:1,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0
OUTS:1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1
OUTALL:1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1
```

## Kitchen Theme
https://github.com/canisminor1990/sd-web-ui-kitchen-theme

A custom theme for webui.

![image](https://user-images.githubusercontent.com/98228077/223572989-e51eb877-74f4-41ec-8204-233221f2e981.png)


## Bilingual Localization
https://github.com/journey-ad/sd-webui-bilingual-localization

Bilingual translation, no need to worry about how to find the original button. Compatible with language pack extensions, no need to re-import. 

![image](https://user-images.githubusercontent.com/98228077/223564624-61594e71-d1dd-4f32-9293-9697b07a7735.png)

## Composable Lora
https://github.com/opparco/stable-diffusion-webui-composable-lora

Enables using AND keyword(composable diffusion) to limit LoRAs to subprompts. Useful when paired with Latent Couple extension.

## Clip Interrogator
https://github.com/pharmapsychotic/clip-interrogator-ext

Clip Interrogator by pharmapsychotic ported to an extension. Features a variety of clip models and interrogate settings.

![image](https://user-images.githubusercontent.com/98228077/223572478-093030bf-e25e-42c0-b621-597515deaf69.png)

## Latent-Couple
https://github.com/opparco/stable-diffusion-webui-two-shot

An extension of the built-in Composable Diffusion, allows you to determine the region of the latent space that reflects your subprompts.

![image](https://user-images.githubusercontent.com/98228077/223571685-95a300f9-b768-4bca-96d4-684aadae9863.png)


## OpenPose Editor
https://github.com/fkunn1326/openpose-editor

This can add multiple pose characters, detect pose from image, save to PNG, and send to controlnet extension.

![image](https://user-images.githubusercontent.com/98228077/223571127-6c107bd8-7ca4-4774-bdb8-41930863fdcc.png)


## SuperMerger
https://github.com/hako-mikan/sd-webui-supermerger

Merge and run without saving to drive. Sequential XY merge generations; extract and merge loras, bind loras to ckpt, merge block weights, and more.

![image](https://user-images.githubusercontent.com/98228077/223570729-d25ca5c4-a434-42fd-b85d-7e16a7af1fc8.png)

## Prompt Translator
https://github.com/butaixianran/Stable-Diffusion-Webui-Prompt-Translator

A integrated translator for translating prompts to English using Deepl or Baidu.

![image](https://user-images.githubusercontent.com/98228077/223565541-43f618ea-e009-41b5-880c-7360a9ebec5f.png)

## Video Loopback
https://github.com/fishslot/video_loopback_for_webui

https://user-images.githubusercontent.com/122792358/218375476-a4116c74-5a9a-41e2-970a-c3cc09f796ae.mp4

## Mine Diffusion
https://github.com/fropych/mine-diffusion

This extension converts images into blocks and creates schematics for easy importing into Minecraft using the Litematica mod.

<details><summary>Example: (Click to expand:)</summary>

![](https://github.com/fropych/mine-diffusion/blob/master/README_images/demo.gif)

</details>

## anti-burn
https://github.com/klimaleksus/stable-diffusion-webui-anti-burn

Smoothing generated images by skipping a few very last steps and averaging together some images before them.

![image](https://user-images.githubusercontent.com/98228077/223562829-1abe8eed-dca5-4891-88e2-6714966e02bc.png)

## Embedding Merge
https://github.com/klimaleksus/stable-diffusion-webui-embedding-merge

Merging Textual Inversion embeddings at runtime from string literals.

![image](https://user-images.githubusercontent.com/98228077/223562706-af7cc1e6-7b6c-4069-a89f-8087a2dba4da.png)

## gif2gif

The purpose of this script is to accept an animated gif as input, process frames as img2img typically would, and recombine them back into an animated gif. Intended to provide a fun, fast, gif-to-gif workflow that supports new models and methods such as Controlnet and InstructPix2Pix. Drop in a gif and go. Referenced code from prompts_from_file.

<details><summary>Example: (Click to expand:)</summary>

![](https://user-images.githubusercontent.com/93007558/216803715-81dfc9e6-8c9a-47d5-9879-27acfac34eb8.gif)

</details>

## cafe-aesthetic
https://github.com/p1atdev/stable-diffusion-webui-cafe-aesthetic

Pre-trained model, determines if aesthetic/non-aesthetic, does 5 different style recognition modes, and Waifu confirmation. Also has a tab with Batch processing.

![image](https://user-images.githubusercontent.com/98228077/223562229-cba2db0a-3368-4f13-9456-ebe2053c01a3.png)


## Catppuccin themes
https://github.com/catppuccin/stable-diffusion-webui

Catppuccin is a community-driven pastel theme that aims to be the middle ground between low and high contrast themes. Adds set of themes which are in compliance with catppucin guidebook.

![image](https://user-images.githubusercontent.com/98228077/223562461-13ec3132-4734-4787-a161-b2c408646835.png)


## Dynamic Thresholding
Dynamic Thresholding Adds customizable dynamic thresholding to allow high CFG Scale values without the burning / 'pop art' effect.

Adds customizable dynamic thresholding to allow high CFG Scale values without the burning / 'pop art' effect.


## Custom Diffusion
https://github.com/guaneec/custom-diffusion-webui

Custom Diffusion is, in short, finetuning-lite with TI, instead of tuning the whole model. Similar speed and memory requirements to TI and supposedly gives better results in less steps.


## Fusion
https://github.com/ljleb/prompt-fusion-extension

Adds prompt-travel and shift-attention-like interpolations (see exts), but during/within the sampling steps. Always-on + works w/ existing prompt-editing syntax. Various interpolation modes. See their wiki for more info.

<details><summary>Example: (Click to expand:)</summary>

![](https://user-images.githubusercontent.com/32277961/214725976-b72bafc6-0c5d-4491-9c95-b73da41da082.gif)

</details>

## Pixelization
https://github.com/AUTOMATIC1111/stable-diffusion-webui-pixelization

Using pre-trained models, produce pixel art out of images in the extras tab.

![image](https://user-images.githubusercontent.com/98228077/223563687-cb0eb3fe-0fce-4822-8170-20b719f394fa.png)

			
## Instruct-pix2pix
https://github.com/Klace/stable-diffusion-webui-instruct-pix2pix

Adds a tab for doing img2img editing with the instruct-pix2pix model. The author added the feature to webui, so this doesn't need to be used.

## System Info
https://github.com/vladmandic/sd-extension-system-info

Creates a top-level **System Info** tab in Automatic WebUI with 

*Note*:
- State & memory info are auto-updated every second if tab is visible  
  (no updates are performed when tab is not visible)  
- All other information is updated once upon WebUI load and  
  can be force refreshed if required  

![screenshot](https://raw.githubusercontent.com/vladmandic/sd-extension-system-info/main/system-info.jpg)

## Steps Animation
https://github.com/vladmandic/sd-extension-steps-animation

Extension to create animation sequence from denoised intermediate steps  
Registers a script in **txt2img** and **img2img** tabs

Creating animation has minimum impact on overall performance as it does not require separate runs  
except adding overhead of saving each intermediate step as image plus few seconds to actually create movie file  

Supports **color** and **motion** interpolation to achieve animation of desired duration from any number of interim steps  
Resulting movie fiels are typically very small (*~1MB being average*) due to optimized codec settings  

![screenshot](https://raw.githubusercontent.com/vladmandic/sd-extension-steps-animation/main/steps-animation.jpg)

### [Example](https://user-images.githubusercontent.com/57876960/212490617-f0444799-50e5-485e-bc5d-9c24a9146d38.mp4)


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


## Discord Rich Presence
https://github.com/kabachuha/discord-rpc-for-automatic1111-webui

Provides connection to Discord RPC, showing a fancy table in the user profile.


## Promptgen
https://github.com/AUTOMATIC1111/stable-diffusion-webui-promptgen

Use transformers models to generate prompts.

![image](https://user-images.githubusercontent.com/98228077/223561862-27815193-acfd-47cb-ae67-fcc435b2c875.png)


## haku-img
https://github.com/KohakuBlueleaf/a1111-sd-webui-haku-img

Image utils extension. Allows blending, layering, hue and color adjustments, blurring and sketch effects, and basic pixelization.

![image](https://user-images.githubusercontent.com/98228077/223561769-294ee4fa-f857-4dc9-afbf-dfe953e8c6ad.png)


## Merge Block Weighted
https://github.com/bbc-mc/sdweb-merge-block-weighted-gui

Merge models with separate rate for each 25 U-Net block (input, middle, output).

![image](https://user-images.githubusercontent.com/98228077/223561099-c9cb6fab-c3c6-42fb-92fd-6811474d073c.png)


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


## Stable Horde

### Stable Horde Client
https://github.com/natanjunges/stable-diffusion-webui-stable-horde

Generate pictures using other user's PC. You should be able to receive images from the stable horde with anonymous `0000000000` api key, however it is recommended to get your own - https://stablehorde.net/register

Note: Retrieving Images may take 2 minutes or more, especially if you have no kudos.


## Multiple hypernetworks 
https://github.com/antis0007/sd-webui-multiple-hypernetworks

Extension that allows the use of multiple hypernetworks at once

![image](https://user-images.githubusercontent.com/32306715/212293588-a8b4d1e9-4099-4a2e-a61a-f549a70f6096.png) 


## Hypernetwork-Monkeypatch-Extension
https://github.com/aria1th/Hypernetwork-MonkeyPatch-Extension

Extension that provides additional training features for hypernetwork training, and supports multiple hypernetworks.

![image](https://user-images.githubusercontent.com/35677394/212069329-7f3d427f-efad-4424-8dca-4bec010ea429.png)


## Ultimate SD Upscaler
https://github.com/Coyote-A/ultimate-upscale-for-automatic1111

![image](https://user-images.githubusercontent.com/98228077/223559884-5498d495-c5f3-4068-8711-f9f31fb2d435.png)


More advanced options for SD Upscale, less artifacts than original using higher denoise ratio (0.3-0.5).


## Model Converter
https://github.com/Akegarasu/sd-webui-model-converter

Model convert extension, supports convert fp16/bf16 no-ema/ema-only safetensors.

## Kohya-ss Additional Networks
https://github.com/kohya-ss/sd-webui-additional-networks

Allows the Web UI to use networks (LoRA) trained by their scripts to generate images. Edit safetensors prompt and additional metadata, and use 2.X LoRAs.
![image](https://user-images.githubusercontent.com/98228077/223559083-9a5dc069-f73e-48d2-a22c-4db7b983ea40.png)


## Add image number to grid
https://github.com/AlUlkesh/sd_grid_add_image_number

Add the image's number to its picture in the grid.


## quick-css
https://github.com/Gerschel/sd-web-ui-quickcss

Extension for quickly selecting and applying custom.css files, for customizing look and placement of elements in ui.

![image](https://user-images.githubusercontent.com/98228077/210076676-5f6a8e72-5352-4860-8f3d-468ab8e31355.png)![image](https://user-images.githubusercontent.com/98228077/210076407-1c904a6c-6913-4954-8f20-36100df99fba.png)


## Prompt Generator
https://github.com/imrayya/stable-diffusion-webui-Prompt_Generator

Adds a tab to the webui that allows the user to generate a prompt from a small base prompt. Based on [FredZhang7/distilgpt2-stable-diffusion-v2](https://huggingface.co/FredZhang7/distilgpt2-stable-diffusion-v2).

![image](https://user-images.githubusercontent.com/98228077/210076951-36f5d90c-b8c4-4b12-b909-582193deeec1.png)


## model-keyword
https://github.com/mix1009/model-keyword

Inserts matching keyword(s) to the prompt automatically. Update extension to get the latest model+keyword mappings.

![image](https://user-images.githubusercontent.com/98228077/209717531-e0ae74ab-b753-4ad1-99b2-e1eda3de5433.png)


## sd-model-preview
https://github.com/Vetchems/sd-model-preview

Allows you to create a txt file and jpg/png's with the same name as your model and have this info easily displayed for later reference in webui.

![image](https://user-images.githubusercontent.com/98228077/209715309-3c523945-5345-4e3d-b1a3-14f923e1bb40.png)


## Enhanced-img2img
https://github.com/OedoSoldier/enhanced-img2img

An extension with support for batched and better inpainting. See [readme](https://github.com/OedoSoldier/enhanced-img2img#usage) for more details.

![image](https://user-images.githubusercontent.com/98228077/217990537-e8bdbc74-7210-4864-8140-a076a342c695.png)


## openOutpaint extension
https://github.com/zero01101/openOutpaint-webUI-extension

A tab with the full openOutpaint UI. Run with the --api flag.

![image](https://user-images.githubusercontent.com/98228077/209453250-8bd2d766-56b5-4887-97bc-cb0e22f98dde.png)


## Save Intermediate Images
https://github.com/AlUlkesh/sd_save_intermediate_images

Implements saving intermediate images, with more advanced features.

![noisy](https://user-images.githubusercontent.com/98228077/211706803-f747691d-cca8-4692-90ef-f6a2859ed5cb.jpg)
![not](https://user-images.githubusercontent.com/98228077/211706801-fc593dbf-67c4-4983-8a80-c88355ffeba2.jpg)

![image](https://user-images.githubusercontent.com/98228077/217990312-15b4eda2-858a-44b7-91a3-b34af7c487b6.png)


## Riffusion
https://github.com/enlyth/sd-webui-riffusion

Use Riffusion model to produce music in gradio. To replicate [original](https://www.riffusion.com/about) interpolation technique, input the [prompt travel extension](https://github.com/Kahsolt/stable-diffusion-webui-prompt-travel) output frames into the riffusion tab.

![image](https://user-images.githubusercontent.com/98228077/209539460-f5c23891-b5e6-46c7-b1a5-b7440a3f031b.png)![image](https://user-images.githubusercontent.com/98228077/209539472-031e623e-f7a2-4da9-9711-8bf73d0cfe6e.png)


## DH Patch
https://github.com/d8ahazard/sd_auto_fix

Random patches by D8ahazard. Auto-load config YAML files for v2, 2.1 models; patch latent-diffusion to fix attention on 2.1 models (black boxes without no-half), whatever else I come up with.


## Preset Utilities
https://github.com/Gerschel/sd_web_ui_preset_utils

Preset tool for UI. Supports presets for some custom scripts.

![image](https://user-images.githubusercontent.com/98228077/209540881-2a870282-edb6-4c94-869b-5493cdced01f.png)


## Config-Presets
https://github.com/Zyin055/Config-Presets

Adds a configurable dropdown to allow you to change UI preset settings in the txt2img and img2img tabs.

![image](https://user-images.githubusercontent.com/98228077/208332322-24339554-0274-4add-88a7-d33bba1e3823.png)


## Diffusion Defender
https://github.com/WildBanjos/DiffusionDefender

Prompt blacklist, find and replace, for semi-private and public instances.


## NSFW checker
https://github.com/AUTOMATIC1111/stable-diffusion-webui-nsfw-censor

Replaces NSFW images with black.


## Infinity Grid Generator
https://github.com/mcmonkeyprojects/sd-infinity-grid-generator-script

Build a yaml file with your chosen parameters, and generate infinite-dimensional grids. Built-in ability to add description text to fields. See readme for usage details.

![image](https://user-images.githubusercontent.com/98228077/208332269-88983668-ea7e-45a8-a6d5-cd7a9cb64b3a.png)


## embedding-inspector
https://github.com/tkalayci71/embedding-inspector

Inspect any token(a word) or Textual-Inversion embeddings and find out which embeddings are similar. You can mix, modify, or create the embeddings in seconds. Much more intriguing options have since been released, see [here.](https://github.com/tkalayci71/embedding-inspector#whats-new)

![image](https://user-images.githubusercontent.com/98228077/209546038-3f4206bf-2c43-4d58-bf83-6318ade393f4.png)


## Prompt Gallery
https://github.com/dr413677671/PromptGallery-stable-diffusion-webui

Build a yaml file filled with prompts of your character, hit generate, and quickly preview them by their word attributes and modifiers.

![image](https://user-images.githubusercontent.com/98228077/208332199-7652146c-2428-4f44-9011-66e81bc87426.png)


## DAAM
https://github.com/toriato/stable-diffusion-webui-daam

DAAM stands for Diffusion Attentive Attribution Maps. Enter the attention text (must be a string contained in the prompt) and run. An overlapping image with a heatmap for each attention will be generated along with the original image.

![image](https://user-images.githubusercontent.com/98228077/208332173-ffb92131-bd02-4a07-9531-136822d06c86.png)


## Visualize Cross-Attention
https://github.com/benkyoujouzu/stable-diffusion-webui-visualize-cross-attention-extension

![image](https://user-images.githubusercontent.com/98228077/208332131-6acdae9a-2b25-4e71-8ab0-6e375c7c0419.png)

Generates highlighted sectors of a submitted input image, based on input prompts. Use with tokenizer extension. See the readme for more info.


## ABG_extension
https://github.com/KutsuyaYuki/ABG_extension

Automatically remove backgrounds. Uses an onnx model fine-tuned for anime images. Runs on GPU.

| ![test](https://user-images.githubusercontent.com/98228077/209423352-e8ea64e7-8522-4c2c-9350-c7cd50c35c7c.png) |  ![00035-4190733039-cow](https://user-images.githubusercontent.com/98228077/209423400-720ddde9-258a-4e68-8a93-73b67dad714e.png) | ![00021-1317075604-samdoesarts portrait](https://user-images.githubusercontent.com/98228077/209423428-da7c68db-e7d1-45a1-b931-817de9233a67.png) | ![00025-2023077221-](https://user-images.githubusercontent.com/98228077/209423446-79e676ae-460e-4282-a591-b8b986bfd869.png) |
| :---: | :---: | :---: | :---: |
| ![img_-0002-3313071906-bust shot of person](https://user-images.githubusercontent.com/98228077/209423467-789c17ad-d7ed-41a9-a039-802cfcae324a.png) | ![img_-0022-4190733039-cow](https://user-images.githubusercontent.com/98228077/209423493-dcee7860-a09e-41e0-9397-715f52bdcaab.png) | ![img_-0008-1317075604-samdoesarts portrait](https://user-images.githubusercontent.com/98228077/209423521-736f33ca-aafb-4f8b-b067-14916ec6955f.png) | ![img_-0012-2023077221-](https://user-images.githubusercontent.com/98228077/209423546-31b2305a-3159-443f-8a98-4da22c29c415.png) |


## depthmap2mask
https://github.com/Extraltodeus/depthmap2mask

Create masks for img2img based on a depth estimation made by MiDaS.

![image](https://user-images.githubusercontent.com/15731540/204050868-eca8db02-2193-4115-a5b8-e8f5c796e035.png)![image](https://user-images.githubusercontent.com/15731540/204050888-41b00335-50b4-4328-8cfd-8fc5e9cec78b.png)![image](https://user-images.githubusercontent.com/15731540/204050899-8757b774-f2da-4c15-bfa7-90d8270e8287.png)


## multi-subject-render
https://github.com/Extraltodeus/multi-subject-render

It is a depth aware extension that can help to create multiple complex subjects on a single image. It generates a background, then multiple foreground subjects, cuts their backgrounds after a depth analysis, paste them onto the background and finally does an img2img for a clean finish.

![image](https://user-images.githubusercontent.com/98228077/208331952-019dfd64-182d-4695-bdb0-4367c81e4c43.png)


## Depth Maps
https://github.com/thygate/stable-diffusion-webui-depthmap-script

Creates depthmaps from the generated images. The result can be viewed on 3D or holographic devices like VR headsets or lookingglass display, used in Render or Game- Engines on a plane with a displacement modifier, and maybe even 3D printed.

![image](https://user-images.githubusercontent.com/98228077/208331747-9acba3f0-3039-485e-96ab-f0cf5619ec3b.png)


## Merge Board
https://github.com/bbc-mc/sdweb-merge-board

Multiple lane merge support(up to 10). Save and Load your merging combination as Recipes, which is simple text.

![image](https://user-images.githubusercontent.com/98228077/208331651-09a0d70e-1906-4f80-8bc1-faf3c0ca8fad.png)

also see:\
https://github.com/Maurdekye/model-kitchen


## gelbooru-prompt
https://github.com/antis0007/sd-webui-gelbooru-prompt

Fetch tags using your image's hash.


## booru2prompt
https://github.com/Malisius/booru2prompt

This SD extension allows you to turn posts from various image boorus into stable diffusion prompts. It does so by pulling a list of tags down from their API. You can copy-paste in a link to the post you want yourself, or use the built-in search feature to do it all without leaving SD.

![image](https://user-images.githubusercontent.com/98228077/208331612-dad61ef7-33dd-4008-9cc7-06b0b0a7cb6d.png)

also see:\
https://github.com/stysmmaker/stable-diffusion-webui-booru-prompt


## WD 1.4 Tagger
https://github.com/toriato/stable-diffusion-webui-wd14-tagger

Uses a trained model file, produces WD 1.4 Tags. Model link - https://mega.nz/file/ptA2jSSB#G4INKHQG2x2pGAVQBn-yd_U5dMgevGF8YYM9CR_R1SY

![image](https://user-images.githubusercontent.com/98228077/208331569-2cf82c5c-f4c3-4181-84bd-2bdced0c2cff.png)


## DreamArtist
https://github.com/7eu7d7/DreamArtist-sd-webui-extension

Towards Controllable One-Shot Text-to-Image Generation via Contrastive Prompt-Tuning.

![image](https://user-images.githubusercontent.com/98228077/208331536-069783ae-32f7-4897-8c1b-94e0ae14f9cd.png)


## Auto TLS-HTTPS
https://github.com/papuSpartan/stable-diffusion-webui-auto-tls-https

Allows you to easily, or even completely automatically start using HTTPS.


## Randomize
~~https://github.com/stysmmaker/stable-diffusion-webui-randomize~~
fork: https://github.com/innightwolfsleep/stable-diffusion-webui-randomize

Allows for random parameters during txt2img generation. This script is processed for all generations, regardless of the script selected, meaning this script will function with others as well, such as AUTOMATIC1111/stable-diffusion-webui-wildcards.


## conditioning-highres-fix
https://github.com/klimaleksus/stable-diffusion-webui-conditioning-highres-fix

This is Extension for rewriting Inpainting conditioning mask strength value relative to Denoising strength at runtime. This is useful for Inpainting models such as sd-v1-5-inpainting.ckpt

![image](https://user-images.githubusercontent.com/98228077/208331374-5a271cf3-cfac-449b-9e09-c63ddc9ca03a.png)


## Detection Detailer
https://github.com/dustysys/ddetailer

An object detection and auto-mask extension for Stable Diffusion web UI.

<img src="https://github.com/dustysys/ddetailer/raw/master/misc/ddetailer_example_3.gif"/>


## Sonar
https://github.com/Kahsolt/stable-diffusion-webui-sonar

Improve the generated image quality, searches for similar (yet even better!) images in the neighborhood of some known image, focuses on single prompt optimization rather than traveling between multiple prompts.

![image](https://user-images.githubusercontent.com/98228077/209545702-c796a3f8-4d8c-4e2b-9b2e-920008ec2f32.png)![image](https://user-images.githubusercontent.com/98228077/209545756-31c94fec-d783-447f-8aac-4a5bba43ea15.png)


## prompt travel
https://github.com/Kahsolt/stable-diffusion-webui-prompt-travel

Extension script for AUTOMATIC1111/stable-diffusion-webui to travel between prompts in latent space.

<details><summary>Example: (Click to expand:)</summary>
<img src="https://github.com/ClashSAN/bloated-gifs/blob/main/prompt_travel.gif" width="512" height="512" />
</details>


## shift-attention
https://github.com/yownas/shift-attention

Generate a sequence of images shifting attention in the prompt. This script enables you to give a range to the weight of tokens in a prompt and then generate a sequence of images stepping from the first one to the second.

https://user-images.githubusercontent.com/13150150/193368939-c0a57440-1955-417c-898a-ccd102e207a5.mp4


## seed travel
https://github.com/yownas/seed_travel

Small script for AUTOMATIC1111/stable-diffusion-webui to create images that exists between seeds.

<details><summary>Example: (Click to expand:)</summary>
<img src="https://github.com/ClashSAN/bloated-gifs/blob/main/seedtravel.gif" width="512" height="512" />
</details>


## Embeddings editor
https://github.com/CodeExplode/stable-diffusion-webui-embedding-editor

Allows you to manually edit textual inversion embeddings using sliders.

![image](https://user-images.githubusercontent.com/98228077/208331138-cdfe8f43-78f7-499e-b746-c42355ee8d6d.png)


## Latent Mirroring
https://github.com/dfaker/SD-latent-mirroring

Applies mirroring and flips to the latent images to produce anything from subtle balanced compositions to perfect reflections

![image](https://user-images.githubusercontent.com/98228077/208331098-3b7fefce-6d38-486d-9543-258f5b2b0fd6.png)


## StylePile
https://github.com/some9000/StylePile
			
An easy way to mix and match elements to prompts that affect the style of the result.

![image](https://user-images.githubusercontent.com/98228077/208331056-2956d050-a7a4-4b6f-b064-72f6a7d7ee0d.png)


## Push to 🤗 Hugging Face

https://github.com/camenduru/stable-diffusion-webui-huggingface

![Push Folder to Hugging Face](https://user-images.githubusercontent.com/54370274/206897701-9e86ce7c-af06-4d95-b9ea-385276c99d3a.jpg)

To install it, clone the repo into the `extensions` directory and restart the web ui:

`git clone https://github.com/camenduru/stable-diffusion-webui-huggingface`

`pip install huggingface-hub`


## Tokenizer
https://github.com/AUTOMATIC1111/stable-diffusion-webui-tokenizer

Adds a tab that lets you preview how CLIP model would tokenize your text.

![about](https://user-images.githubusercontent.com/20920490/200113798-50b55f5a-45db-4b6f-93c0-ae6be75e5788.png)


## novelai-2-local-prompt
https://github.com/animerl/novelai-2-local-prompt

Add a button to convert the prompts used in NovelAI for use in the WebUI. In addition, add a button that allows you to recall a previously used prompt.

![pic](https://user-images.githubusercontent.com/113022648/197382468-65f4a96d-48af-4890-8fcf-0ec7c3b9ec3a.png)


## Booru tag autocompletion
https://github.com/DominikDoom/a1111-sd-webui-tagcomplete

Displays autocompletion hints for tags from "image booru" boards such as Danbooru. Uses local tag CSV files and includes a config for customization.

![image](https://user-images.githubusercontent.com/20920490/200016417-9451efdb-5d0d-4131-bd9e-39a687be8dd7.png)


## Unprompted
https://github.com/ThereforeGames/unprompted
 
Supercharge your prompt workflow with this powerful scripting language!

![unprompted_header](https://user-images.githubusercontent.com/95403634/199041569-7c6c5748-e7dc-4068-943f-c2d92745dbb5.png)

**Unprompted** is a highly modular extension for [AUTOMATIC1111's Stable Diffusion Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) that allows you to include various shortcodes in your prompts. You can pull text from files, set up your own variables, process text through conditional functions, and so much more - it's like wildcards on steroids.

While the intended usecase is Stable Diffusion, **this engine is also flexible enough to serve as an all-purpose text generator.**


## training-picker
https://github.com/Maurdekye/training-picker

Adds a tab to the webui that allows the user to automatically extract keyframes from video, and manually extract 512x512 crops of those frames for use in model training.

![image](https://user-images.githubusercontent.com/2313721/199614791-1f573573-a2e2-4358-836d-5655825077e1.png)

**Installation**

- Install [AUTOMATIC1111's Stable Diffusion Webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
- Install [ffmpeg](https://ffmpeg.org/) for your operating system
- Clone this repository into the extensions folder inside the webui
- Drop videos you want to extract cropped frames from into the training-picker/videos folder


## auto-sd-paint-ext

https://github.com/Interpause/auto-sd-paint-ext

>Extension for AUTOMATIC1111's webUI with Krita Plugin (other drawing studios soon?)

![image](https://user-images.githubusercontent.com/98228077/217986983-d23a334d-50bc-4bb1-ac8b-60f99a3b07b9.png)


- Optimized workflow (txt2img, img2img, inpaint, upscale) & UI design.
- Only drawing studio plugin that exposes the Script API.

See https://github.com/Interpause/auto-sd-paint-ext/issues/41 for planned developments.
See [CHANGELOG.md](https://github.com/Interpause/auto-sd-paint-ext/blob/main/CHANGELOG.md) for the full changelog.


## Dataset Tag Editor
https://github.com/toshiaki1729/stable-diffusion-webui-dataset-tag-editor

[日本語 Readme](https://github.com/toshiaki1729/stable-diffusion-webui-dataset-tag-editor/blob/main/README-JP.md)

This is an extension to edit captions in training dataset for [Stable Diffusion web UI by AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui).

It works well with text captions in comma-separated style (such as the tags generated by DeepBooru interrogator).

Caption in the filenames of images can be loaded, but edited captions can only be saved in the form of text files.

![picture](https://github.com/toshiaki1729/stable-diffusion-webui-dataset-tag-editor/blob/main/ss01.png)


## Aesthetic Image Scorer
https://github.com/tsngo/stable-diffusion-webui-aesthetic-image-scorer

Extension for https://github.com/AUTOMATIC1111/stable-diffusion-webui

Calculates aesthetic score for generated images using [CLIP+MLP Aesthetic Score Predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor) based on [Chad Scorer](https://github.com/grexzen/SD-Chad/blob/main/chad_scorer.py)

See [Discussions](https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/1831)

Saves score to windows tags with other options planned

![picture](https://github.com/tsngo/stable-diffusion-webui-aesthetic-image-scorer/blob/main/tag_group_by.png)


## Artists to study
https://github.com/camenduru/stable-diffusion-webui-artists-to-study

https://artiststostudy.pages.dev/ adapted to an extension for [web ui](https://github.com/AUTOMATIC1111/stable-diffusion-webui).

To install it, clone the repo into the `extensions` directory and restart the web ui:

`git clone https://github.com/camenduru/stable-diffusion-webui-artists-to-study`

You can add the artist name to the clipboard by clicking on it. (thanks for the idea @gmaciocci)

![picture](https://user-images.githubusercontent.com/54370274/197829512-e7d30d44-2697-4ecd-b9a7-3665217918c7.jpg)


## Deforum
https://github.com/deforum-art/deforum-for-automatic1111-webui


The official port of Deforum, an extensive script for 2D and 3D animations, supporting keyframable sequences, dynamic math parameters (even inside the prompts), dynamic masking, depth estimation and warping.

![image](https://user-images.githubusercontent.com/98228077/217986819-67fd1e3c-b007-475c-b8c9-3bf28ee3aa67.png)


## Inspiration
https://github.com/yfszzx/stable-diffusion-webui-inspiration

Randomly display the pictures of the artist's or artistic genres typical style, more pictures of this artist or genre is displayed after selecting. So you don't have to worry about how hard it is to choose the right style of art when you create.

![68747470733a2f2f73362e6a70672e636d2f323032322f31302f32322f504a596f4e4c2e706e67](https://user-images.githubusercontent.com/20920490/197518700-3f753132-8799-4ad0-8cdf-bcdcbf7798aa.png)


## Image Browser
https://github.com/AlUlkesh/stable-diffusion-webui-images-browser

Provides an interface to browse created images in the web browser, allows for sorting and filtering by EXIF data.

![image](https://user-images.githubusercontent.com/23466035/217083703-0845da05-3305-4f5a-af53-f3829de6a29d.png)


## Smart Process
https://github.com/d8ahazard/sd_smartprocess

Intelligent cropping, captioning, and image enhancement.

![image](https://user-images.githubusercontent.com/1633844/201435094-433d765c-56e8-4573-82d9-71af2b112159.png)


## Dreambooth
https://github.com/d8ahazard/sd_dreambooth_extension

Dreambooth in the UI. Refer to the project readme for tuning and configuration requirements. Includes [LoRA](https://github.com/cloneofsimo/lora) (Low Rank Adaptation)

Based on ShivamShiaro's repo.

![image](https://user-images.githubusercontent.com/1633844/201434706-2c2744ba-082e-427e-9f8d-af03de204583.png)


## Dynamic Prompts
https://github.com/adieyal/sd-dynamic-prompts

A custom extension for [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) that implements an expressive template language for random or combinatorial prompt generation along with features to support deep wildcard directory structures.

More features and additions are shown in the [readme](https://github.com/adieyal/sd-dynamic-prompts).

![image](https://github.com/adieyal/sd-dynamic-prompts/raw/main/images/extension.png)

Using this extension, the prompt:

`A {house|apartment|lodge|cottage} in {summer|winter|autumn|spring} by {2$$artist1|artist2|artist3}`

Will any of the following prompts:

- A house in summer by artist1, artist2
- A lodge in autumn by artist3, artist1
- A cottage in winter by artist2, artist3
- ...

This is especially useful if you are searching for interesting combinations of artists and styles.

You can also pick a random string from a file. Assuming you have the file seasons.txt in WILDCARD_DIR (see below), then:

`__seasons__ is coming`

Might generate the following:

- Winter is coming
- Spring is coming
- ...

You can also use the same wildcard twice

`I love __seasons__ better than __seasons__`

- I love Winter better than Summer
- I love Spring better than Spring

## Wildcards
https://github.com/AUTOMATIC1111/stable-diffusion-webui-wildcards

Allows you to use `__name__` syntax in your prompt to get a random line from a file named `name.txt` in the wildcards directory.

## Aesthetic Gradients
https://github.com/AUTOMATIC1111/stable-diffusion-webui-aesthetic-gradients

Create an embedding from one or few pictures and use it to apply their style to generated images.

![firefox_FgKg9dx9eF](https://user-images.githubusercontent.com/20920490/197466300-6b042bcf-5cba-4600-97d7-ad2652875706.png)

## 3D Model&Pose Loader
https://github.com/jtydhr88/sd-3dmodel-loader

A custom extension that allows you to load your local 3D model/animation inside webui, or edit pose as well, then send screenshot to txt2img or img2img as your ControlNet's reference image.

![1](https://user-images.githubusercontent.com/860985/236643711-140579dc-bb74-4a84-ba40-3a64823285ab.png)

## Canvas Editor
https://github.com/jtydhr88/sd-canvas-editor

A custom extension for sd-webui that integrated a full capability canvas editor which you can use layer, text, image, elements, etc.

![overall](https://user-images.githubusercontent.com/860985/236643824-946ebdef-83af-4e85-80ea-6ac7d2fb520e.png)