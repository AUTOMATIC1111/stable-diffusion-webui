# General info

Extensions are a more convenient form of user scripts.

Extensions all exist in their own subdirectory inside the `extensions` directory. You can use git to install an extension like this:

```
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui-aesthetic-gradients extensions/aesthetic-gradients
```

This installs an extension from `https://github.com/AUTOMATIC1111/stable-diffusion-webui-aesthetic-gradients` into the `extensions/aesthetic-gradients` directory.

Alternatively you can just copy-paste a directory into `extensions`.

For developing extensions, see [Developing extensions](Developing-extensions).

# Security

As extensions allow the user to install and run arbitrary code, this can be used maliciously, and is disabled by default when running with options that allow remote users to connect to the server (`--share` or `--listen`) - you'll still have the UI, but trying to install anything will result in error. If you want to use those options and still be able to install extensions, use `--enable-insecure-extension-access` command line flag.

# Extensions

## Aesthetic Gradients
https://github.com/AUTOMATIC1111/stable-diffusion-webui-aesthetic-gradients

Create an embedding from one or few pictures and use it to apply their style to generated images.

![firefox_FgKg9dx9eF](https://user-images.githubusercontent.com/20920490/197466300-6b042bcf-5cba-4600-97d7-ad2652875706.png)

## Wildcards
https://github.com/AUTOMATIC1111/stable-diffusion-webui-wildcards

Allows you to use `__name__` syntax in your prompt to get a random line from a file named `name.txt` in the wildcards directory.

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

## Dreambooth
https://github.com/d8ahazard/sd_dreambooth_extension

Dreambooth in the UI. Refer to the project readme for tuning and configuration requirements.

Based on ShivamShiaro's repo.

![image](https://user-images.githubusercontent.com/1633844/201434706-2c2744ba-082e-427e-9f8d-af03de204583.png)


## Smart Process
https://github.com/d8ahazard/sd_smartprocess

Intelligent cropping, captioning, and image enhancement.

![image](https://user-images.githubusercontent.com/1633844/201435094-433d765c-56e8-4573-82d9-71af2b112159.png)


## Image browser
https://github.com/yfszzx/stable-diffusion-webui-images-browser

Provides an interface to browse created images in the web browser.

![68747470733a2f2f73362e6a70672e636d2f323032322f31302f32342f504a6a755a742e706e67](https://user-images.githubusercontent.com/20920490/197518762-a23f3e34-f174-4275-8283-eb8d2ff65ef2.png)

## Inspiration
https://github.com/yfszzx/stable-diffusion-webui-inspiration

Randomly display the pictures of the artist's or artistic genres typical style, more pictures of this artist or genre is displayed after selecting. So you don't have to worry about how hard it is to choose the right style of art when you create.

![68747470733a2f2f73362e6a70672e636d2f323032322f31302f32322f504a596f4e4c2e706e67](https://user-images.githubusercontent.com/20920490/197518700-3f753132-8799-4ad0-8cdf-bcdcbf7798aa.png)

## Deforum
https://github.com/deforum-art/deforum-for-automatic1111-webui


The official port of Deforum, an extensive script for 2D and 3D animations, supporting keyframable sequences, dynamic math parameters (even inside the prompts), dynamic masking, depth estimation and warping.

![ui](https://user-images.githubusercontent.com/20920490/197619558-c088a329-3672-4f0a-8685-cf539996ad1e.png)

## Artists to study
https://github.com/camenduru/stable-diffusion-webui-artists-to-study

https://artiststostudy.pages.dev/ adapted to an extension for [web ui](https://github.com/AUTOMATIC1111/stable-diffusion-webui).

To install it, clone the repo into the `extensions` directory and restart the web ui:

`git clone https://github.com/camenduru/stable-diffusion-webui-artists-to-study`

You can add the artist name to the clipboard by clicking on it. (thanks for the idea @gmaciocci)

![picture](https://user-images.githubusercontent.com/54370274/197829512-e7d30d44-2697-4ecd-b9a7-3665217918c7.jpg)

## Aesthetic Image Scorer
https://github.com/tsngo/stable-diffusion-webui-aesthetic-image-scorer

Extension for https://github.com/AUTOMATIC1111/stable-diffusion-webui

Calculates aesthetic score for generated images using [CLIP+MLP Aesthetic Score Predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor) based on [Chad Scorer](https://github.com/grexzen/SD-Chad/blob/main/chad_scorer.py)

See [Discussions](https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/1831)

Saves score to windows tags with other options planned

![picture](https://github.com/tsngo/stable-diffusion-webui-aesthetic-image-scorer/blob/main/tag_group_by.png)

## Dataset Tag Editor
https://github.com/toshiaki1729/stable-diffusion-webui-dataset-tag-editor

[日本語 Readme](https://github.com/toshiaki1729/stable-diffusion-webui-dataset-tag-editor/blob/main/README-JP.md)

This is an extension to edit captions in training dataset for [Stable Diffusion web UI by AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui).

It works well with text captions in comma-separated style (such as the tags generated by DeepBooru interrogator).

Caption in the filenames of images can be loaded, but edited captions can only be saved in the form of text files.

![picture](https://github.com/toshiaki1729/stable-diffusion-webui-dataset-tag-editor/blob/main/ss01.png)

## auto-sd-paint-ext

https://github.com/Interpause/auto-sd-paint-ext

Formerly known as `auto-sd-krita`.

>Extension for AUTOMATIC1111's webUI with Krita Plugin (other drawing studios soon?)

Outdated demo | New UI (TODO: demo image)
--- | ---
![demo image](https://user-images.githubusercontent.com/42513874/194701722-e7a3f7eb-be4a-4f43-93a5-480835c9260f.jpg) | ![demo image 2](https://user-images.githubusercontent.com/42513874/199507299-66729f9b-3581-43a3-b5f4-57eb90b8f981.png)

**Differences**

- UI no longer freezes during image update
- Inpainting layer no longer has to be manually hidden, nor use white specifically
- UI has been improved & squeezed further
- Scripts API is now possible

## training-picker
https://github.com/Maurdekye/training-picker

Adds a tab to the webui that allows the user to automatically extract keyframes from video, and manually extract 512x512 crops of those frames for use in model training.

![image](https://user-images.githubusercontent.com/2313721/199614791-1f573573-a2e2-4358-836d-5655825077e1.png)

**Installation**

- Install [AUTOMATIC1111's Stable Diffusion Webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
- Install [ffmpeg](https://ffmpeg.org/) for your operating system
- Clone this repository into the extensions folder inside the webui
- Drop videos you want to extract cropped frames from into the training-picker/videos folder

## Unprompted
https://github.com/ThereforeGames/unprompted
 
Supercharge your prompt workflow with this powerful scripting language!

![unprompted_header](https://user-images.githubusercontent.com/95403634/199041569-7c6c5748-e7dc-4068-943f-c2d92745dbb5.png)

**Unprompted** is a highly modular extension for [AUTOMATIC1111's Stable Diffusion Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) that allows you to include various shortcodes in your prompts. You can pull text from files, set up your own variables, process text through conditional functions, and so much more - it's like wildcards on steroids.

While the intended usecase is Stable Diffusion, **this engine is also flexible enough to serve as an all-purpose text generator.**

## Booru tag autocompletion
https://github.com/DominikDoom/a1111-sd-webui-tagcomplete

Displays autocompletion hints for tags from "image booru" boards such as Danbooru. Uses local tag CSV files and includes a config for customization.

![image](https://user-images.githubusercontent.com/20920490/200016417-9451efdb-5d0d-4131-bd9e-39a687be8dd7.png)

## novelai-2-local-prompt
https://github.com/animerl/novelai-2-local-prompt

Add a button to convert the prompts used in NovelAI for use in the WebUI. In addition, add a button that allows you to recall a previously used prompt.

![pic](https://user-images.githubusercontent.com/113022648/197382468-65f4a96d-48af-4890-8fcf-0ec7c3b9ec3a.png)

## Tokenizer
https://github.com/AUTOMATIC1111/stable-diffusion-webui-tokenizer

Adds a tab that lets you preview how CLIP model would tokenize your text.

![about](https://user-images.githubusercontent.com/20920490/200113798-50b55f5a-45db-4b6f-93c0-ae6be75e5788.png)

## Push to 🤗 Hugging Face

https://github.com/camenduru/stable-diffusion-webui-huggingface

![Push Folder to Hugging Face](https://user-images.githubusercontent.com/54370274/206897701-9e86ce7c-af06-4d95-b9ea-385276c99d3a.jpg)

To install it, clone the repo into the `extensions` directory and restart the web ui:

`git clone https://github.com/camenduru/stable-diffusion-webui-huggingface`

`pip install huggingface-hub`

## StylePile
https://github.com/some9000/StylePile
			
An easy way to mix and match elements to prompts that affect the style of the result.

![image](https://user-images.githubusercontent.com/98228077/208331056-2956d050-a7a4-4b6f-b064-72f6a7d7ee0d.png)

## Latent Mirroring
https://github.com/dfaker/SD-latent-mirroring

Applies mirroring and flips to the latent images to produce anything from subtle balanced compositions to perfect reflections

![image](https://user-images.githubusercontent.com/98228077/208331098-3b7fefce-6d38-486d-9543-258f5b2b0fd6.png)

## Embeddings editor
https://github.com/CodeExplode/stable-diffusion-webui-embedding-editor

Allows you to manually edit textual inversion embeddings using sliders.

![image](https://user-images.githubusercontent.com/98228077/208331138-cdfe8f43-78f7-499e-b746-c42355ee8d6d.png)

## seed travel
https://github.com/yownas/seed_travel.git

Small script for AUTOMATIC1111/stable-diffusion-webui to create images that exists between seeds.


## shift-attention
https://github.com/yownas/shift-attention.git

Generate a sequence of images shifting attention in the prompt. This script enables you to give a range to the weight of tokens in a prompt and then generate a sequence of images stepping from the first one to the second.
## prompt travel
https://github.com/Kahsolt/stable-diffusion-webui-prompt-travel

Extension script for AUTOMATIC1111/stable-diffusion-webui to travel between prompts in latent space.

## Detection Detailer
https://github.com/dustysys/ddetailer

An object detection and auto-mask extension for Stable Diffusion web UI.

![image](https://user-images.githubusercontent.com/98228077/208331343-eb272451-ef41-4bbd-90e5-ed27d0a59769.png)

## conditioning-highres-fix
https://github.com/klimaleksus/stable-diffusion-webui-conditioning-highres-fix

This is Extension for rewriting Inpainting conditioning mask strength value relative to Denoising strength at runtime. This is useful for Inpainting models such as sd-v1-5-inpainting.ckpt

![image](https://user-images.githubusercontent.com/98228077/208331374-5a271cf3-cfac-449b-9e09-c63ddc9ca03a.png)

## Randomize
https://github.com/stysmmaker/stable-diffusion-webui-randomize

Allows for random parameters during txt2img generation. This script is processed for all generations, regardless of the script selected, meaning this script will function with others as well, such as AUTOMATIC1111/stable-diffusion-webui-wildcards.
## Auto TLS-HTTPS
https://github.com/papuSpartan/stable-diffusion-webui-auto-tls-https

Allows you to easily, or even completely automatically start using HTTPS.
## DreamArtist
https://github.com/7eu7d7/DreamArtist-sd-webui-extension

Towards Controllable One-Shot Text-to-Image Generation via Contrastive Prompt-Tuning.

![image](https://user-images.githubusercontent.com/98228077/208331536-069783ae-32f7-4897-8c1b-94e0ae14f9cd.png)

## WD 1.4 Tagger
https://github.com/toriato/stable-diffusion-webui-wd14-tagger

Uses a trained model file, produces WD 1.4 Tags. Model link - https://mega.nz/file/ptA2jSSB#G4INKHQG2x2pGAVQBn-yd_U5dMgevGF8YYM9CR_R1SY

![image](https://user-images.githubusercontent.com/98228077/208331569-2cf82c5c-f4c3-4181-84bd-2bdced0c2cff.png)

## booru2prompt
https://github.com/Malisius/booru2prompt

This SD extension allows you to turn posts from various image boorus into stable diffusion prompts. It does so by pulling a list of tags down from their API. You can copy-paste in a link to the post you want yourself, or use the built-in search feature to do it all without leaving SD.

![image](https://user-images.githubusercontent.com/98228077/208331612-dad61ef7-33dd-4008-9cc7-06b0b0a7cb6d.png)

also see:\
https://github.com/stysmmaker/stable-diffusion-webui-booru-prompt

## gelbooru-prompt
https://github.com/antis0007/sd-webui-gelbooru-prompt

Fetch tags with image hash. Updates planned.

## Merge Board
https://github.com/bbc-mc/sdweb-merge-board

Multiple lane merge support(up to 10). Save and Load your merging combination as Recipes, which is simple text.

![image](https://user-images.githubusercontent.com/98228077/208331651-09a0d70e-1906-4f80-8bc1-faf3c0ca8fad.png)

also see:\
https://github.com/Maurdekye/model-kitchen
## Depth Maps
https://github.com/thygate/stable-diffusion-webui-depthmap-script

Creates depthmaps from the generated images. The result can be viewed on 3D or holographic devices like VR headsets or lookingglass display, used in Render or Game- Engines on a plane with a displacement modifier, and maybe even 3D printed.

![image](https://user-images.githubusercontent.com/98228077/208331747-9acba3f0-3039-485e-96ab-f0cf5619ec3b.png)

## multi-subject-render
https://github.com/Extraltodeus/multi-subject-render

It is a depth aware extension that can help to create multiple complex subjects on a single image. It generates a background, then multiple foreground subjects, cuts their backgrounds after a depth analysis, paste them onto the background and finally does an img2img for a clean finish.

![image](https://user-images.githubusercontent.com/98228077/208331952-019dfd64-182d-4695-bdb0-4367c81e4c43.png)

## depthmap2mask
https://github.com/Extraltodeus/depthmap2mask

Create masks for img2img based on a depth estimation made by MiDaS.
## Visualize Cross-Attention
https://github.com/benkyoujouzu/stable-diffusion-webui-visualize-cross-attention-extension

![image](https://user-images.githubusercontent.com/98228077/208332131-6acdae9a-2b25-4e71-8ab0-6e375c7c0419.png)

Generates highlighted sectors of a submitted input image, based on input prompts. Use with tokenizer extension. See the readme for more info.
## DAAM
https://github.com/kousw/stable-diffusion-webui-daam

DAAM stands for Diffusion Attentive Attribution Maps. Enter the attention text (must be a string contained in the prompt) and run. An overlapping image with a heatmap for each attention will be generated along with the original image.

![image](https://user-images.githubusercontent.com/98228077/208332173-ffb92131-bd02-4a07-9531-136822d06c86.png)

## Prompt Gallery
https://github.com/dr413677671/PromptGallery-stable-diffusion-webui

Build a yaml file filled with prompts of your character, hit generate, and quickly preview them by their word attributes and modifiers.

![image](https://user-images.githubusercontent.com/98228077/208332199-7652146c-2428-4f44-9011-66e81bc87426.png)

## embedding-inspector
https://github.com/tkalayci71/embedding-inspector

Inspect any token(a word) or Textual-Inversion embeddings and find out which embeddings are similar. You can mix, modify, or create the embeddings in seconds.
## Infinity Grid Generator
https://github.com/mcmonkeyprojects/sd-infinity-grid-generator-script

Build a yaml file with your chosen parameters, and generate infinite-dimensional grids. Built-in ability to add description text to fields. See readme for usage details.

![image](https://user-images.githubusercontent.com/98228077/208332269-88983668-ea7e-45a8-a6d5-cd7a9cb64b3a.png)

## NSFW checker
https://github.com/AUTOMATIC1111/stable-diffusion-webui-nsfw-censor

Replaces NSFW images with black.
## Config-Presets
https://github.com/Zyin055/Config-Presets

Adds a configurable dropdown to allow you to change UI preset settings in the txt2img and img2img tabs.

![image](https://user-images.githubusercontent.com/98228077/208332322-24339554-0274-4add-88a7-d33bba1e3823.png)

## Preset Utilities
https://github.com/Gerschel/sd_web_ui_preset_utils

Preset tool for UI. Planned support for some other custom scripts.

## DH Patch
https://github.com/d8ahazard/sd_auto_fix

Random patches by D8ahazard. Auto-load config YAML files for v2, 2.1 models; patch latent-diffusion to fix attention on 2.1 models (black boxes without no-half), whatever else I come up with.

## Riffusion
https://github.com/enlyth/sd-webui-riffusion

Use Riffusion model to produce music in gradio, see readme for interpolation technique.

## Enhanced-img2img
https://github.com/OedoSoldier/enhanced-img2img

An extension with support for batched and better inpainting.

## WIP
https://github.com/sdwebui-w-horde/sd-webui-stable-horde-worker
https://github.com/space-nuko/sd-promptbook

## Test
https://github.com/benkyoujouzu/stable-diffusion-webui-print-processed-prompt-extension
https://github.com/mix1009/model-keyword
https://github.com/bbc-mc/merge-percentage-visualize
https://github.com/152334H/sd-webui-whisper
https://github.com/bbc-mc/sdweb-xyplus
https://github.com/zero01101/openOutpaint-webUI-extension
https://github.com/Merramore/stable-diffusion-webui-extension-DirectML
https://github.com/64617/pony-diffusion-webui
https://github.com/Le-Chat-BottE/file-download-extension
https://github.com/stysmmaker/stable-diffusion-webui-send-to-hydrus
https://github.com/sayoonarachu/sd_prompt_gen
https://github.com/liuhongboCM/stable-diffusion-webui-generate-forever https://github.com/LunaOn/stable-diffusion-webui-generate-forever
https://github.com/CodeExplode/stable-diffusion-webui-finetuning-tokens
https://github.com/bbc-mc/uiTweaks_txt2img_prompt-controll-btn
https://github.com/auto-webui-extensions-and-scripts/auto-backup

https://github.com/mayurdhagiya/Smart-Toilet