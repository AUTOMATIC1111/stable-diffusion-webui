# Generation Scripts

## txt2img2img 
https://github.com/ThereforeGames/txt2img2img

Greatly improve the editability of any character/subject while retaining their likeness. The main motivation for this script is improving the editability of embeddings created through [Textual Inversion](https://textual-inversion.github.io/).

(be careful with cloning as it has a bit of venv checked in)

<details><summary>Example: (Click to expand:)</summary>
<img src="https://user-images.githubusercontent.com/98228077/200106431-21a22657-db24-4e9c-b7fa-e3a8e9096b89.png" width="624" height="312" />
</details>

## txt2mask
https://github.com/ThereforeGames/txt2mask

Allows you to specify an inpainting mask with text, as opposed to the brush.

<details><summary>Example: (Click to expand:)</summary>
<img src="https://user-images.githubusercontent.com/95403634/190878562-d020887c-ccb0-411c-ab37-38e2115552eb.png" width="674" height="312" />
</details>

## Mask drawing UI
https://github.com/dfaker/stable-diffusion-webui-cv2-external-masking-script

Provides a local popup window powered by CV2 that allows addition of a mask before processing.

<details><summary>Example: (Click to expand:)</summary>
<img src="https://user-images.githubusercontent.com/98228077/200109495-3d6741f1-0e25-4ae5-9f84-d93f886f302a.png" width="302" height="312" />
</details>

## Img2img Video
https://github.com/memes-forever/Stable-diffusion-webui-video

Using img2img, generates pictures one after another.

## Advanced Seed Blending
https://github.com/amotile/stable-diffusion-backend/tree/master/src/process/implementations/automatic1111_scripts

This script allows you to base the initial noise on multiple weighted seeds.

Ex. `seed1:2, seed2:1, seed3:1`

The weights are normalized so you can use bigger once like above, or you can do floating point numbers:

Ex. `seed1:0.5, seed2:0.25, seed3:0.25`

## Prompt Blending
https://github.com/amotile/stable-diffusion-backend/tree/master/src/process/implementations/automatic1111_scripts

This script allows you to combine multiple weighted prompts together by mathematically combining their textual embeddings before generating the image.

Ex.

`Crystal containing elemental {fire|ice}`

It supports nested definitions so you can do this as well:

`Crystal containing elemental {{fire:5|ice}|earth}`

## Animator
https://github.com/Animator-Anon/Animator

A basic img2img script that will dump frames and build a video file. Suitable for creating interesting zoom in warping movies, but not too much else at this time.

## Alternate Noise Schedules
https://gist.github.com/dfaker/f88aa62e3a14b559fe4e5f6b345db664

Uses alternate generators for the sampler's sigma schedule.

Allows access to Karras, Exponential and Variance Preserving schedules from crowsonkb/k-diffusion along with their parameters.

## Vid2Vid
https://github.com/Filarius/stable-diffusion-webui/blob/master/scripts/vid2vid.py

From real video, img2img the frames and stitch them together. Does not unpack frames to hard drive.

## Video Loopback
https://github.com/fishslot/video_loopback_for_webui

A video2video script that tries to improve on the temporal consistency and flexibility of normal vid2vid.
Works with any SD model without finetune, but better with a LoRA or DreamBooth for your specified character.

## Force Symmetry
https://gist.github.com/1ort/2fe6214cf1abe4c07087aac8d91d0d8a

see https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/2441

applies symmetry to the image every n steps and sends the result further to img2img.

<details><summary>Example: (Click to expand:)</summary>
<img src="https://user-images.githubusercontent.com/83316072/196016119-0a03664b-c3e4-49f0-81ac-a9e719b24bd1.png" width="624" height="312" />
</details>

## txt2palette
https://github.com/1ort/txt2palette

Generate palettes by text description. This script takes the generated images and converts them into color palettes.

<details><summary>Example: (Click to expand:)</summary>
<img src="https://user-images.githubusercontent.com/83316072/199360686-62f0f5ec-ed3d-4c0f-95b4-af9c67d1e248.png" width="352" height="312" />
</details>

## img2tiles
https://github.com/arcanite24/img2tiles

generate tiles from a base image. Based on SD upscale script.

<details><summary>Example: (Click to expand:)</summary>
<img src="https://github.com/arcanite24/img2tiles/raw/master/examples/example5.png" width="312" height="312" />
</details>

## img2mosiac
https://github.com/1ort/img2mosaic

Generate mosaics from images. The script cuts the image into tiles and processes each tile separately. The size of each tile is chosen randomly.

<details><summary>Example: (Click to expand:)</summary>
<img src="https://user-images.githubusercontent.com/83316072/200170569-0e7131e4-1da8-4caf-9cd9-5b785c9d21b0.png" width="758" height="312" />
</details>

## gif2gif
https://github.com/LonicaMewinsky/gif2gif

The purpose of this script is to accept an animated gif as input, process frames as img2img typically would, and recombine them back into an animated gif. Not intended to have extensive functionality. Referenced code from prompts_from_file.

## Saving steps of the sampling process

This script will save steps of the sampling process to a directory.
```python
import os.path

import modules.scripts as scripts
import gradio as gr

from modules import sd_samplers, shared
from modules.processing import Processed, process_images


class Script(scripts.Script):
    def title(self):
        return "Save steps of the sampling process to files"

    def ui(self, is_img2img):
        path = gr.Textbox(label="Save images to path")
        return [path]

    def run(self, p, path):
        index = [0]

        def store_latent(x):
            image = shared.state.current_image = sd_samplers.sample_to_image(x)
            image.save(os.path.join(path, f"sample-{index[0]:05}.png"))
            index[0] += 1
            fun(x)

        fun = sd_samplers.store_latent
        sd_samplers.store_latent = store_latent

        try:
            proc = process_images(p)
        finally:
            sd_samplers.store_latent = fun

        return Processed(p, proc.images, p.seed, "")
```
