# Installing and Using Custom Scripts
To install custom scripts, place them into the `scripts` directory and click the `Reload custom script` button at the bottom in the settings tab. Custom scripts will appear in the lower-left dropdown menu on the txt2img and img2img tabs after being installed. Below are some notable custom scripts created by Web UI users:

# Custom Scripts from Users

## Advanced prompt matrix
https://github.com/GRMrGecko/stable-diffusion-webui-automatic/blob/advanced_matrix/scripts/advanced_prompt_matrix.py

It allows a matrix prompt as follows:
`<cyber|cyborg|> cat <photo|image|artistic photo|oil painting> in a <car|boat|cyber city>`

Does not actually draw a matrix, just produces pictures.

## Wildcards
This and the more feature rich dynamic prompts script have been turned into [extensions.](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Extensions)

## txt2img2img 
https://github.com/ThereforeGames/txt2img2img/blob/main/scripts/txt2img2img.py

Greatly improve the editability of any character/subject while retaining their likeness.

Full description in original repo: https://github.com/ThereforeGames/txt2img2img (be careful with cloning as it has a bit of venv checked in)

## txt2mask
https://github.com/ThereforeGames/txt2mask

Allows you to specify an inpainting mask with text, as opposed to the brush.

## Mask drawing UI
https://github.com/dfaker/stable-diffusion-webui-cv2-external-masking-script/blob/main/external_masking.py

Provides a local popup window powered by CV2 that allows addition of a mask before processing. [Readme](https://github.com/dfaker/stable-diffusion-webui-cv2-external-masking-script).

## Img2img Video
https://github.com/memes-forever/Stable-diffusion-webui-video
Using img2img, generates pictures one after another.

## Seed Travel
https://github.com/yownas/seed_travel

Pick two (or more) seeds and generate a sequence of images interpolating between them. Optionally, let it create a video of the result.

Example of what you can do with it:
https://www.youtube.com/watch?v=4c71iUclY4U

## Animator
https://github.com/Animator-Anon/Animator/blob/main/animation.py

Animation Script - https://github.com/Animator-Anon/Animator

A basic img2img script that will dump frames and build a video file. Suitable for creating interesting zoom in warping movies, but not too much else at this time.

## Parameter Sequencer
https://github.com/rewbs/sd-parseq

Generate videos with tight control and flexible interpolation over many Stable Diffusion parameters (such as seed, scale, prompt weights, denoising strength...), as well as input processing parameter (such as zoom, pan, 3D rotation...)

## Alternate Noise Schedules
https://gist.github.com/dfaker/f88aa62e3a14b559fe4e5f6b345db664

Uses alternate generators for the sampler's sigma schedule.

Allows access to Karras, Exponential and Variance Preserving schedules from crowsonkb/k-diffusion along with their parameters.

## Vid2Vid
https://github.com/Filarius/stable-diffusion-webui/blob/master/scripts/vid2vid.py

From real video, img2img the frames and stitch them together. Does not unpack frames to hard drive.

## Txt2VectorGraphics
https://github.com/GeorgLegato/Txt2Vectorgraphics

Create custom, scaleable icons from your prompts as SVG or PDF.

## Shift Attention
https://github.com/yownas/shift-attention

Generate a sequence of images shifting attention in the prompt.

This script enables you to give a range to the weight of tokens in a prompt and then generate a sequence of images stepping from the first one to the second.

## Loopback and Superimpose
https://github.com/DiceOwl/StableDiffusionStuff

https://github.com/DiceOwl/StableDiffusionStuff/blob/main/loopback_superimpose.py

Mixes output of img2img with original input image at strength alpha. The result is fed into img2img again (at loop>=2), and this procedure repeats. Tends to sharpen the image, improve consistency, reduce creativity and reduce fine detail.

## Interpolate
https://github.com/DiceOwl/StableDiffusionStuff

https://github.com/DiceOwl/StableDiffusionStuff/blob/main/interpolate.py

An img2img script to produce in-between images. Allows two input images for interpolation. More features shown in the [readme](https://github.com/DiceOwl/StableDiffusionStuff).

## Run n times
https://gist.github.com/camenduru/9ec5f8141db9902e375967e93250860f

Run n times with random seed.

## Advanced Loopback
https://github.com/Extraltodeus/advanced-loopback-for-sd-webui

Dynamic zoom loopback with parameters variations and prompt switching amongst other features!

## prompt-morph
https://github.com/feffy380/prompt-morph

Generate morph sequences with Stable Diffusion. Interpolate between two or more prompts and create an image at each step.

Uses the new AND keyword and can optionally export the sequence as a video.

## prompt interpolation
https://github.com/EugeoSynthesisThirtyTwo/prompt-interpolation-script-for-sd-webui

With this script, you can interpolate between two prompts (using the "AND" keyword), generate as many images as you want.
You can also generate a gif with the result. Works for both txt2img and img2img.

## Asymmetric Tiling
https://github.com/tjm35/asymmetric-tiling-sd-webui/

Control horizontal/vertical seamless tiling independently of each other.

## Force Symmetry
https://gist.github.com/1ort/2fe6214cf1abe4c07087aac8d91d0d8a

see https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/2441

applies symmetry to the image every n steps and sends the result further to img2img.

## StylePile
https://github.com/some9000/StylePile

Mix and match keywords to quickly get different results without wasting a lot of time writing prompts.

## XYZ Plot Script
https://github.com/xrpgame/xyz_plot_script

Generates an .html file to interactively browse the imageset. Use the scroll wheel or arrow keys to move in the Z dimension.

## Booru tag autocompletion
https://github.com/DominikDoom/a1111-sd-webui-tagcomplete

Displays autocompletion hints for tags from "image booru" boards such as Danbooru. Uses local tag CSV files and includes a config for customization.

Also supports completion for [wildcards](Custom-Scripts#wildcards)

## Embedding to PNG
https://github.com/dfaker/embedding-to-png-script

Converts existing embeddings to the shareable image versions.

## Basic Canvas Outpainting Test
https://github.com/TKoestlerx/sdexperiments

Script to allow for easier outpainting. Appears to be infinite outpainting.

## Random Steps and CFG
https://github.com/lilly1987/AI-WEBUI-scripts-Random

## Stable Diffusion Aesthetic Scorer
https://github.com/grexzen/SD-Chad

## img2tiles
https://github.com/arcanite24/img2tiles

generate tiles from a base image. Based on SD upscale script.

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