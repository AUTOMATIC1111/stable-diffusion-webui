> ℹ️ **Note:**
> This page is not actively maintained. For a more up-to-date list of scripts and extensions, you may use the built-in tab within the web UI (`Extensions` -> `Available`)

# Installing and Using Custom Scripts
To install custom scripts, place them into the `scripts` directory and click the `Reload custom script` button at the bottom in the settings tab. Custom scripts will appear in the lower-left dropdown menu on the txt2img and img2img tabs after being installed. Below are some notable custom scripts created by Web UI users:


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

<details><summary>Example: (Click to expand:)</summary>

| prompt  |PNG  |SVG |
| :--------  | :-----------------: | :---------------------: |
| Happy Einstein | <img src="https://user-images.githubusercontent.com/7210708/193370360-506eb6b5-4fa7-4b2a-9fec-6430f6d027f5.png" width="40%" /> | <img src="https://user-images.githubusercontent.com/7210708/193370379-2680aa2a-f460-44e7-9c4e-592cf096de71.svg" width=30%/> |
| Mountainbike Downhill | <img src="https://user-images.githubusercontent.com/7210708/193371353-f0f5ff6f-12f7-423b-a481-f9bd119631dd.png" width=40%/> | <img src="https://user-images.githubusercontent.com/7210708/193371585-68dea4ca-6c1a-4d31-965d-c1b5f145bb6f.svg" width=30%/> |
coffe mug in shape of a heart | <img src="https://user-images.githubusercontent.com/7210708/193374299-98379ca1-3106-4ceb-bcd3-fa129e30817a.png" width=40%/> | <img src="https://user-images.githubusercontent.com/7210708/193374525-460395af-9588-476e-bcf6-6a8ad426be8e.svg" width=30%/> |
| Headphones | <img src="https://user-images.githubusercontent.com/7210708/193376238-5c4d4a8f-1f06-4ba4-b780-d2fa2e794eda.png" width=40%/> | <img src="https://user-images.githubusercontent.com/7210708/193376255-80e25271-6313-4bff-a98e-ba3ae48538ca.svg" width=30%/> |

</details>

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

<details><summary>Example: (Click to expand:)</summary>

![gif](https://user-images.githubusercontent.com/24735555/195470874-afc3dfdc-7b35-4b23-9c34-5888a4100ac1.gif)

</details>

## Asymmetric Tiling
https://github.com/tjm35/asymmetric-tiling-sd-webui/

Control horizontal/vertical seamless tiling independently of each other.

<details><summary>Example: (Click to expand:)</summary>
<img src="https://user-images.githubusercontent.com/19196175/195132862-8c050327-92f3-44a4-9c02-0f11cce0b609.png" width="624" height="312" />
</details>

## Force Symmetry
https://gist.github.com/missionfloyd/69e5a5264ad09ccaab52355b45e7c08f

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

## XYZ Plot Script
https://github.com/xrpgame/xyz_plot_script

Generates an .html file to interactively browse the imageset. Use the scroll wheel or arrow keys to move in the Z dimension.

<details><summary>Example: (Click to expand:)</summary>
<img src="https://raw.githubusercontent.com/xrpgame/xyz_plot_script/master/example1.png" width="522" height="312" />
</details>

## Expanded-XY-grid
https://github.com/0xALIVEBEEF/Expanded-XY-grid

Custom script for AUTOMATIC1111's stable-diffusion-webui that adds more features to the standard xy grid:

- Multitool: Allows multiple parameters in one axis, theoretically allows unlimited parameters to be adjusted in one xy grid

- Customizable prompt matrix

- Group files in a directory

- S/R Placeholder - replace a placeholder value (the first value in the list of parameters) with desired values.

- Add PNGinfo to grid image

<details><summary>Example: (Click to expand:)</summary>

<img src="https://user-images.githubusercontent.com/80003301/202277871-a4a3341b-13f7-42f4-a3e6-ca8f8cd8250a.png" width="574" height="197" />

Example images: Prompt: "darth vader riding a bicycle, modifier"; X: Multitool: "Prompt S/R: bicycle, motorcycle | CFG scale: 7.5, 10 | Prompt S/R Placeholder: modifier, 4k, artstation"; Y: Multitool: "Sampler: Euler, Euler a | Steps: 20, 50" 

</details>

## Embedding to PNG
https://github.com/dfaker/embedding-to-png-script

Converts existing embeddings to the shareable image versions.

<details><summary>Example: (Click to expand:)</summary>
<img src="https://user-images.githubusercontent.com/35278260/196052398-268a3a3e-0fad-46cd-b37d-9808480ceb18.png" width="263" height="256" />
</details>

## Alpha Canvas
https://github.com/TKoestlerx/sdexperiments

Outpaint a region. Infinite outpainting concept, used the two existing outpainting scripts from the AUTOMATIC1111 repo as a basis.

<details><summary>Example: (Click to expand:)</summary>
<img src="https://user-images.githubusercontent.com/86352149/199517938-3430170b-adca-487c-992b-eb89b3b63681.jpg" width="446" height="312" />
</details>

## Random grid
https://github.com/lilly1987/AI-WEBUI-scripts-Random

Randomly enter xy grid values.

<details><summary>Example: (Click to expand:)</summary>
<img src="https://user-images.githubusercontent.com/20321215/197346726-f93b7e84-f808-4167-9969-dc42763eeff1.png" width="198" height="312" />

Basic logic is same as x/y plot, only internally, x type is fixed as step, and type y is fixed as cfg.
Generates x values as many as the number of step counts (10) within the range of step1|2 values (10-30)
Generates x values as many as the number of cfg counts (10) within the range of cfg1|2 values (6-15)
Even if you put the 1|2 range cap upside down, it will automatically change it.
In the case of the cfg value, it is treated as an int type and the decimal value is not read.
</details>

## Random
https://github.com/lilly1987/AI-WEBUI-scripts-Random

Repeat a simple number of times without a grid.

<details><summary>Example: (Click to expand:)</summary>
<img src="https://user-images.githubusercontent.com/20321215/197346617-0ed1cd09-0ddd-48ad-8161-bc1540d628ad.png" width="258" height="312" />
</details>

## Stable Diffusion Aesthetic Scorer
https://github.com/grexzen/SD-Chad

Rates your images.

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

## Test my prompt
https://github.com/Extraltodeus/test_my_prompt

Have you ever used a very long prompt full of words that you are not sure have any actual impact on your image? Did you lose the courage to try to remove them one by one to test if their effects are worthy of your pwescious GPU?

WELL now you don't need any courage as this script has been MADE FOR YOU!

It generates as many images as there are words in your prompt (you can select the separator of course).

<details><summary>Example: (Click to expand:)</summary>

Here the prompt is simply : "**banana, on fire, snow**" and so as you can see it has generated each image without each description in it.

<img src="https://user-images.githubusercontent.com/15731540/200349119-e45d3cfb-39f0-4999-a8f0-4671a6393824.png" width="512" height="512" />

You can also test your negative prompt.

</details>

## Pixel Art
https://github.com/C10udburst/stable-diffusion-webui-scripts

Simple script which resizes images by a variable amount, also converts image to use a color palette of a given size.

<details><summary>Example: (Click to expand:)</summary>

| Disabled | Enabled x8, no resize back, no color palette | Enabled x8, no color palette | Enabled x8, 16 color palette |
| :---: | :---: | :---: | :---: |
|![preview](https://user-images.githubusercontent.com/18114966/201491785-e30cfa9d-c850-4853-98b8-11db8de78c8d.png) | ![preview](https://user-images.githubusercontent.com/18114966/201492204-f4303694-e98d-4ea3-8256-538a88ea26b6.png) | ![preview](https://user-images.githubusercontent.com/18114966/201491864-d0c0c9f1-e34f-4cb6-a68e-7043ec5ce74e.png) | ![preview](https://user-images.githubusercontent.com/18114966/201492175-c55fa260-a17d-47c9-a919-9116e1caa8fe.png) |

[model used](https://publicprompts.art/all-in-one-pixel-art-dreambooth-model/)
```text
japanese pagoda with blossoming cherry trees, full body game asset, in pixelsprite style
Steps: 20, Sampler: DDIM, CFG scale: 7, Seed: 4288895889, Size: 512x512, Model hash: 916ea38c, Batch size: 4
```

</details>

## Scripts by FartyPants
https://github.com/FartyPants/sd_web_ui_scripts

### Hallucinate

- swaps negative and positive prompts

### Mr. Negativity

- more advanced script that swaps negative and positive tokens depending on Mr. negativity rage

## gif2gif
https://github.com/LonicaMewinsky/gif2gif

The purpose of this script is to accept an animated gif as input, process frames as img2img typically would, and recombine them back into an animated gif. Not intended to have extensive functionality. Referenced code from prompts_from_file.

## Post-Face-Restore-Again
https://github.com/butaixianran/Stable-Diffusion-Webui-Post-Face-Restore-Again

Run face restore twice in one go, from extras tab.

## Infinite Zoom
https://github.com/coolzilj/infinite-zoom

Generate Zoom in/out videos, with outpainting, as a custom script for inpaint mode in img2img tab.

## ImageReward Scorer

https://github.com/THUDM/ImageReward#integration-into-stable-diffusion-web-ui

An image **scorer** based on [ImageReward](https://github.com/THUDM/ImageReward), the first general-purpose text-to-image human preference RM, which is trained on in total **137k pairs of expert comparisons**.

[**Features**](https://github.com/THUDM/ImageReward#features) developed to date (2023-04-24) include:  (click to expand demo video)
<details>
    <summary>1. Score generated images and append to image information</summary>
    
https://user-images.githubusercontent.com/98524878/233889441-d593675a-dff4-43aa-ad6b-48cc68326fb0.mp4
  
</details>
<details>
    <summary>2. Automatically filter out images with low scores</summary>
    
https://user-images.githubusercontent.com/98524878/233889490-5c4a062f-bb5e-4179-ba98-b336cda4d290.mp4
  
</details>

For details including **installing** and **feature-specific usage**, check [the script introduction](https://github.com/THUDM/ImageReward#integration-into-stable-diffusion-web-ui).


## Saving steps of the sampling process
(Example Script) \
This script will save steps of the sampling process to a directory.
```python
import os.path

import modules.scripts as scripts
import gradio as gr

from modules import shared, sd_samplers_common
from modules.processing import Processed, process_images

class Script(scripts.Script):
    def title(self):
        return "Save steps of the sampling process to files"

    def ui(self, is_img2img):
        path = gr.Textbox(label="Save images to path", placeholder="Enter folder path here. Defaults to webui's root folder")
        return [path]

    def run(self, p, path):
        if not os.path.exists(path):
            os.makedirs(path)
        index = [0]

        def store_latent(x):
            image = shared.state.current_image = sd_samplers_common.sample_to_image(x)
            image.save(os.path.join(path, f"sample-{index[0]:05}.png"))
            index[0] += 1
            fun(x)

        fun = sd_samplers_common.store_latent
        sd_samplers_common.store_latent = store_latent

        try:
            proc = process_images(p)
        finally:
            sd_samplers_common.store_latent = fun

        return Processed(p, proc.images, p.seed, "")
```
