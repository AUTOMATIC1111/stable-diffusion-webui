# Prompt Scripts

## Improved prompt matrix
https://github.com/ArrowM/auto1111-improved-prompt-matrix

This script is [advanced-prompt-matrix](https://github.com/GRMrGecko/stable-diffusion-webui-automatic/blob/advanced_matrix/scripts/advanced_prompt_matrix.py) modified to support `batch count`. Grids are not created.  

**Usage:**

Use `<` `>` to create a group of alternate texts. Separate text options with `|`. Multiple groups and multiple options can be used. For example:

An input of `a <corgi|cat> wearing <goggles|a hat>`  
Will output 4 prompts: `a corgi wearing goggles`, `a corgi wearing a hat`, `a cat wearing goggles`, `a cat wearing a hat`

When using a `batch count` > 1, each prompt variation will be generated for each seed. `batch size` is ignored.

## Parameter Sequencer
https://github.com/rewbs/sd-parseq

Generate videos with tight control and flexible interpolation over many Stable Diffusion parameters (such as seed, scale, prompt weights, denoising strength...), as well as input processing parameter (such as zoom, pan, 3D rotation...)

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


## Scripts by FartyPants
https://github.com/FartyPants/sd_web_ui_scripts

### Hallucinate

- swaps negative and positive prompts

### Mr. Negativity

- more advanced script that swaps negative and positive tokens depending on Mr. negativity rage

## Post-Face-Restore-Again
https://github.com/butaixianran/Stable-Diffusion-Webui-Post-Face-Restore-Again

Run face restore twice in one go, from extras tab.