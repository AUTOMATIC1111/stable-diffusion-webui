# txt2mask for Stable Diffusion
Automatically create masks for inpainting with Stable Diffusion using natural language.

## Introduction

txt2mask is an addon for [AUTOMATIC1111's Stable Diffusion Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) that allows you to enter a text string in img2img mode which automatically creates an image mask. It is powered by [clipseg](https://github.com/timojl/clipseg). No more messing around with that tempermental brush tool. 😅

This script is still under active development.

![image](https://user-images.githubusercontent.com/95403634/190878562-d020887c-ccb0-411c-ab37-38e2115552eb.png)

## Installation

Simply clone or download this repo and place the files in the base directory of Automatic's web UI.

## Usage

From the img2img screen, select txt2mask as your active script:

![image](https://user-images.githubusercontent.com/95403634/190878234-43134aff-0843-4caf-a0ea-146d6e1891dc.png)

In the `Mask Prompt` field, enter the text to search for within your image. (In the case of the topmost screenshot, this value would be 'business suit' and the prompt box at the top of your UI would say 'sci-fi battle suit.')

Adjust the `Mask Precision` field to increase or decrease the confidence of that which is masked. Lowering this value too much means it may select more than you intend.

Press Generate. That's it!

## Advanced Features & Tips

- The Mask Prompt allows you to search for multiple objects by using `|` as a delimiter. For example, if you enter `a face|a tree|a flower` then clipseg will process these three items independently and stack the resulting submasks into one final mask. This will likely yield a better result than had you searched for `a face and a tree and a flower`.
- You can use the `Mask Padding` option to increase the boundaries of your selection. For example, if you enter `a red shirt` as your prompt but find that it's not quite selecting the whole shirt, and `Mask Precision` isn't helping, then padding may be a good way to address the issue.
- Use the `Negative mask prompt` to subtract from areas selected by `Mask prompt`. For example, if your prompt is `a face` and the negative prompt is `eyes` then the resulting mask will select a face without selecting the eyes.
- **(NEW)** You can combine your text mask with the brush tool or uploaded image mask using the `Brush mask mode` setting. Get the best of both worlds.
-  In general, less is more for masking: instead of trying to mask "a one-armed man doing a backflip off a barn" you will probably have more luck writing "a man."