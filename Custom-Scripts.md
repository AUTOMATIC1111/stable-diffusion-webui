# Installing and Using Custom Scripts

The Script class enables custom extensions to how the image processing is handled after "Generate" is clicked. To install custom scripts, place them into the `scripts` directory and restart the web ui. Custom scripts will appear in the lower-left dropdown menu on the txt2img and img2img tabs after being installed. Below are some notable custom scripts created by Web UI users:

# Custom Scripts from Users

## Advanced prompt matrix
https://github.com/GRMrGecko/stable-diffusion-webui-automatic/blob/advanced_matrix/scripts/advanced_prompt_matrix.py

It allows a matrix prompt as follows:
`<cyber|cyborg|> cat <photo|image|artistic photo|oil painting> in a <car|boat|cyber city>`

Does not actually draw a matrix, just produces pictures.

## Wildcards
https://github.com/jtkelm2/stable-diffusion-webui-1/blob/master/scripts/wildcards.py

Script support so that prompts can contain wildcard terms (indicated by surrounding double underscores), with values instantiated randomly from the corresponding .txt file in the folder `/scripts/wildcards/`. For example:

`a woman at a cafe by __artist__ and __artist__`

will draw two random artists from `artist.txt`. This works independently on each prompt, so that one can e.g. generate a batch of 100 images with the same prompt input using wildcards, and each output will have unique artists (or styles, or genres, or anything that the user creates their own .txt file for) attached to it.

(also see https://github.com/jtkelm2/stable-diffusion-webui-1/tree/master/scripts/wildcards for examples of custom lists)

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

# Creating Custom Scripts

The Script class definition can be found in `modules/scripts.py`. To create your own custom script, create a python script that implements the class and drop it into the `scripts` folder, using the below example or other scripts already in the folder as a guide. 

The Script class has four primary methods, described in further detail below with a simple example script that rotates and/or flips generated images.

```python
import modules.scripts as scripts
import gradio as gr
import os

from modules import images
from modules.processing import process_images, Processed
from modules.processing import Processed
from modules.shared import opts, cmd_opts, state


class Script(scripts.Script):  

# The title of the script. This is what will be displayed in the dropdown menu.
    def title(self):

        return "Flip/Rotate Output"


# Determines when the script should be shown in the dropdown menu via the 
# returned value. As an example:
# is_img2img is True if the current tab is img2img, and False if it is txt2img.
# Thus, return is_img2img to only show the script on the img2img tab.

    def show(self, is_img2img):

        return is_img2img

# How the script's is displayed in the UI. See https://gradio.app/docs/#components
# for the different UI components you can use and how to create them.
# Most UI components can return a value, such as a boolean for a checkbox.
# The returned values are passed to the run method as parameters.

    def ui(self, is_img2img):
        angle = gr.Slider(minimum=0.0, maximum=360.0, step=1, value=0,
        label="Angle")
        hflip = gr.Checkbox(False, label="Horizontal flip")
        vflip = gr.Checkbox(False, label="Vertical flip")
        overwrite = gr.Checkbox(False, label="Overwrite existing files")
        return [angle, hflip, vflip, overwrite]

  

# This is where the additional processing is implemented. The parameters include
# self, the model object "p" (a StableDiffusionProcessing class, see
# processing.py), and the parameters returned by the ui method.
# Custom functions can be defined here, and additional libraries can be imported 
# to be used in processing. The return value should be a Processed object, which is
# what is returned by the process_images method.

    def run(self, p, angle, hflip, vflip, overwrite):

        # function which takes an image from the Processed object, 
        # and the angle and two booleans indicating horizontal and
        # vertical flips from the UI, then returns the 
        # image rotated and flipped accordingly
        def rotate_and_flip(im, angle, hflip, vflip):
            from PIL import Image
            
            raf = im
            
            if angle != 0:
                raf = raf.rotate(angle, expand=True)
            if hflip:
                raf = raf.transpose(Image.FLIP_LEFT_RIGHT)
            if vflip:
                raf = raf.transpose(Image.FLIP_TOP_BOTTOM)
            return raf

  

        # If overwrite is false, append the rotation information to the filename
        # using the "basename" parameter and save it in the same directory.
        # If overwrite is true, stop the model from saving its outputs and
        # save the rotated and flipped images instead.
        basename = ""
        if(not overwrite):
            if angle != 0:
                basename += "rotated_" + str(angle)
            if hflip:
                basename += "_hflip"
            if vflip:
                basename += "_vflip"
        else:
            p.do_not_save_samples = True


        proc = process_images(p)

        # rotate and flip each image in the processed images
        # use the save_images method from images.py to save
        # them.
        for i in range(len(proc.images)):

            proc.images[i] = rotate_and_flip(proc.images[i], angle, hflip, vflip)

            images.save_image(proc.images[i], p.outpath_samples, basename,
            proc.seed + i, proc.prompt, opts.samples_format, info= proc.info, p=p)

        return proc
```


# Saving steps of the sampling process
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