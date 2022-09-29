""" 
using POTRACE as backend cmd line tool for vectorizing SD output
important is to use a proper PROMPT or better STYLE to enforce SD to output bitmaps, 1 color and already well thresholded
In the Settings config the path to your potrace executable (or just "potrace" if it is your PATH variable)
https://potrace.sourceforge.net/#downloading
"""

from inspect import _void
import os
import pathlib
import subprocess

import modules.scripts as scripts
import modules.images as Images
import gradio as gr

from modules.processing import Processed, process_images
from modules.shared import opts

POS_PROMPT = ",(((vector graphic))), (((black white, line art))), atari graphic"
NEG_PROMPT = ",background, colors, shading, details"

class Script(scripts.Script):
    def title(self):
        return "Text to Vectorgraphics"

    def ui(self, is_img2img):
        poFormat = gr.Dropdown(["svg","pdf"], label="Output format", value="svg")
        poOpaque = gr.Checkbox(label="White is Opaque", value=True)
        poTight = gr.Checkbox(label="Cut white margin from input", value=True)
        poKeepPnm = gr.Checkbox(label="Keep temp images", value=False)
        poThreshold = gr.Slider(label="Threshold", minimum=0.0, maximum=1.0, step=0.05, value=0.5)

        return [poFormat,poOpaque, poTight, poKeepPnm, poThreshold]

    def run(self, p, poFormat, poOpaque, poTight, poKeepPnm, poThreshold):
        p.do_not_save_grid = True

        # make SD great b/w stuff
        p.prompt += POS_PROMPT
        p.negative_prompt += NEG_PROMPT

        images = []
        proc = process_images(p)
        images += proc.images
        

        # vectorize
        for i,img in enumerate(images): 
            fullfn = Images.save_image(img, p.outpath_samples, "", p.seed, p.prompt, "pnm" )
            fullof = pathlib.Path(fullfn).with_suffix('.'+poFormat)

            args = [opts.potrace_path,  "-b", poFormat, "-o", fullof, "--blacklevel",  format(poThreshold, 'f')]
            if poOpaque: args.append("--opaque")
            if poTight: args.append("--tight")
            args.append(fullfn)

            p2 = subprocess.Popen(args)

            if not poKeepPnm:
                p2.wait()
                os.remove(fullfn)

        return Processed(p, images, p.seed, "")
