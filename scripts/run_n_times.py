import math
import os
import sys
import traceback

import modules.scripts as scripts
import gradio as gr

from modules.processing import Processed, process_images

class Script(scripts.Script):
    def title(self):
        return "To Infinity and Beyond"

    def ui(self, is_img2img):
        n = gr.Textbox(label="n")
        return [n]

    def run(self, p, n):
        for x in range(int(n)):
            p.seed = -1
            proc = process_images(p)
            image = proc.images
        return Processed(p, image, p.seed, proc.info)