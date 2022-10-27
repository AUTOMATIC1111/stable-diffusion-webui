from  parseq_core import Parseq
import string
from PIL import Image, ImageDraw
import json
import logging
import numpy as np
import re


class DummySDProcessing:

    def process_images(self, p):
         
        input_images = p.init_images
        output_images = []
        for image in input_images:
            # draw = ImageDraw.Draw(image)
            # draw.text((20, 70), re.sub("(.{64})", "\\1\n", json.dumps(p.extra_generation_params, indent=2), 0, re.DOTALL))
            output_images.append(image)

        dummy_processed = type("", (), dict(
            seed = p.seed,
            images = output_images
        ))()

        dummy_processed
        return dummy_processed

class DummyP:
    width=512
    height=512
    n_iter=1
    batch_size=1
    do_not_save_grid=True
    init_images = []
    seed=-1
    denoising_strength=0.8
    scale=7.5
    color_corrections = None
    extra_generation_params = dict()

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

param_script_string = open("../test_data/param_script.json", "r").read()

input_img = Image.open("../test_data/emad.png").convert('RGB')
#input_img=None

#input_path='./prod-30s-cropped.mp4'
input_path=None


Parseq().run(p=DummyP(), input_img=input_img, input_path=input_path, output_path='<img2img_output_path>/parseq-<timestamp>.mp4',
                         save_images=False, dry_run_mode=True, overlay_metadata=False,
                         default_output_dir='../test_data',
                         param_script_string=param_script_string, sd_processor=DummySDProcessing())

