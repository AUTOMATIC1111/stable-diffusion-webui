import math
import os
import sys
import traceback

import modules.scripts as scripts
import gradio as gr

from modules.processing import Processed, process_images
from PIL import Image
from modules.shared import opts, cmd_opts, state


class Script(scripts.Script):
    def title(self):
        return "Batch processing"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        input_dir = gr.Textbox(label="Input directory", lines=1)
        output_dir = gr.Textbox(label="Output directory", lines=1)

        return [input_dir, output_dir]

    def run(self, p, input_dir, output_dir):
        images = [file for file in [os.path.join(input_dir, x) for x in os.listdir(input_dir)] if os.path.isfile(file)]

        batch_count = math.ceil(len(images) / p.batch_size)
        print(f"Will process {len(images)} images in {batch_count} batches.")

        p.batch_count = 1
        p.do_not_save_grid = True
        p.do_not_save_samples = True

        state.job_count = batch_count

        for batch_no in range(batch_count):
            batch_images = []
            for path in images[batch_no*p.batch_size:(batch_no+1)*p.batch_size]:
                try:
                    img = Image.open(path)
                    batch_images.append((img, path))
                except:
                    print(f"Error processing {path}:", file=sys.stderr)
                    print(traceback.format_exc(), file=sys.stderr)

            if len(batch_images) == 0:
                continue

            state.job = f"{batch_no} out of {batch_count}: {batch_images[0][1]}"
            p.init_images = [x[0] for x in batch_images]
            proc = process_images(p)
            for image, (_, path) in zip(proc.images, batch_images):
                filename = os.path.basename(path)
                image.save(os.path.join(output_dir, filename))

        return Processed(p, [], p.seed, "")
