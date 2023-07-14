from modules.shared import opts, cmd_opts, state
from modules.processing import Processed, StableDiffusionProcessingImg2Img, process_images, images
from PIL import Image, ImageFont, ImageDraw, ImageOps
from fonts.ttf import Roboto
import modules.scripts as scripts
import gradio as gr
from collections import namedtuple
from random import randint

class Script(scripts.Script):
    GridSaveFlags = namedtuple('GridSaveFlags', ['never_grid', 'always_grid', 'always_save_grid'], defaults=(False, False, False))
    grid_options_mapping = {
        "Use user settings": GridSaveFlags(),
        "Don't generate": GridSaveFlags(never_grid=True),
        "Generate": GridSaveFlags(always_grid=True),
        "Generate and always save": GridSaveFlags(always_grid=True, always_save_grid=True),
        }
    default_grid_opt = list(grid_options_mapping.keys())[-1]

    def title(self):
        return "Test my prompt!"

    def ui(self, is_img2img):
        neg_pos = gr.Dropdown(label="Test negative or positive", choices=["Positive","Negative"], value="Positive")
        skip_x_first = gr.Slider(minimum=0, maximum=32, step=1, label='Skip X first words', value=0)
        separator = gr.Textbox(label="Separator used", lines=1, value=", ")
        grid_option = gr.Radio(choices=list(self.grid_options_mapping.keys()), label='Grid generation', value=self.default_grid_opt)
        font_size = gr.Slider(minimum=12, maximum=64, step=1, label='Font size', value=32)
        return [neg_pos,skip_x_first,separator,grid_option,font_size]

    def run(self, p,neg_pos,skip_x_first,separator,grid_option,font_size):
        def write_on_image(img, msg):
            ix,iy = img.size
            draw = ImageDraw.Draw(img)
            margin=2
            fontsize=font_size
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype(Roboto, fontsize)
            text_height=iy-60
            tx = draw.textbbox((0,0),msg,font)
            draw.text((int((ix-tx[2])/2),text_height+margin),msg,(0,0,0),font=font)
            draw.text((int((ix-tx[2])/2),text_height-margin),msg,(0,0,0),font=font)
            draw.text((int((ix-tx[2])/2+margin),text_height),msg,(0,0,0),font=font)
            draw.text((int((ix-tx[2])/2-margin),text_height),msg,(0,0,0),font=font)
            draw.text((int((ix-tx[2])/2),text_height), msg,(255,255,255),font=font)
            return img


        p.do_not_save_samples = True
        initial_seed = p.seed
        if initial_seed == -1:
            initial_seed = randint(1000000,9999999)
        if neg_pos == "Positive":
            initial_prompt =  p.prompt
            prompt_array = p.prompt
        else:
            initial_prompt =  p.negative_prompt
            prompt_array = p.negative_prompt

        prompt_array = prompt_array.split(separator)
        print("total images :", len(prompt_array))
        for g in range(len(prompt_array)+1):
            f = g-1
            if f >= 0 and f < skip_x_first:
                continue
            if f >= 0:
                new_prompt =  separator.join([prompt_array[x] for x in range(len(prompt_array)) if x is not f])
            else:
                new_prompt = initial_prompt

            if neg_pos == "Positive":
                p.prompt = new_prompt
            else:
                p.negative_prompt = new_prompt
            p.seed = initial_seed
            if g == 0:
                proc = process_images(p)
            else:
                appendimages = process_images(p)
                proc.images.insert(0,appendimages.images[0])
                proc.infotexts.insert(0,appendimages.infotexts[0])
            if f >= 0:
                proc.images[0] = write_on_image(proc.images[0], "no "+prompt_array[f])
            else:
                proc.images[0] = write_on_image(proc.images[0], "full prompt")

            if opts.samples_save:
                images.save_image(proc.images[0], p.outpath_samples, "", proc.seed, proc.prompt, opts.samples_format, info= proc.info, p=p)

        grid_flags = self.grid_options_mapping[grid_option]
        unwanted_grid_because_of_img_count = len(proc.images) < 2 and opts.grid_only_if_multiple
        if ((opts.return_grid or opts.grid_save) and not p.do_not_save_grid and not grid_flags.never_grid and not unwanted_grid_because_of_img_count) or grid_flags.always_grid:
            grid = images.image_grid(proc.images)
            proc.images.insert(0,grid)
            proc.infotexts.insert(0, proc.infotexts[-1])
            if opts.grid_save or grid_flags.always_save_grid:
                images.save_image(grid, p.outpath_grids, "grid", initial_seed, initial_prompt, opts.grid_format, info=proc.info, short_filename=not opts.grid_extended_filename, p=p, grid=True)
        return proc
