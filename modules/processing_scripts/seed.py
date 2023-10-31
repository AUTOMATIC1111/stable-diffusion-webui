import json

import gradio as gr

from modules import scripts, ui, errors
from modules.shared import cmd_opts
from modules.ui_components import ToolButton


class ScriptSeed(scripts.ScriptBuiltinUI):
    section = "seed"
    create_group = False

    def __init__(self):
        self.seed = None
        self.reuse_seed = None
        self.reuse_subseed = None

    def title(self):
        return "Seed"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Row(elem_id=self.elem_id("seed_row")):
            if cmd_opts.use_textbox_seed:
                self.seed = gr.Textbox(label='Seed', value="", elem_id=self.elem_id("seed"), min_width=100)
            else:
                self.seed = gr.Number(label='Seed', value=-1, elem_id=self.elem_id("seed"), min_width=100, precision=0)

            random_seed = ToolButton(ui.random_symbol, elem_id=self.elem_id("random_seed"), label='Random seed')
            reuse_seed = ToolButton(ui.reuse_symbol, elem_id=self.elem_id("reuse_seed"), label='Reuse seed')

            seed_checkbox = gr.Checkbox(label='Extra', elem_id=self.elem_id("subseed_show"), value=False)

        with gr.Group(visible=False, elem_id=self.elem_id("seed_extras")) as seed_extras:
            with gr.Row(elem_id=self.elem_id("subseed_row")):
                subseed = gr.Number(label='Variation seed', value=-1, elem_id=self.elem_id("subseed"), precision=0)
                random_subseed = ToolButton(ui.random_symbol, elem_id=self.elem_id("random_subseed"))
                reuse_subseed = ToolButton(ui.reuse_symbol, elem_id=self.elem_id("reuse_subseed"))
                subseed_strength = gr.Slider(label='Variation strength', value=0.0, minimum=0, maximum=1, step=0.01, elem_id=self.elem_id("subseed_strength"))

            with gr.Row(elem_id=self.elem_id("seed_resize_from_row")):
                seed_resize_from_w = gr.Slider(minimum=0, maximum=2048, step=8, label="Resize seed from width", value=0, elem_id=self.elem_id("seed_resize_from_w"))
                seed_resize_from_h = gr.Slider(minimum=0, maximum=2048, step=8, label="Resize seed from height", value=0, elem_id=self.elem_id("seed_resize_from_h"))

        random_seed.click(fn=None, _js="function(){setRandomSeed('" + self.elem_id("seed") + "')}", show_progress=False, inputs=[], outputs=[])
        random_subseed.click(fn=None, _js="function(){setRandomSeed('" + self.elem_id("subseed") + "')}", show_progress=False, inputs=[], outputs=[])

        seed_checkbox.change(lambda x: gr.update(visible=x), show_progress=False, inputs=[seed_checkbox], outputs=[seed_extras])

        self.infotext_fields = [
            (self.seed, "Seed"),
            (seed_checkbox, lambda d: "Variation seed" in d or "Seed resize from-1" in d),
            (subseed, "Variation seed"),
            (subseed_strength, "Variation seed strength"),
            (seed_resize_from_w, "Seed resize from-1"),
            (seed_resize_from_h, "Seed resize from-2"),
        ]

        self.on_after_component(lambda x: connect_reuse_seed(self.seed, reuse_seed, x.component, False), elem_id=f'generation_info_{self.tabname}')
        self.on_after_component(lambda x: connect_reuse_seed(subseed, reuse_subseed, x.component, True), elem_id=f'generation_info_{self.tabname}')

        return self.seed, seed_checkbox, subseed, subseed_strength, seed_resize_from_w, seed_resize_from_h

    def setup(self, p, seed, seed_checkbox, subseed, subseed_strength, seed_resize_from_w, seed_resize_from_h):
        p.seed = seed

        if seed_checkbox and subseed_strength > 0:
            p.subseed = subseed
            p.subseed_strength = subseed_strength

        if seed_checkbox and seed_resize_from_w > 0 and seed_resize_from_h > 0:
            p.seed_resize_from_w = seed_resize_from_w
            p.seed_resize_from_h = seed_resize_from_h



def connect_reuse_seed(seed: gr.Number, reuse_seed: gr.Button, generation_info: gr.Textbox, is_subseed):
    """ Connects a 'reuse (sub)seed' button's click event so that it copies last used
        (sub)seed value from generation info the to the seed field. If copying subseed and subseed strength
        was 0, i.e. no variation seed was used, it copies the normal seed value instead."""

    def copy_seed(gen_info_string: str, index):
        res = -1

        try:
            gen_info = json.loads(gen_info_string)
            index -= gen_info.get('index_of_first_image', 0)

            if is_subseed and gen_info.get('subseed_strength', 0) > 0:
                all_subseeds = gen_info.get('all_subseeds', [-1])
                res = all_subseeds[index if 0 <= index < len(all_subseeds) else 0]
            else:
                all_seeds = gen_info.get('all_seeds', [-1])
                res = all_seeds[index if 0 <= index < len(all_seeds) else 0]

        except json.decoder.JSONDecodeError:
            if gen_info_string:
                errors.report(f"Error parsing JSON generation info: {gen_info_string}")

        return [res, gr.update()]

    reuse_seed.click(
        fn=copy_seed,
        _js="(x, y) => [x, selected_gallery_index()]",
        show_progress=False,
        inputs=[generation_info, seed],
        outputs=[seed, seed]
    )
