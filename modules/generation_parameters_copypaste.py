import base64
import io
import json
import os
import re

import gradio as gr
from modules.paths import data_path
from modules import shared, ui_tempdir, script_callbacks
from PIL import Image

re_param_code = r'\s*([\w ]+):\s*("(?:\\"[^,]|\\"|\\|[^\"])+"|[^,]*)(?:,|$)'
re_param = re.compile(re_param_code)
re_imagesize = re.compile(r"^(\d+)x(\d+)$")
re_hypernet_hash = re.compile("\(([0-9a-f]+)\)$")
type_of_gr_update = type(gr.update())

paste_fields = {}
registered_param_bindings = []


class ParamBinding:
    def __init__(self, paste_button, tabname, source_text_component=None, source_image_component=None, source_tabname=None, override_settings_component=None, paste_field_names=None):
        self.paste_button = paste_button
        self.tabname = tabname
        self.source_text_component = source_text_component
        self.source_image_component = source_image_component
        self.source_tabname = source_tabname
        self.override_settings_component = override_settings_component
        self.paste_field_names = paste_field_names or []


def reset():
    paste_fields.clear()


def quote(text):
    if ',' not in str(text) and '\n' not in str(text) and ':' not in str(text):
        return text

    return json.dumps(text, ensure_ascii=False)


def unquote(text):
    if len(text) == 0 or text[0] != '"' or text[-1] != '"':
        return text

    try:
        return json.loads(text)
    except Exception:
        return text


def image_from_url_text(filedata):
    if filedata is None:
        return None

    if type(filedata) == list and filedata and type(filedata[0]) == dict and filedata[0].get("is_file", False):
        filedata = filedata[0]

    if type(filedata) == dict and filedata.get("is_file", False):
        filename = filedata["name"]
        is_in_right_dir = ui_tempdir.check_tmp_file(shared.demo, filename)
        assert is_in_right_dir, 'trying to open image file outside of allowed directories'

        filename = filename.rsplit('?', 1)[0]
        return Image.open(filename)

    if type(filedata) == list:
        if len(filedata) == 0:
            return None

        filedata = filedata[0]

    if filedata.startswith("data:image/png;base64,"):
        filedata = filedata[len("data:image/png;base64,"):]

    filedata = base64.decodebytes(filedata.encode('utf-8'))
    image = Image.open(io.BytesIO(filedata))
    return image


def add_paste_fields(tabname, init_img, fields, override_settings_component=None):
    paste_fields[tabname] = {"init_img": init_img, "fields": fields, "override_settings_component": override_settings_component}

    # backwards compatibility for existing extensions
    import modules.ui
    if tabname == 'txt2img':
        modules.ui.txt2img_paste_fields = fields
    elif tabname == 'img2img':
        modules.ui.img2img_paste_fields = fields


def create_buttons(tabs_list):
    buttons = {}
    for tab in tabs_list:
        buttons[tab] = gr.Button(f"Send to {tab}", elem_id=f"{tab}_tab")
    return buttons


def bind_buttons(buttons, send_image, send_generate_info):
    """old function for backwards compatibility; do not use this, use register_paste_params_button"""
    for tabname, button in buttons.items():
        source_text_component = send_generate_info if isinstance(send_generate_info, gr.components.Component) else None
        source_tabname = send_generate_info if isinstance(send_generate_info, str) else None

        register_paste_params_button(ParamBinding(paste_button=button, tabname=tabname, source_text_component=source_text_component, source_image_component=send_image, source_tabname=source_tabname))


def register_paste_params_button(binding: ParamBinding):
    registered_param_bindings.append(binding)


def connect_paste_params_buttons():
    binding: ParamBinding
    for binding in registered_param_bindings:
        destination_image_component = paste_fields[binding.tabname]["init_img"]
        fields = paste_fields[binding.tabname]["fields"]
        override_settings_component = binding.override_settings_component or paste_fields[binding.tabname]["override_settings_component"]

        destination_width_component = next(iter([field for field, name in fields if name == "Size-1"] if fields else []), None)
        destination_height_component = next(iter([field for field, name in fields if name == "Size-2"] if fields else []), None)

        if binding.source_image_component and destination_image_component:
            if isinstance(binding.source_image_component, gr.Gallery):
                func = send_image_and_dimensions if destination_width_component else image_from_url_text
                jsfunc = "extract_image_from_gallery"
            else:
                func = send_image_and_dimensions if destination_width_component else lambda x: x
                jsfunc = None

            binding.paste_button.click(
                fn=func,
                _js=jsfunc,
                inputs=[binding.source_image_component],
                outputs=[destination_image_component, destination_width_component, destination_height_component] if destination_width_component else [destination_image_component],
                show_progress=False,
            )

        if binding.source_text_component is not None and fields is not None:
            connect_paste(binding.paste_button, fields, binding.source_text_component, override_settings_component, binding.tabname)

        if binding.source_tabname is not None and fields is not None:
            paste_field_names = ['Prompt', 'Negative prompt', 'Steps', 'Face restoration'] + (["Seed"] if shared.opts.send_seed else []) + binding.paste_field_names
            binding.paste_button.click(
                fn=lambda *x: x,
                inputs=[field for field, name in paste_fields[binding.source_tabname]["fields"] if name in paste_field_names],
                outputs=[field for field, name in fields if name in paste_field_names],
                show_progress=False,
            )

        binding.paste_button.click(
            fn=None,
            _js=f"switch_to_{binding.tabname}",
            inputs=None,
            outputs=None,
            show_progress=False,
        )


def send_image_and_dimensions(x):
    if isinstance(x, Image.Image):
        img = x
    else:
        img = image_from_url_text(x)

    if shared.opts.send_size and isinstance(img, Image.Image):
        w = img.width
        h = img.height
    else:
        w = gr.update()
        h = gr.update()

    return img, w, h


def restore_old_hires_fix_params(res):
    """for infotexts that specify old First pass size parameter, convert it into
    width, height, and hr scale"""

    firstpass_width = res.get('First pass size-1', None)
    firstpass_height = res.get('First pass size-2', None)

    if shared.opts.use_old_hires_fix_width_height:
        hires_width = int(res.get("Hires resize-1", 0))
        hires_height = int(res.get("Hires resize-2", 0))

        if hires_width and hires_height:
            res['Size-1'] = hires_width
            res['Size-2'] = hires_height
            return

    if firstpass_width is None or firstpass_height is None:
        return

    firstpass_width, firstpass_height = int(firstpass_width), int(firstpass_height)
    width = int(res.get("Size-1", 512))
    height = int(res.get("Size-2", 512))

    if firstpass_width == 0 or firstpass_height == 0:
        from modules import processing
        firstpass_width, firstpass_height = processing.old_hires_fix_first_pass_dimensions(width, height)

    res['Size-1'] = firstpass_width
    res['Size-2'] = firstpass_height
    res['Hires resize-1'] = width
    res['Hires resize-2'] = height


def parse_generation_parameters(x: str):
    """parses generation parameters string, the one you see in text field under the picture in UI:
```
girl with an artist's beret, determined, blue eyes, desert scene, computer monitors, heavy makeup, by Alphonse Mucha and Charlie Bowater, ((eyeshadow)), (coquettish), detailed, intricate
Negative prompt: ugly, fat, obese, chubby, (((deformed))), [blurry], bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), messy drawing
Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: 965400086, Size: 512x512, Model hash: 45dee52b
```

    returns a dict with field values
    """

    res = {}

    prompt = ""
    negative_prompt = ""

    done_with_prompt = False

    *lines, lastline = x.strip().split("\n")
    if len(re_param.findall(lastline)) < 3:
        lines.append(lastline)
        lastline = ''

    for line in lines:
        line = line.strip()
        if line.startswith("Negative prompt:"):
            done_with_prompt = True
            line = line[16:].strip()
        if done_with_prompt:
            negative_prompt += ("" if negative_prompt == "" else "\n") + line
        else:
            prompt += ("" if prompt == "" else "\n") + line

    if shared.opts.infotext_styles != "Ignore":
        found_styles, prompt, negative_prompt = shared.prompt_styles.extract_styles_from_prompt(prompt, negative_prompt)

        if shared.opts.infotext_styles == "Apply":
            res["Styles array"] = found_styles
        elif shared.opts.infotext_styles == "Apply if any" and found_styles:
            res["Styles array"] = found_styles

    res["Prompt"] = prompt
    res["Negative prompt"] = negative_prompt

    for k, v in re_param.findall(lastline):
        try:
            if v[0] == '"' and v[-1] == '"':
                v = unquote(v)

            m = re_imagesize.match(v)
            if m is not None:
                res[f"{k}-1"] = m.group(1)
                res[f"{k}-2"] = m.group(2)
            else:
                res[k] = v
        except Exception:
            print(f"Error parsing \"{k}: {v}\"")

    # Missing CLIP skip means it was set to 1 (the default)
    if "Clip skip" not in res:
        res["Clip skip"] = "1"

    hypernet = res.get("Hypernet", None)
    if hypernet is not None:
        res["Prompt"] += f"""<hypernet:{hypernet}:{res.get("Hypernet strength", "1.0")}>"""

    if "Hires resize-1" not in res:
        res["Hires resize-1"] = 0
        res["Hires resize-2"] = 0

    if "Hires sampler" not in res:
        res["Hires sampler"] = "Use same sampler"

    if "Hires prompt" not in res:
        res["Hires prompt"] = ""

    if "Hires negative prompt" not in res:
        res["Hires negative prompt"] = ""

    restore_old_hires_fix_params(res)

    # Missing RNG means the default was set, which is GPU RNG
    if "RNG" not in res:
        res["RNG"] = "GPU"

    if "Schedule type" not in res:
        res["Schedule type"] = "Automatic"

    if "Schedule max sigma" not in res:
        res["Schedule max sigma"] = 0

    if "Schedule min sigma" not in res:
        res["Schedule min sigma"] = 0

    if "Schedule rho" not in res:
        res["Schedule rho"] = 0

    return res


infotext_to_setting_name_mapping = [
    ('Clip skip', 'CLIP_stop_at_last_layers', ),
    ('Conditional mask weight', 'inpainting_mask_weight'),
    ('Model hash', 'sd_model_checkpoint'),
    ('ENSD', 'eta_noise_seed_delta'),
    ('Schedule type', 'k_sched_type'),
    ('Schedule max sigma', 'sigma_max'),
    ('Schedule min sigma', 'sigma_min'),
    ('Schedule rho', 'rho'),
    ('Noise multiplier', 'initial_noise_multiplier'),
    ('Eta', 'eta_ancestral'),
    ('Eta DDIM', 'eta_ddim'),
    ('Discard penultimate sigma', 'always_discard_next_to_last_sigma'),
    ('UniPC variant', 'uni_pc_variant'),
    ('UniPC skip type', 'uni_pc_skip_type'),
    ('UniPC order', 'uni_pc_order'),
    ('UniPC lower order final', 'uni_pc_lower_order_final'),
    ('Token merging ratio', 'token_merging_ratio'),
    ('Token merging ratio hr', 'token_merging_ratio_hr'),
    ('RNG', 'randn_source'),
    ('NGMS', 's_min_uncond'),
    ('Pad conds', 'pad_cond_uncond'),
]


def create_override_settings_dict(text_pairs):
    """creates processing's override_settings parameters from gradio's multiselect

    Example input:
        ['Clip skip: 2', 'Model hash: e6e99610c4', 'ENSD: 31337']

    Example output:
        {'CLIP_stop_at_last_layers': 2, 'sd_model_checkpoint': 'e6e99610c4', 'eta_noise_seed_delta': 31337}
    """

    res = {}

    params = {}
    for pair in text_pairs:
        k, v = pair.split(":", maxsplit=1)

        params[k] = v.strip()

    for param_name, setting_name in infotext_to_setting_name_mapping:
        value = params.get(param_name, None)

        if value is None:
            continue

        res[setting_name] = shared.opts.cast_value(setting_name, value)

    return res


def connect_paste(button, paste_fields, input_comp, override_settings_component, tabname):
    def paste_func(prompt):
        if not prompt and not shared.cmd_opts.hide_ui_dir_config:
            filename = os.path.join(data_path, "params.txt")
            if os.path.exists(filename):
                with open(filename, "r", encoding="utf8") as file:
                    prompt = file.read()

        params = parse_generation_parameters(prompt)
        script_callbacks.infotext_pasted_callback(prompt, params)
        res = []

        for output, key in paste_fields:
            if callable(key):
                v = key(params)
            else:
                v = params.get(key, None)

            if v is None:
                res.append(gr.update())
            elif isinstance(v, type_of_gr_update):
                res.append(v)
            else:
                try:
                    valtype = type(output.value)

                    if valtype == bool and v == "False":
                        val = False
                    else:
                        val = valtype(v)

                    res.append(gr.update(value=val))
                except Exception:
                    res.append(gr.update())

        return res

    if override_settings_component is not None:
        def paste_settings(params):
            vals = {}

            for param_name, setting_name in infotext_to_setting_name_mapping:
                v = params.get(param_name, None)
                if v is None:
                    continue

                if setting_name == "sd_model_checkpoint" and shared.opts.disable_weights_auto_swap:
                    continue

                v = shared.opts.cast_value(setting_name, v)
                current_value = getattr(shared.opts, setting_name, None)

                if v == current_value:
                    continue

                vals[param_name] = v

            vals_pairs = [f"{k}: {v}" for k, v in vals.items()]

            return gr.Dropdown.update(value=vals_pairs, choices=vals_pairs, visible=bool(vals_pairs))

        paste_fields = paste_fields + [(override_settings_component, paste_settings)]

    button.click(
        fn=paste_func,
        inputs=[input_comp],
        outputs=[x[0] for x in paste_fields],
        show_progress=False,
    )
    button.click(
        fn=None,
        _js=f"recalculate_prompts_{tabname}",
        inputs=[],
        outputs=[],
        show_progress=False,
    )
