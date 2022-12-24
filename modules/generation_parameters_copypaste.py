import base64
import io
import os
import re
from pathlib import Path

import gradio as gr
from modules.shared import script_path
from modules import shared
import tempfile
from PIL import Image

re_param_code = r'\s*([\w ]+):\s*("(?:\\|\"|[^\"])+"|[^,]*)(?:,|$)'
re_param = re.compile(re_param_code)
re_params = re.compile(r"^(?:" + re_param_code + "){3,}$")
re_imagesize = re.compile(r"^(\d+)x(\d+)$")
re_hypernet_hash = re.compile("\(([0-9a-f]+)\)$")
type_of_gr_update = type(gr.update())
paste_fields = {}
bind_list = []


def reset():
    paste_fields.clear()
    bind_list.clear()


def quote(text):
    if ',' not in str(text):
        return text

    text = str(text)
    text = text.replace('\\', '\\\\')
    text = text.replace('"', '\\"')
    return f'"{text}"'


def image_from_url_text(filedata):
    if type(filedata) == dict and filedata["is_file"]:
        filename = filedata["name"]
        is_in_right_dir = any(Path(temp_dir).resolve() in Path(filename).resolve().parents for temp_dir in shared.demo.temp_dirs)
        assert is_in_right_dir, 'trying to open image file outside of allowed directories'

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


def add_paste_fields(tabname, init_img, fields):
    paste_fields[tabname] = {"init_img": init_img, "fields": fields}

    # backwards compatibility for existing extensions
    import modules.ui
    if tabname == 'txt2img':
        modules.ui.txt2img_paste_fields = fields
    elif tabname == 'img2img':
        modules.ui.img2img_paste_fields = fields


def integrate_settings_paste_fields(component_dict):
    from modules import ui

    settings_map = {
        'sd_hypernetwork': 'Hypernet',
        'sd_hypernetwork_strength': 'Hypernet strength',
        'CLIP_stop_at_last_layers': 'Clip skip',
        'inpainting_mask_weight': 'Conditional mask weight',
        'sd_model_checkpoint': 'Model hash',
        'eta_noise_seed_delta': 'ENSD',
        'initial_noise_multiplier': 'Noise multiplier',
    }
    settings_paste_fields = [
        (component_dict[k], lambda d, k=k, v=v: ui.apply_setting(k, d.get(v, None)))
        for k, v in settings_map.items()
    ]

    for tabname, info in paste_fields.items():
        if info["fields"] is not None:
            info["fields"] += settings_paste_fields


def create_buttons(tabs_list):
    buttons = {}
    for tab in tabs_list:
        buttons[tab] = gr.Button(f"Send to {tab}")
    return buttons


#if send_generate_info is a tab name, mean generate_info comes from the params fields of the tab
def bind_buttons(buttons, send_image, send_generate_info):
    bind_list.append([buttons, send_image, send_generate_info])


def run_bind():
    for buttons, send_image, send_generate_info in bind_list:
        for tab in buttons:
            button = buttons[tab]
            if send_image and paste_fields[tab]["init_img"]:
                if type(send_image) == gr.Gallery:
                    button.click(
                        fn=lambda x: image_from_url_text(x),
                        _js="extract_image_from_gallery",
                        inputs=[send_image],
                        outputs=[paste_fields[tab]["init_img"]],
                    )
                else:
                    button.click(
                        fn=lambda x: x,
                        inputs=[send_image],
                        outputs=[paste_fields[tab]["init_img"]],
                    )

            if send_generate_info and paste_fields[tab]["fields"] is not None:
                if send_generate_info in paste_fields:
                    paste_field_names = ['Prompt', 'Negative prompt', 'Steps', 'Face restoration'] +  (['Size-1', 'Size-2'] if shared.opts.send_size else []) + (["Seed"] if shared.opts.send_seed else [])
                    button.click(
                        fn=lambda *x: x,
                        inputs=[field for field, name in paste_fields[send_generate_info]["fields"] if name in paste_field_names],
                        outputs=[field for field, name in paste_fields[tab]["fields"] if name in paste_field_names],
                    )
                else:
                    connect_paste(button, paste_fields[tab]["fields"], send_generate_info)

            button.click(
                fn=None,
                _js=f"switch_to_{tab}",
                inputs=None,
                outputs=None,
            )


def find_hypernetwork_key(hypernet_name, hypernet_hash=None):
    """Determines the config parameter name to use for the hypernet based on the parameters in the infotext.

    Example: an infotext provides "Hypernet: ke-ta" and "Hypernet hash: 1234abcd". For the "Hypernet" config
    parameter this means there should be an entry that looks like "ke-ta-10000(1234abcd)" to set it to.

    If the infotext has no hash, then a hypernet with the same name will be selected instead.
    """
    hypernet_name = hypernet_name.lower()
    if hypernet_hash is not None:
        # Try to match the hash in the name
        for hypernet_key in shared.hypernetworks.keys():
            result = re_hypernet_hash.search(hypernet_key)
            if result is not None and result[1] == hypernet_hash:
                return hypernet_key
    else:
        # Fall back to a hypernet with the same name
        for hypernet_key in shared.hypernetworks.keys():
            if hypernet_key.lower().startswith(hypernet_name):
                return hypernet_key

    return None


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
    if not re_params.match(lastline):
        lines.append(lastline)
        lastline = ''

    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("Negative prompt:"):
            done_with_prompt = True
            line = line[16:].strip()

        if done_with_prompt:
            negative_prompt += ("" if negative_prompt == "" else "\n") + line
        else:
            prompt += ("" if prompt == "" else "\n") + line

    res["Prompt"] = prompt
    res["Negative prompt"] = negative_prompt

    for k, v in re_param.findall(lastline):
        m = re_imagesize.match(v)
        if m is not None:
            res[k+"-1"] = m.group(1)
            res[k+"-2"] = m.group(2)
        else:
            res[k] = v

    # Missing CLIP skip means it was set to 1 (the default)
    if "Clip skip" not in res:
        res["Clip skip"] = "1"

    if "Hypernet strength" not in res:
        res["Hypernet strength"] = "1"

    if "Hypernet" in res:
        hypernet_name = res["Hypernet"]
        hypernet_hash = res.get("Hypernet hash", None)
        res["Hypernet"] = find_hypernetwork_key(hypernet_name, hypernet_hash)

    return res


def connect_paste(button, paste_fields, input_comp, jsfunc=None):
    def paste_func(prompt):
        if not prompt and not shared.cmd_opts.hide_ui_dir_config:
            filename = os.path.join(script_path, "params.txt")
            if os.path.exists(filename):
                with open(filename, "r", encoding="utf8") as file:
                    prompt = file.read()

        params = parse_generation_parameters(prompt)
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

    button.click(
        fn=paste_func,
        _js=jsfunc,
        inputs=[input_comp],
        outputs=[x[0] for x in paste_fields],
    )


