import base64
import io
import os
import re
import gradio as gr
from modules.shared import script_path
from modules import shared
import tempfile
from PIL import Image, PngImagePlugin

re_param_code = r'\s*([\w ]+):\s*("(?:\\|\"|[^\"])+"|[^,]*)(?:,|$)'
re_param = re.compile(re_param_code)
re_params = re.compile(r"^(?:" + re_param_code + "){3,}$")
re_imagesize = re.compile(r"^(\d+)x(\d+)$")
type_of_gr_update = type(gr.update())
paste_fields = {}
bind_list = []


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
        tempdir = os.path.normpath(tempfile.gettempdir())
        normfn = os.path.normpath(filename)
        assert normfn.startswith(tempdir), 'trying to open image file not in temporary directory'

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
                        fn=lambda x:x,
                        inputs=[send_image],
                        outputs=[paste_fields[tab]["init_img"]],
                    )

            if send_generate_info and paste_fields[tab]["fields"] is not None:
                paste_field_names = ['Prompt', 'Negative prompt', 'Steps', 'Face restoration', 'Size-1', 'Size-2']
                if shared.opts.send_seed:
                    paste_field_names += ["Seed"]
                if send_generate_info in paste_fields:
                    button.click(
                        fn=lambda *x:x,
                        inputs=[field for field,name in paste_fields[send_generate_info]["fields"] if name in paste_field_names],
                        outputs=[field for field,name in paste_fields[tab]["fields"] if name in paste_field_names],
                    )

                else:
                    connect_paste(button, [(field, name) for field, name in paste_fields[tab]["fields"]  if name in paste_field_names], send_generate_info)

            button.click(
                fn=None,
                _js=f"switch_to_{tab}",
                inputs=None,
                outputs=None,
            )


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


