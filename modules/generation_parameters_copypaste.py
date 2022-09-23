from collections import namedtuple
import re
import gradio as gr

re_param = re.compile(r"\s*([\w ]+):\s*([^,]+)(?:,|$)")
re_imagesize = re.compile(r"^(\d+)x(\d+)$")


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
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("Negative prompt:"):
            done_with_prompt = True
            line = line[16:].strip()

        if done_with_prompt:
            negative_prompt += line
        else:
            prompt += line

    if len(prompt) > 0:
        res["Prompt"] = prompt

    if len(negative_prompt) > 0:
        res["Negative prompt"] = negative_prompt

    for k, v in re_param.findall(lastline):
        m = re_imagesize.match(v)
        if m is not None:
            res[k+"-1"] = m.group(1)
            res[k+"-2"] = m.group(2)
        else:
            res[k] = v

    return res


def connect_paste(button, d, input_comp, js=None):
    items = []
    outputs = []

    def paste_func(prompt):
        params = parse_generation_parameters(prompt)
        res = []

        for key, output in zip(items, outputs):
            v = params.get(key, None)

            if v is None:
                res.append(gr.update())
            else:
                try:
                    valtype = type(output.value)
                    val = valtype(v)
                    res.append(gr.update(value=val))
                except Exception:
                    res.append(gr.update())

        return res

    for k, v in d.items():
        items.append(k)
        outputs.append(v)

    button.click(
        fn=paste_func,
        _js=js,
        inputs=[input_comp],
        outputs=outputs,
    )
