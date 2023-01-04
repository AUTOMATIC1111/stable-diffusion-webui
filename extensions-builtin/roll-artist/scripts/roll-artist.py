import random

from modules import script_callbacks, shared
import gradio as gr

art_symbol = '\U0001f3a8'  # 🎨
global_prompt = None
related_ids = {"txt2img_prompt", "txt2img_clear_prompt", "img2img_prompt", "img2img_clear_prompt" }


def roll_artist(prompt):
    allowed_cats = set([x for x in shared.artist_db.categories() if len(shared.opts.random_artist_categories)==0 or x in shared.opts.random_artist_categories])
    artist = random.choice([x for x in shared.artist_db.artists if x.category in allowed_cats])

    return prompt + ", " + artist.name if prompt != '' else artist.name


def add_roll_button(prompt):
    roll = gr.Button(value=art_symbol, elem_id="roll", visible=len(shared.artist_db.artists) > 0)

    roll.click(
        fn=roll_artist,
        _js="update_txt2img_tokens",
        inputs=[
            prompt,
        ],
        outputs=[
            prompt,
        ]
    )


def after_component(component, **kwargs):
    global global_prompt

    elem_id = kwargs.get('elem_id', None)
    if elem_id not in related_ids:
        return

    if elem_id == "txt2img_prompt":
        global_prompt = component
    elif elem_id == "txt2img_clear_prompt":
        add_roll_button(global_prompt)
    elif elem_id == "img2img_prompt":
        global_prompt = component
    elif elem_id == "img2img_clear_prompt":
        add_roll_button(global_prompt)


script_callbacks.on_after_component(after_component)
