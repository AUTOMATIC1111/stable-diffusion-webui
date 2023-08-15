import os

import gradio as gr

from modules import errors, shared
from modules.paths_internal import script_path


# https://huggingface.co/datasets/freddyaboulton/gradio-theme-subdomains/resolve/main/subdomains.json
gradio_hf_hub_themes = [
    "gradio/base",
    "gradio/glass",
    "gradio/monochrome",
    "gradio/seafoam",
    "gradio/soft",
    "gradio/dracula_test",
    "abidlabs/dracula_test",
    "abidlabs/Lime",
    "abidlabs/pakistan",
    "Ama434/neutral-barlow",
    "dawood/microsoft_windows",
    "finlaymacklon/smooth_slate",
    "Franklisi/darkmode",
    "freddyaboulton/dracula_revamped",
    "freddyaboulton/test-blue",
    "gstaff/xkcd",
    "Insuz/Mocha",
    "Insuz/SimpleIndigo",
    "JohnSmith9982/small_and_pretty",
    "nota-ai/theme",
    "nuttea/Softblue",
    "ParityError/Anime",
    "reilnuud/polite",
    "remilia/Ghostly",
    "rottenlittlecreature/Moon_Goblin",
    "step-3-profit/Midnight-Deep",
    "Taithrah/Minimal",
    "ysharma/huggingface",
    "ysharma/steampunk",
    "NoCrypt/miku"
]


def reload_gradio_theme(theme_name=None):
    if not theme_name:
        theme_name = shared.opts.gradio_theme

    default_theme_args = dict(
        font=["Source Sans Pro", 'ui-sans-serif', 'system-ui', 'sans-serif'],
        font_mono=['IBM Plex Mono', 'ui-monospace', 'Consolas', 'monospace'],
    )

    if theme_name == "Default":
        shared.gradio_theme = gr.themes.Default(**default_theme_args)
    else:
        try:
            theme_cache_dir = os.path.join(script_path, 'tmp', 'gradio_themes')
            theme_cache_path = os.path.join(theme_cache_dir, f'{theme_name.replace("/", "_")}.json')
            if shared.opts.gradio_themes_cache and os.path.exists(theme_cache_path):
                shared.gradio_theme = gr.themes.ThemeClass.load(theme_cache_path)
            else:
                os.makedirs(theme_cache_dir, exist_ok=True)
                shared.gradio_theme = gr.themes.ThemeClass.from_hub(theme_name)
                shared.gradio_theme.dump(theme_cache_path)
        except Exception as e:
            errors.display(e, "changing gradio theme")
            shared.gradio_theme = gr.themes.Default(**default_theme_args)
