import os
import json
import gradio as gr
import modules.shared


gradio_theme = gr.themes.Base()
# modules.shared.opts.onchange("gradio_theme", reload_gradio_theme)


def list_builtin_themes():
    files = [os.path.splitext(f)[0] for f in os.listdir('javascript') if f.endswith('.css') and f not in ['base.css', 'sdnext.css', 'style.css']]
    return files


def list_themes():
    fn = os.path.join('html', 'themes.json')
    if not os.path.exists(fn):
        refresh_themes()
    if os.path.exists(fn):
        with open(fn, mode='r', encoding='utf=8') as f:
            res = json.loads(f.read())
    else:
        res = []
    builtin = list_builtin_themes()
    default = ["gradio/default", "gradio/base", "gradio/glass", "gradio/monochrome", "gradio/soft"]
    external = {x['id'] for x in res if x['status'] == 'RUNNING' and 'test' not in x['id'].lower()}
    external = [f'huggingface/{x}' for x in external]
    modules.shared.log.debug(f'Themes: builtin={len(builtin)} default={len(default)} external={len(external)}')
    themes = sorted(builtin) + sorted(default) + sorted(external, key=str.casefold)
    return themes


def refresh_themes():
    try:
        r = modules.shared.req('https://huggingface.co/datasets/freddyaboulton/gradio-theme-subdomains/resolve/main/subdomains.json')
        if r.status_code == 200:
            res = r.json()
            fn = os.path.join('html', 'themes.json')
            modules.shared.writefile(res, fn)
            list_themes()
        else:
            modules.shared.log.error('Error refreshing UI themes')
    except Exception:
        modules.shared.log.error('Exception refreshing UI themes')


def reload_gradio_theme(theme_name=None):
    global gradio_theme # pylint: disable=global-statement
    if not theme_name:
        theme_name = modules.shared.opts.gradio_theme
    default_font_params = {}
    """
    res = 0
    try:
        import urllib.request
        request = urllib.request.Request("https://fonts.googleapis.com/css2?family=IBM+Plex+Mono", method="HEAD")
        res = urllib.request.urlopen(request, timeout=3.0).status # pylint: disable=consider-using-with
    except Exception:
        res = 0
    if res != 200:
        modules.shared.log.info('No internet access detected, using default fonts')
    """
    default_font_params = {
        'font':['Helvetica', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        'font_mono':['IBM Plex Mono', 'ui-monospace', 'Consolas', 'monospace']
    }
    is_builtin = theme_name in list_builtin_themes()
    modules.shared.log.info(f'Load UI theme: name="{theme_name}" style={modules.shared.opts.theme_style} base={"sdnext.css" if is_builtin else "base.css"}')
    if is_builtin:
        gradio_theme = gr.themes.Base(**default_font_params)
    elif theme_name.startswith("gradio/"):
        modules.shared.log.warning('Using Gradio default theme which is not optimized for SD.Next')
        if theme_name == "gradio/default":
            gradio_theme = gr.themes.Default(**default_font_params)
        if theme_name == "gradio/base":
            gradio_theme = gr.themes.Base(**default_font_params)
        if theme_name == "gradio/glass":
            gradio_theme = gr.themes.Glass(**default_font_params)
        if theme_name == "gradio/monochrome":
            gradio_theme = gr.themes.Monochrome(**default_font_params)
        if theme_name == "gradio/soft":
            gradio_theme = gr.themes.Soft(**default_font_params)
    else:
        try:
            hf_theme_name = theme_name.replace('huggingface/', '')
            modules.shared.log.warning('Using 3rd party theme which is not optimized for SD.Next')
            gradio_theme = gr.themes.ThemeClass.from_hub(hf_theme_name)
        except Exception:
            modules.shared.log.error("Theme download error accessing HuggingFace")
            gradio_theme = gr.themes.Default(**default_font_params)
    return is_builtin
