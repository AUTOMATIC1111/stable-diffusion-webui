import os
import gradio as gr

from modules import localization, shared, scripts, util
from modules.paths import script_path, data_path


def webpath(fn):
    return f'file={util.truncate_path(fn)}?{os.path.getmtime(fn)}'


def javascript_html():
    # Ensure localization is in `window` before scripts
    head = f'<script type="text/javascript">{localization.localization_js(shared.opts.localization)}</script>\n'

    script_js = os.path.join(script_path, "script.js")
    head += f'<script type="text/javascript" src="{webpath(script_js)}"></script>\n'

    for script in scripts.list_scripts("javascript", ".js"):
        head += f'<script type="text/javascript" src="{webpath(script.path)}"></script>\n'

    for script in scripts.list_scripts("javascript", ".mjs"):
        head += f'<script type="module" src="{webpath(script.path)}"></script>\n'

    if shared.cmd_opts.theme:
        head += f'<script type="text/javascript">set_theme(\"{shared.cmd_opts.theme}\");</script>\n'

    return head


def css_html():
    head = ""

    def stylesheet(fn):
        return f'<link rel="stylesheet" property="stylesheet" href="{webpath(fn)}">'

    for cssfile in scripts.list_files_with_name("style.css"):
        head += stylesheet(cssfile)

    user_css = os.path.join(data_path, "user.css")
    if os.path.exists(user_css):
        head += stylesheet(user_css)

    from modules.shared_gradio_themes import resolve_var
    light = resolve_var('background_fill_primary')
    dark = resolve_var('background_fill_primary_dark')
    head += f'<style>html {{ background-color: {light}; }} @media (prefers-color-scheme: dark) {{ html {{background-color:  {dark}; }} }}</style>'

    return head


def reload_javascript():
    js = javascript_html()
    css = css_html()

    def template_response(*args, **kwargs):
        res = shared.GradioTemplateResponseOriginal(*args, **kwargs)
        res.body = res.body.replace(b'</head>', f'{js}<meta name="referrer" content="no-referrer"/></head>'.encode("utf8"))
        res.body = res.body.replace(b'</body>', f'{css}</body>'.encode("utf8"))
        res.init_headers()
        return res

    gr.routes.templates.TemplateResponse = template_response


if not hasattr(shared, 'GradioTemplateResponseOriginal'):
    shared.GradioTemplateResponseOriginal = gr.routes.templates.TemplateResponse
