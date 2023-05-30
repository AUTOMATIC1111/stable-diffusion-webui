import os
import pathlib

from gradio.themes.utils import ThemeAsset


def create_theme_dropdown():
    import gradio as gr

    asset_path = pathlib.Path() / "themes"
    themes = []
    for theme_asset in os.listdir(str(asset_path)):
        themes.append(
            (ThemeAsset(theme_asset), gr.Theme.load(str(asset_path / theme_asset)))
        )

    def make_else_if(theme_asset):
        return f"""
        else if (theme == '{str(theme_asset[0].version)}') {{
            var theme_css = `{theme_asset[1]._get_theme_css()}`
        }}"""

    head, tail = themes[0], themes[1:]
    if_statement = f"""
        if (theme == "{str(head[0].version)}") {{
            var theme_css = `{head[1]._get_theme_css()}`
        }} {" ".join(make_else_if(t) for t in tail)}
    """

    latest_to_oldest = sorted([t[0] for t in themes], key=lambda asset: asset.version)[
        ::-1
    ]
    latest_to_oldest = [str(t.version) for t in latest_to_oldest]

    component = gr.Dropdown(
        choices=latest_to_oldest,
        value=latest_to_oldest[0],
        render=False,
        label="Select Version",
    ).style(container=False)

    return (
        component,
        f"""
        (theme) => {{
            if (!document.querySelector('.theme-css')) {{
                var theme_elem = document.createElement('style');
                theme_elem.classList.add('theme-css');
                document.head.appendChild(theme_elem);
            }} else {{
                var theme_elem = document.querySelector('.theme-css');
            }}
            {if_statement}
            theme_elem.innerHTML = theme_css;
        }}
    """,
    )
