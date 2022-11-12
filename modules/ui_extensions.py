import json
import os.path
import shutil
import sys
import time
import traceback

import git

import gradio as gr
import html

from modules import extensions, shared, paths


available_extensions = {"extensions": []}


def check_access():
    assert not shared.cmd_opts.disable_extension_access, "extension access disabed because of commandline flags"


def apply_and_restart(disable_list, update_list):
    check_access()

    disabled = json.loads(disable_list)
    assert type(disabled) == list, f"wrong disable_list data for apply_and_restart: {disable_list}"

    update = json.loads(update_list)
    assert type(update) == list, f"wrong update_list data for apply_and_restart: {update_list}"

    update = set(update)

    for ext in extensions.extensions:
        if ext.name not in update:
            continue

        try:
            ext.pull()
        except Exception:
            print(f"Error pulling updates for {ext.name}:", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)

    shared.opts.disabled_extensions = disabled
    shared.opts.save(shared.config_filename)

    shared.state.interrupt()
    shared.state.need_restart = True


def check_updates():
    check_access()

    for ext in extensions.extensions:
        if ext.remote is None:
            continue

        try:
            ext.check_updates()
        except Exception:
            print(f"Error checking updates for {ext.name}:", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)

    return extension_table()


def extension_table():
    code = f"""<!-- {time.time()} -->
    <table id="extensions">
        <thead>
            <tr>
                <th><abbr title="Use checkbox to enable the extension; it will be enabled or disabled when you click apply button">Extension</abbr></th>
                <th>URL</th>
                <th><abbr title="Use checkbox to mark the extension for update; it will be updated when you click apply button">Update</abbr></th>
            </tr>
        </thead>
        <tbody>
    """

    for ext in extensions.extensions:
        if ext.can_update:
            ext_status = f"""<label><input class="gr-check-radio gr-checkbox" name="update_{html.escape(ext.name)}" checked="checked" type="checkbox">{html.escape(ext.status)}</label>"""
        else:
            ext_status = ext.status

        code += f"""
            <tr>
                <td><label><input class="gr-check-radio gr-checkbox" name="enable_{html.escape(ext.name)}" type="checkbox" {'checked="checked"' if ext.enabled else ''}>{html.escape(ext.name)}</label></td>
                <td><a href="{html.escape(ext.remote or '')}" target="_blank">{html.escape(ext.remote or '')}</a></td>
                <td{' class="extension_status"' if ext.remote is not None else ''}>{ext_status}</td>
            </tr>
    """

    code += """
        </tbody>
    </table>
    """

    return code


def normalize_git_url(url):
    if url is None:
        return ""

    url = url.replace(".git", "")
    return url


def install_extension_from_url(dirname, url):
    check_access()

    assert url, 'No URL specified'

    if dirname is None or dirname == "":
        *parts, last_part = url.split('/')
        last_part = normalize_git_url(last_part)

        dirname = last_part

    target_dir = os.path.join(extensions.extensions_dir, dirname)
    assert not os.path.exists(target_dir), f'Extension directory already exists: {target_dir}'

    normalized_url = normalize_git_url(url)
    assert len([x for x in extensions.extensions if normalize_git_url(x.remote) == normalized_url]) == 0, 'Extension with this URL is already installed'

    tmpdir = os.path.join(paths.script_path, "tmp", dirname)

    try:
        shutil.rmtree(tmpdir, True)

        repo = git.Repo.clone_from(url, tmpdir)
        repo.remote().fetch()

        os.rename(tmpdir, target_dir)

        import launch
        launch.run_extension_installer(target_dir)

        extensions.list_extensions()
        return [extension_table(), html.escape(f"Installed into {target_dir}. Use Installed tab to restart.")]
    finally:
        shutil.rmtree(tmpdir, True)


def install_extension_from_index(url, hide_tags):
    ext_table, message = install_extension_from_url(None, url)

    code, _ = refresh_available_extensions_from_data(hide_tags)

    return code, ext_table, message


def refresh_available_extensions(url, hide_tags):
    global available_extensions

    import urllib.request
    with urllib.request.urlopen(url) as response:
        text = response.read()

    available_extensions = json.loads(text)

    code, tags = refresh_available_extensions_from_data(hide_tags)

    return url, code, gr.CheckboxGroup.update(choices=tags), ''


def refresh_available_extensions_for_tags(hide_tags):
    code, _ = refresh_available_extensions_from_data(hide_tags)

    return code, ''


def refresh_available_extensions_from_data(hide_tags):
    extlist = available_extensions["extensions"]
    installed_extension_urls = {normalize_git_url(extension.remote): extension.name for extension in extensions.extensions}

    tags = available_extensions.get("tags", {})
    tags_to_hide = set(hide_tags)
    hidden = 0

    code = f"""<!-- {time.time()} -->
    <table id="available_extensions">
        <thead>
            <tr>
                <th>Extension</th>
                <th>Description</th>
                <th>Action</th>
            </tr>
        </thead>
        <tbody>
    """

    for ext in extlist:
        name = ext.get("name", "noname")
        url = ext.get("url", None)
        description = ext.get("description", "")
        extension_tags = ext.get("tags", [])

        if url is None:
            continue

        if len([x for x in extension_tags if x in tags_to_hide]) > 0:
            hidden += 1
            continue

        existing = installed_extension_urls.get(normalize_git_url(url), None)

        install_code = f"""<input onclick="install_extension_from_index(this, '{html.escape(url)}')" type="button" value="{"Install" if not existing else "Installed"}" {"disabled=disabled" if existing else ""} class="gr-button gr-button-lg gr-button-secondary">"""

        tags_text = ", ".join([f"<span class='extension-tag' title='{tags.get(x, '')}'>{x}</span>" for x in extension_tags])

        code += f"""
            <tr>
                <td><a href="{html.escape(url)}" target="_blank">{html.escape(name)}</a><br />{tags_text}</td>
                <td>{html.escape(description)}</td>
                <td>{install_code}</td>
            </tr>
    """

    code += """
        </tbody>
    </table>
    """

    if hidden > 0:
        code += f"<p>Extension hidden: {hidden}</p>"

    return code, list(tags)


def create_ui():
    import modules.ui

    with gr.Blocks(analytics_enabled=False) as ui:
        with gr.Tabs(elem_id="tabs_extensions") as tabs:
            with gr.TabItem("Installed"):

                with gr.Row():
                    apply = gr.Button(value="Apply and restart UI", variant="primary")
                    check = gr.Button(value="Check for updates")
                    extensions_disabled_list = gr.Text(elem_id="extensions_disabled_list", visible=False).style(container=False)
                    extensions_update_list = gr.Text(elem_id="extensions_update_list", visible=False).style(container=False)

                extensions_table = gr.HTML(lambda: extension_table())

                apply.click(
                    fn=apply_and_restart,
                    _js="extensions_apply",
                    inputs=[extensions_disabled_list, extensions_update_list],
                    outputs=[],
                )

                check.click(
                    fn=check_updates,
                    _js="extensions_check",
                    inputs=[],
                    outputs=[extensions_table],
                )

            with gr.TabItem("Available"):
                with gr.Row():
                    refresh_available_extensions_button = gr.Button(value="Load from:", variant="primary")
                    available_extensions_index = gr.Text(value="https://raw.githubusercontent.com/wiki/AUTOMATIC1111/stable-diffusion-webui/Extensions-index.md", label="Extension index URL").style(container=False)
                    extension_to_install = gr.Text(elem_id="extension_to_install", visible=False)
                    install_extension_button = gr.Button(elem_id="install_extension_button", visible=False)

                with gr.Row():
                    hide_tags = gr.CheckboxGroup(value=["ads", "localization"], label="Hide extensions with tags", choices=["script", "ads", "localization"])

                install_result = gr.HTML()
                available_extensions_table = gr.HTML()

                refresh_available_extensions_button.click(
                    fn=modules.ui.wrap_gradio_call(refresh_available_extensions, extra_outputs=[gr.update(), gr.update(), gr.update()]),
                    inputs=[available_extensions_index, hide_tags],
                    outputs=[available_extensions_index, available_extensions_table, hide_tags, install_result],
                )

                install_extension_button.click(
                    fn=modules.ui.wrap_gradio_call(install_extension_from_index, extra_outputs=[gr.update(), gr.update()]),
                    inputs=[extension_to_install, hide_tags],
                    outputs=[available_extensions_table, extensions_table, install_result],
                )

                hide_tags.change(
                    fn=modules.ui.wrap_gradio_call(refresh_available_extensions_for_tags, extra_outputs=[gr.update()]),
                    inputs=[hide_tags],
                    outputs=[available_extensions_table, install_result]
                )

            with gr.TabItem("Install from URL"):
                install_url = gr.Text(label="URL for extension's git repository")
                install_dirname = gr.Text(label="Local directory name", placeholder="Leave empty for auto")
                install_button = gr.Button(value="Install", variant="primary")
                install_result = gr.HTML(elem_id="extension_install_result")

                install_button.click(
                    fn=modules.ui.wrap_gradio_call(install_extension_from_url, extra_outputs=[gr.update()]),
                    inputs=[install_dirname, install_url],
                    outputs=[extensions_table, install_result],
                )

    return ui
