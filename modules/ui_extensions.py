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
                <td><a href="{html.escape(ext.remote or '')}">{html.escape(ext.remote or '')}</a></td>
                <td{' class="extension_status"' if ext.remote is not None else ''}>{ext_status}</td>
            </tr>
    """

    code += """
        </tbody>
    </table>
    """

    return code


def install_extension_from_url(dirname, url):
    check_access()

    assert url, 'No URL specified'

    if dirname is None or dirname == "":
        *parts, last_part = url.split('/')
        last_part = last_part.replace(".git", "")

        dirname = last_part

    target_dir = os.path.join(extensions.extensions_dir, dirname)
    assert not os.path.exists(target_dir), f'Extension directory already exists: {target_dir}'

    assert len([x for x in extensions.extensions if x.remote == url]) == 0, 'Extension with this URL is already installed'

    tmpdir = os.path.join(paths.script_path, "tmp", dirname)

    try:
        shutil.rmtree(tmpdir, True)

        repo = git.Repo.clone_from(url, tmpdir)
        repo.remote().fetch()

        os.rename(tmpdir, target_dir)

        extensions.list_extensions()
        return [extension_table(), html.escape(f"Installed into {target_dir}. Use Installed tab to restart.")]
    finally:
        shutil.rmtree(tmpdir, True)


def create_ui():
    import modules.ui

    with gr.Blocks(analytics_enabled=False) as ui:
        with gr.Tabs(elem_id="tabs_extensions") as tabs:
            with gr.TabItem("Installed"):
                extensions_disabled_list = gr.Text(elem_id="extensions_disabled_list", visible=False)
                extensions_update_list = gr.Text(elem_id="extensions_update_list", visible=False)

                with gr.Row():
                    apply = gr.Button(value="Apply and restart UI", variant="primary")
                    check = gr.Button(value="Check for updates")

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

            with gr.TabItem("Install from URL"):
                install_url = gr.Text(label="URL for extension's git repository")
                install_dirname = gr.Text(label="Local directory name", placeholder="Leave empty for auto")
                intall_button = gr.Button(value="Install", variant="primary")
                intall_result = gr.HTML(elem_id="extension_install_result")

                intall_button.click(
                    fn=modules.ui.wrap_gradio_call(install_extension_from_url, extra_outputs=[gr.update()]),
                    inputs=[install_dirname, install_url],
                    outputs=[extensions_table, intall_result],
                )

    return ui
