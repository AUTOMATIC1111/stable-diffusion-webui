import json
import os.path
import sys
import time
from datetime import datetime
import traceback

import git

import gradio as gr
import html
import shutil
import errno

from modules import extensions, shared, paths, config_states
from modules.paths_internal import config_states_dir
from modules.call_queue import wrap_gradio_gpu_call

available_extensions = {"extensions": []}
STYLE_PRIMARY = ' style="color: var(--primary-400)"'


def check_access():
    assert not shared.cmd_opts.disable_extension_access, "extension access disabled because of command line flags"


def apply_and_restart(disable_list, update_list, disable_all):
    check_access()

    disabled = json.loads(disable_list)
    assert type(disabled) == list, f"wrong disable_list data for apply_and_restart: {disable_list}"

    update = json.loads(update_list)
    assert type(update) == list, f"wrong update_list data for apply_and_restart: {update_list}"

    if update:
        save_config_state("Backup (pre-update)")

    update = set(update)

    for ext in extensions.extensions:
        if ext.name not in update:
            continue

        try:
            ext.fetch_and_reset_hard()
        except Exception:
            print(f"Error getting updates for {ext.name}:", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)

    shared.opts.disabled_extensions = disabled
    shared.opts.disable_all_extensions = disable_all
    shared.opts.save(shared.config_filename)

    shared.state.interrupt()
    shared.state.need_restart = True


def save_config_state(name):
    current_config_state = config_states.get_config()
    if not name:
        name = "Config"
    current_config_state["name"] = name
    timestamp = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    filename = os.path.join(config_states_dir, f"{timestamp}_{name}.json")
    print(f"Saving backup of webui/extension state to {filename}.")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(current_config_state, f)
    config_states.list_config_states()
    new_value = next(iter(config_states.all_config_states.keys()), "Current")
    new_choices = ["Current"] + list(config_states.all_config_states.keys())
    return gr.Dropdown.update(value=new_value, choices=new_choices), f"<span>Saved current webui/extension state to \"{filename}\"</span>"


def restore_config_state(confirmed, config_state_name, restore_type):
    if config_state_name == "Current":
        return "<span>Select a config to restore from.</span>"
    if not confirmed:
        return "<span>Cancelled.</span>"

    check_access()

    config_state = config_states.all_config_states[config_state_name]

    print(f"*** Restoring webui state from backup: {restore_type} ***")

    if restore_type == "extensions" or restore_type == "both":
        shared.opts.restore_config_state_file = config_state["filepath"]
        shared.opts.save(shared.config_filename)

    if restore_type == "webui" or restore_type == "both":
        config_states.restore_webui_config(config_state)

    shared.state.interrupt()
    shared.state.need_restart = True

    return ""


def check_updates(id_task, disable_list):
    check_access()

    disabled = json.loads(disable_list)
    assert type(disabled) == list, f"wrong disable_list data for apply_and_restart: {disable_list}"

    exts = [ext for ext in extensions.extensions if ext.remote is not None and ext.name not in disabled]
    shared.state.job_count = len(exts)

    for ext in exts:
        shared.state.textinfo = ext.name

        try:
            ext.check_updates()
        except FileNotFoundError as e:
            if 'FETCH_HEAD' not in str(e):
                raise
        except Exception:
            print(f"Error checking updates for {ext.name}:", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)

        shared.state.nextjob()

    return extension_table(), ""


def make_commit_link(commit_hash, remote, text=None):
    if text is None:
        text = commit_hash[:8]
    if remote.startswith("https://github.com/"):
        href = os.path.join(remote, "commit", commit_hash)
        return f'<a href="{href}" target="_blank">{text}</a>'
    else:
        return text


def extension_table():
    code = f"""<!-- {time.time()} -->
    <table id="extensions">
        <thead>
            <tr>
                <th><abbr title="Use checkbox to enable the extension; it will be enabled or disabled when you click apply button">Extension</abbr></th>
                <th>URL</th>
                <th><abbr title="Extension version">Version</abbr></th>
                <th><abbr title="Use checkbox to mark the extension for update; it will be updated when you click apply button">Update</abbr></th>
            </tr>
        </thead>
        <tbody>
    """

    for ext in extensions.extensions:
        ext.read_info_from_repo()

        remote = f"""<a href="{html.escape(ext.remote or '')}" target="_blank">{html.escape("built-in" if ext.is_builtin else ext.remote or '')}</a>"""

        if ext.can_update:
            ext_status = f"""<label><input class="gr-check-radio gr-checkbox" name="update_{html.escape(ext.name)}" checked="checked" type="checkbox">{html.escape(ext.status)}</label>"""
        else:
            ext_status = ext.status

        style = ""
        if shared.opts.disable_all_extensions == "extra" and not ext.is_builtin or shared.opts.disable_all_extensions == "all":
            style = STYLE_PRIMARY

        version_link = ext.version
        if ext.commit_hash and ext.remote:
            version_link = make_commit_link(ext.commit_hash, ext.remote, ext.version)

        code += f"""
            <tr>
                <td><label{style}><input class="gr-check-radio gr-checkbox" name="enable_{html.escape(ext.name)}" type="checkbox" {'checked="checked"' if ext.enabled else ''}>{html.escape(ext.name)}</label></td>
                <td>{remote}</td>
                <td>{version_link}</td>
                <td{' class="extension_status"' if ext.remote is not None else ''}>{ext_status}</td>
            </tr>
    """

    code += """
        </tbody>
    </table>
    """

    return code


def update_config_states_table(state_name):
    if state_name == "Current":
        config_state = config_states.get_config()
    else:
        config_state = config_states.all_config_states[state_name]

    config_name = config_state.get("name", "Config")
    created_date = time.asctime(time.gmtime(config_state["created_at"]))
    filepath = config_state.get("filepath", "<unknown>")

    code = f"""<!-- {time.time()} -->"""

    webui_remote = config_state["webui"]["remote"] or ""
    webui_branch = config_state["webui"]["branch"]
    webui_commit_hash = config_state["webui"]["commit_hash"] or "<unknown>"
    webui_commit_date = config_state["webui"]["commit_date"]
    if webui_commit_date:
        webui_commit_date = time.asctime(time.gmtime(webui_commit_date))
    else:
        webui_commit_date = "<unknown>"

    remote = f"""<a href="{html.escape(webui_remote)}" target="_blank">{html.escape(webui_remote or '')}</a>"""
    commit_link = make_commit_link(webui_commit_hash, webui_remote)
    date_link = make_commit_link(webui_commit_hash, webui_remote, webui_commit_date)

    current_webui = config_states.get_webui_config()

    style_remote = ""
    style_branch = ""
    style_commit = ""
    if current_webui["remote"] != webui_remote:
        style_remote = STYLE_PRIMARY
    if current_webui["branch"] != webui_branch:
        style_branch = STYLE_PRIMARY
    if current_webui["commit_hash"] != webui_commit_hash:
        style_commit = STYLE_PRIMARY

    code += f"""<h2>Config Backup: {config_name}</h2>
      <div><b>Filepath:</b> {filepath}</div>
      <div><b>Created at:</b> {created_date}</div>"""

    code += f"""<h2>WebUI State</h2>
      <table id="config_state_webui">
        <thead>
            <tr>
                <th>URL</th>
                <th>Branch</th>
                <th>Commit</th>
                <th>Date</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><label{style_remote}>{remote}</label></td>
                <td><label{style_branch}>{webui_branch}</label></td>
                <td><label{style_commit}>{commit_link}</label></td>
                <td><label{style_commit}>{date_link}</label></td>
            </tr>
        </tbody>
      </table>
    """

    code += """<h2>Extension State</h2>
      <table id="config_state_extensions">
        <thead>
            <tr>
                <th>Extension</th>
                <th>URL</th>
                <th>Branch</th>
                <th>Commit</th>
                <th>Date</th>
            </tr>
        </thead>
        <tbody>
    """

    ext_map = {ext.name: ext for ext in extensions.extensions}

    for ext_name, ext_conf in config_state["extensions"].items():
        ext_remote = ext_conf["remote"] or ""
        ext_branch = ext_conf["branch"] or "<unknown>"
        ext_enabled = ext_conf["enabled"]
        ext_commit_hash = ext_conf["commit_hash"] or "<unknown>"
        ext_commit_date = ext_conf["commit_date"]
        if ext_commit_date:
            ext_commit_date = time.asctime(time.gmtime(ext_commit_date))
        else:
            ext_commit_date = "<unknown>"

        remote = f"""<a href="{html.escape(ext_remote)}" target="_blank">{html.escape(ext_remote or '')}</a>"""
        commit_link = make_commit_link(ext_commit_hash, ext_remote)
        date_link = make_commit_link(ext_commit_hash, ext_remote, ext_commit_date)

        style_enabled = ""
        style_remote = ""
        style_branch = ""
        style_commit = ""
        if ext_name in ext_map:
            current_ext = ext_map[ext_name]
            current_ext.read_info_from_repo()
            if current_ext.enabled != ext_enabled:
                style_enabled = STYLE_PRIMARY
            if current_ext.remote != ext_remote:
                style_remote = STYLE_PRIMARY
            if current_ext.branch != ext_branch:
                style_branch = STYLE_PRIMARY
            if current_ext.commit_hash != ext_commit_hash:
                style_commit = STYLE_PRIMARY

        code += f"""
            <tr>
                <td><label{style_enabled}><input class="gr-check-radio gr-checkbox" type="checkbox" disabled="true" {'checked="checked"' if ext_enabled else ''}>{html.escape(ext_name)}</label></td>
                <td><label{style_remote}>{remote}</label></td>
                <td><label{style_branch}>{ext_branch}</label></td>
                <td><label{style_commit}>{commit_link}</label></td>
                <td><label{style_commit}>{date_link}</label></td>
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


def install_extension_from_url(dirname, url, branch_name=None):
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

    tmpdir = os.path.join(paths.data_path, "tmp", dirname)

    try:
        shutil.rmtree(tmpdir, True)
        if not branch_name:
            # if no branch is specified, use the default branch
            with git.Repo.clone_from(url, tmpdir) as repo:
                repo.remote().fetch()
                for submodule in repo.submodules:
                    submodule.update()
        else:
            with git.Repo.clone_from(url, tmpdir, branch=branch_name) as repo:
                repo.remote().fetch()
                for submodule in repo.submodules:
                    submodule.update()
        try:
            os.rename(tmpdir, target_dir)
        except OSError as err:
            if err.errno == errno.EXDEV:
                # Cross device link, typical in docker or when tmp/ and extensions/ are on different file systems
                # Since we can't use a rename, do the slower but more versitile shutil.move()
                shutil.move(tmpdir, target_dir)
            else:
                # Something else, not enough free space, permissions, etc.  rethrow it so that it gets handled.
                raise err

        import launch
        launch.run_extension_installer(target_dir)

        extensions.list_extensions()
        return [extension_table(), html.escape(f"Installed into {target_dir}. Use Installed tab to restart.")]
    finally:
        shutil.rmtree(tmpdir, True)


def install_extension_from_index(url, hide_tags, sort_column, filter_text):
    ext_table, message = install_extension_from_url(None, url)

    code, _ = refresh_available_extensions_from_data(hide_tags, sort_column, filter_text)

    return code, ext_table, message, ''


def refresh_available_extensions(url, hide_tags, sort_column):
    global available_extensions

    import urllib.request
    with urllib.request.urlopen(url) as response:
        text = response.read()

    available_extensions = json.loads(text)

    code, tags = refresh_available_extensions_from_data(hide_tags, sort_column)

    return url, code, gr.CheckboxGroup.update(choices=tags), '', ''


def refresh_available_extensions_for_tags(hide_tags, sort_column, filter_text):
    code, _ = refresh_available_extensions_from_data(hide_tags, sort_column, filter_text)

    return code, ''


def search_extensions(filter_text, hide_tags, sort_column):
    code, _ = refresh_available_extensions_from_data(hide_tags, sort_column, filter_text)

    return code, ''


sort_ordering = [
    # (reverse, order_by_function)
    (True, lambda x: x.get('added', 'z')),
    (False, lambda x: x.get('added', 'z')),
    (False, lambda x: x.get('name', 'z')),
    (True, lambda x: x.get('name', 'z')),
    (False, lambda x: 'z'),
]


def refresh_available_extensions_from_data(hide_tags, sort_column, filter_text=""):
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

    sort_reverse, sort_function = sort_ordering[sort_column if 0 <= sort_column < len(sort_ordering) else 0]

    for ext in sorted(extlist, key=sort_function, reverse=sort_reverse):
        name = ext.get("name", "noname")
        added = ext.get('added', 'unknown')
        url = ext.get("url", None)
        description = ext.get("description", "")
        extension_tags = ext.get("tags", [])

        if url is None:
            continue

        existing = installed_extension_urls.get(normalize_git_url(url), None)
        extension_tags = extension_tags + ["installed"] if existing else extension_tags

        if len([x for x in extension_tags if x in tags_to_hide]) > 0:
            hidden += 1
            continue

        if filter_text and filter_text.strip():
            if filter_text.lower() not in html.escape(name).lower() and filter_text.lower() not in html.escape(description).lower():
                hidden += 1
                continue

        install_code = f"""<button onclick="install_extension_from_index(this, '{html.escape(url)}')" {"disabled=disabled" if existing else ""} class="lg secondary gradio-button custom-button">{"Install" if not existing else "Installed"}</button>"""

        tags_text = ", ".join([f"<span class='extension-tag' title='{tags.get(x, '')}'>{x}</span>" for x in extension_tags])

        code += f"""
            <tr>
                <td><a href="{html.escape(url)}" target="_blank">{html.escape(name)}</a><br />{tags_text}</td>
                <td>{html.escape(description)}<p class="info"><span class="date_added">Added: {html.escape(added)}</span></p></td>
                <td>{install_code}</td>
            </tr>
        
        """

        for tag in [x for x in extension_tags if x not in tags]:
            tags[tag] = tag

    code += """
        </tbody>
    </table>
    """

    if hidden > 0:
        code += f"<p>Extension hidden: {hidden}</p>"

    return code, list(tags)


def create_ui():
    import modules.ui

    config_states.list_config_states()

    with gr.Blocks(analytics_enabled=False) as ui:
        with gr.Tabs(elem_id="tabs_extensions") as tabs:
            with gr.TabItem("Installed", id="installed"):

                with gr.Row(elem_id="extensions_installed_top"):
                    apply = gr.Button(value="Apply and restart UI", variant="primary")
                    check = gr.Button(value="Check for updates")
                    extensions_disable_all = gr.Radio(label="Disable all extensions", choices=["none", "extra", "all"], value=shared.opts.disable_all_extensions, elem_id="extensions_disable_all")
                    extensions_disabled_list = gr.Text(elem_id="extensions_disabled_list", visible=False).style(container=False)
                    extensions_update_list = gr.Text(elem_id="extensions_update_list", visible=False).style(container=False)

                html = ""
                if shared.opts.disable_all_extensions != "none":
                    html = """
<span style="color: var(--primary-400);">
    "Disable all extensions" was set, change it to "none" to load all extensions again
</span>
                    """
                info = gr.HTML(html)
                extensions_table = gr.HTML(lambda: extension_table())

                apply.click(
                    fn=apply_and_restart,
                    _js="extensions_apply",
                    inputs=[extensions_disabled_list, extensions_update_list, extensions_disable_all],
                    outputs=[],
                )

                check.click(
                    fn=wrap_gradio_gpu_call(check_updates, extra_outputs=[gr.update()]),
                    _js="extensions_check",
                    inputs=[info, extensions_disabled_list],
                    outputs=[extensions_table, info],
                )

            with gr.TabItem("Available", id="available"):
                with gr.Row():
                    refresh_available_extensions_button = gr.Button(value="Load from:", variant="primary")
                    available_extensions_index = gr.Text(value="https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui-extensions/master/index.json", label="Extension index URL").style(container=False)
                    extension_to_install = gr.Text(elem_id="extension_to_install", visible=False)
                    install_extension_button = gr.Button(elem_id="install_extension_button", visible=False)

                with gr.Row():
                    hide_tags = gr.CheckboxGroup(value=["ads", "localization", "installed"], label="Hide extensions with tags", choices=["script", "ads", "localization", "installed"])
                    sort_column = gr.Radio(value="newest first", label="Order", choices=["newest first", "oldest first", "a-z", "z-a", "internal order", ], type="index")

                with gr.Row(): 
                    search_extensions_text = gr.Text(label="Search").style(container=False)
                   
                install_result = gr.HTML()
                available_extensions_table = gr.HTML()

                refresh_available_extensions_button.click(
                    fn=modules.ui.wrap_gradio_call(refresh_available_extensions, extra_outputs=[gr.update(), gr.update(), gr.update()]),
                    inputs=[available_extensions_index, hide_tags, sort_column],
                    outputs=[available_extensions_index, available_extensions_table, hide_tags, install_result, search_extensions_text],
                )

                install_extension_button.click(
                    fn=modules.ui.wrap_gradio_call(install_extension_from_index, extra_outputs=[gr.update(), gr.update()]),
                    inputs=[extension_to_install, hide_tags, sort_column, search_extensions_text],
                    outputs=[available_extensions_table, extensions_table, install_result],
                )

                search_extensions_text.change(
                    fn=modules.ui.wrap_gradio_call(search_extensions, extra_outputs=[gr.update()]),
                    inputs=[search_extensions_text, hide_tags, sort_column],
                    outputs=[available_extensions_table, install_result],
                )

                hide_tags.change(
                    fn=modules.ui.wrap_gradio_call(refresh_available_extensions_for_tags, extra_outputs=[gr.update()]),
                    inputs=[hide_tags, sort_column, search_extensions_text],
                    outputs=[available_extensions_table, install_result]
                )

                sort_column.change(
                    fn=modules.ui.wrap_gradio_call(refresh_available_extensions_for_tags, extra_outputs=[gr.update()]),
                    inputs=[hide_tags, sort_column, search_extensions_text],
                    outputs=[available_extensions_table, install_result]
                )

            with gr.TabItem("Install from URL", id="install_from_url"):
                install_url = gr.Text(label="URL for extension's git repository")
                install_branch = gr.Text(label="Specific branch name", placeholder="Leave empty for default main branch")
                install_dirname = gr.Text(label="Local directory name", placeholder="Leave empty for auto")
                install_button = gr.Button(value="Install", variant="primary")
                install_result = gr.HTML(elem_id="extension_install_result")

                install_button.click(
                    fn=modules.ui.wrap_gradio_call(install_extension_from_url, extra_outputs=[gr.update()]),
                    inputs=[install_dirname, install_url, install_branch],
                    outputs=[extensions_table, install_result],
                )

            with gr.TabItem("Backup/Restore"):
                with gr.Row(elem_id="extensions_backup_top_row"):
                    config_states_list = gr.Dropdown(label="Saved Configs", elem_id="extension_backup_saved_configs", value="Current", choices=["Current"] + list(config_states.all_config_states.keys()))
                    modules.ui.create_refresh_button(config_states_list, config_states.list_config_states, lambda: {"choices": ["Current"] + list(config_states.all_config_states.keys())}, "refresh_config_states")
                    config_restore_type = gr.Radio(label="State to restore", choices=["extensions", "webui", "both"], value="extensions", elem_id="extension_backup_restore_type")
                    config_restore_button = gr.Button(value="Restore Selected Config", variant="primary", elem_id="extension_backup_restore")
                with gr.Row(elem_id="extensions_backup_top_row2"):
                    config_save_name = gr.Textbox("", placeholder="Config Name", show_label=False)
                    config_save_button = gr.Button(value="Save Current Config")

                config_states_info = gr.HTML("")
                config_states_table = gr.HTML(lambda: update_config_states_table("Current"))

                config_save_button.click(fn=save_config_state, inputs=[config_save_name], outputs=[config_states_list, config_states_info])

                dummy_component = gr.Label(visible=False)
                config_restore_button.click(fn=restore_config_state, _js="config_state_confirm_restore", inputs=[dummy_component, config_states_list, config_restore_type], outputs=[config_states_info])

                config_states_list.change(
                    fn=update_config_states_table,
                    inputs=[config_states_list],
                    outputs=[config_states_table],
                )

    return ui
