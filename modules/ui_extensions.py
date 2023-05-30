import json
import os.path
import shutil
import errno
import html
from datetime import datetime
import git
import gradio as gr
from modules import extensions, shared, paths, errors
from modules.call_queue import wrap_gradio_gpu_call


extensions_index = "https://vladmandic.github.io/sd-data/pages/extensions.json"
hide_tags = ["localization"]
extensions_list = []
sort_ordering = {
    "default": (True, lambda x: x.get('sort_default', '')),
    "user extensions": (True, lambda x: x.get('sort_user', '')),
    "update avilable": (True, lambda x: x.get('sort_update', '')),
    "updated date": (True, lambda x: x.get('updated', '2000-01-01T00:00')),
    "created date": (True, lambda x: x.get('created', '2000-01-01T00:00')),
    "name": (False, lambda x: x.get('name', '').lower()),
    "enabled": (False, lambda x: x.get('sort_enabled', '').lower()),
    "size": (True, lambda x: x.get('size', 0)),
    "stars": (True, lambda x: x.get('stars', 0)),
    "commits": (True, lambda x: x.get('commits', 0)),
    "issues": (True, lambda x: x.get('issues', 0)),
}


def update_extension_list():
    global extensions_list # pylint: disable=global-statement
    try:
        with open(os.path.join(paths.script_path, "html", "extensions.json"), "r", encoding="utf-8") as f:
            extensions_list = json.loads(f.read())
            shared.log.debug(f'Extensions list loaded: {os.path.join(paths.script_path, "html", "extensions.json")}')
    except:
        shared.log.debug(f'Extensions list failed to load: {os.path.join(paths.script_path, "html", "extensions.json")}')
    found = []
    for ext in extensions.extensions:
        ext.read_info_from_repo()
    for ext in extensions_list:
        installed = [extension for extension in extensions.extensions
                     if extension.git_name == ext['name']
                     or extension.name == ext['name']
                     or (extension.remote or '').startswith(ext['url'].replace('.git', ''))]
        if len(installed) > 0:
            found.append(installed[0])
    not_matched = [extension for extension in extensions.extensions if extension not in found]
    for ext in not_matched:
        entry = {
            "name": ext.name or "",
            "description": ext.description or "",
            "url": ext.remote or "",
            "tags": [],
            "stars": 0,
            "issues": 0,
            "commits": 0,
            "size": 0,
            "long": ext.git_name or ext.name or "",
            "added": ext.ctime,
            "created": ext.ctime,
            "updated": ext.mtime,
        }
        extensions_list.append(entry)


def check_access():
    assert not shared.cmd_opts.disable_extension_access, "extension access disabled because of command line flags"


def apply_and_restart(disable_list, update_list, disable_all):
    check_access()
    shared.log.debug(f'Extensions apply: disable={disable_list} update={update_list}')
    disabled = json.loads(disable_list)
    assert type(disabled) == list, f"wrong disable_list data for apply_and_restart: {disable_list}"
    update = json.loads(update_list)
    assert type(update) == list, f"wrong update_list data for apply_and_restart: {update_list}"
    update = set(update)
    for ext in extensions.extensions:
        if ext.name not in update:
            continue
        try:
            ext.fetch_and_reset_hard()
        except Exception as e:
            errors.display(e, f'extensions apply update: {ext.name}')
    shared.opts.disabled_extensions = disabled
    shared.opts.disable_all_extensions = disable_all
    shared.opts.save(shared.config_filename)
    shared.restart_server(restart=True)


def check_updates(_id_task, disable_list, search_text, sort_column):
    check_access()
    disabled = json.loads(disable_list)
    assert type(disabled) == list, f"wrong disable_list data for apply_and_restart: {disable_list}"
    exts = [ext for ext in extensions.extensions if ext.remote is not None and ext.name not in disabled]
    shared.log.info(f'Extensions update check: update={len(exts)} disabled={len(disable_list)}')
    shared.state.job_count = len(exts)
    for ext in exts:
        shared.state.textinfo = ext.name
        try:
            ext.check_updates()
            if ext.can_update:
                ext.fetch_and_reset_hard()
                ext.read_info_from_repo()
                commit_date = ext.commit_date or 1577836800
                shared.log.info(f'Extensions updated: {ext.name} {ext.commit_hash[:8]} {datetime.utcfromtimestamp(commit_date)}')
            else:
                commit_date = ext.commit_date or 1577836800
                shared.log.debug(f'Extensions no update available: {ext.name} {ext.commit_hash[:8]} {datetime.utcfromtimestamp(commit_date)}')
        except FileNotFoundError as e:
            if 'FETCH_HEAD' not in str(e):
                raise
        except Exception:
            errors.display(e, f'extensions check update: {ext.name}')
        shared.state.nextjob()
    return refresh_extensions_list_from_data(search_text, sort_column), "Extension update complete | Restart required"


def make_commit_link(commit_hash, remote, text=None):
    if text is None:
        text = commit_hash[:8]
    if remote.startswith("https://github.com/"):
        href = os.path.join(remote, "commit", commit_hash)
        return f'<a href="{href}" target="_blank">{text}</a>'
    else:
        return text


def normalize_git_url(url):
    if url is None:
        return ""
    url = url.replace(".git", "")
    return url


def install_extension_from_url(dirname, url, branch_name, search_text, sort_column):
    check_access()
    assert url, 'No URL specified'
    if dirname is None or dirname == "":
        *parts, last_part = url.split('/') # pylint: disable=unused-variable
        last_part = normalize_git_url(last_part)
        dirname = last_part
    target_dir = os.path.join(extensions.extensions_dir, dirname)
    shared.log.info(f'Installing extension: {url} into {target_dir}')
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
                shutil.move(tmpdir, target_dir)
            else:
                raise err
        from launch import run_extension_installer
        run_extension_installer(target_dir)
        extensions.list_extensions()
        return [refresh_extensions_list_from_data(search_text, sort_column), html.escape(f"Extension installed: {target_dir} | Restart required")]
    finally:
        shutil.rmtree(tmpdir, True)


def install_extension(extension_to_install, search_text, sort_column):
    shared.log.info(f'Extension install: {extension_to_install}')
    code, message = install_extension_from_url(None, extension_to_install, None, search_text, sort_column)
    return code, message


def uninstall_extension(extension_path, search_text, sort_column):
    def errorRemoveReadonly(func, path, exc):
        import stat
        excvalue = exc[1]
        shared.log.debug(f'Exception during cleanup: {func} {path} {excvalue.strerror}')
        if func in (os.rmdir, os.remove, os.unlink) and excvalue.errno == errno.EACCES:
            shared.log.debug(f'Retrying cleanup: {path}')
            os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            func(path)

    ext = [extension for extension in extensions.extensions if os.path.abspath(extension.path) == os.path.abspath(extension_path)]
    if len(ext) > 0 and os.path.isdir(extension_path):
        found = ext[0]
        try:
            shutil.rmtree(found.path, ignore_errors=False, onerror=errorRemoveReadonly)
        except Exception as e:
            shared.log.warning(f'Extension uninstall failed: {found.path} {e}')
        extensions.extensions = [extension for extension in extensions.extensions if os.path.abspath(found.path) != os.path.abspath(extension_path)]
        update_extension_list()
        code = refresh_extensions_list_from_data(search_text, sort_column)
        shared.log.info(f'Extension uninstalled: {found.path}')
        return code, f"Extension uninstalled: {found.path} | Restart required"
    else:
        shared.log.warning(f'Extension uninstall cannot find extension: {extension_path}')
        code = refresh_extensions_list_from_data(search_text, sort_column)
        return code, f"Extension uninstalled failed: {extension_path}"


def update_extension(extension_path, search_text, sort_column):
    exts = [extension for extension in extensions.extensions if os.path.abspath(extension.path) == os.path.abspath(extension_path)]
    shared.state.job_count = len(exts)
    for ext in exts:
        shared.log.debug(f'Extensions update start: {ext.name} {ext.commit_hash} {ext.commit_date}')
        shared.state.textinfo = ext.name
        try:
            ext.check_updates()
            if ext.can_update:
                ext.fetch_and_reset_hard()
                ext.read_info_from_repo()
                commit_date = ext.commit_date or 1577836800
                shared.log.info(f'Extensions updated: {ext.name} {ext.commit_hash[:8]} {datetime.utcfromtimestamp(commit_date)}')
            else:
                commit_date = ext.commit_date or 1577836800
                shared.log.info(f'Extensions no update available: {ext.name} {ext.commit_hash[:8]} {datetime.utcfromtimestamp(commit_date)}')
        except FileNotFoundError as e:
            if 'FETCH_HEAD' not in str(e):
                raise
        except Exception as e:
            shared.log.error(f'Extensions update failed: {ext.name}')
            errors.display(e, f'extensions check update: {ext.name}')
        shared.log.debug(f'Extensions update finish: {ext.name} {ext.commit_hash} {ext.commit_date}')
        shared.state.nextjob()
    return refresh_extensions_list_from_data(search_text, sort_column), f"Extension updated | {extension_path} | Restart required"


def refresh_extensions_list(search_text, sort_column):
    global extensions_list # pylint: disable=global-statement
    import urllib.request
    try:
        with urllib.request.urlopen(extensions_index) as response:
            text = response.read()
        extensions_list = json.loads(text)
        with open(os.path.join(paths.script_path, "html", "extensions.json"), "w", encoding="utf-8") as outfile:
            json_object = json.dumps(extensions_list, indent=2)
            outfile.write(json_object)
            shared.log.debug(f'Updated extensions list: {len(extensions_list)} {extensions_index} {outfile}')
    except Exception as e:
        shared.log.warning(f'Updated extensions list failed: {extensions_index} {e}')
    update_extension_list()
    code = refresh_extensions_list_from_data(search_text, sort_column)
    return code, f'Extensions | {len(extensions.extensions)} registered | {len(extensions_list)} available'


def search_extensions(search_text, sort_column):
    code = refresh_extensions_list_from_data(search_text, sort_column)
    return code, f'Search | {search_text} | {sort_column}'


def refresh_extensions_list_from_data(search_text, sort_column):
    shared.log.debug(f'Extensions manager: refresh list search="{search_text}" sort="{sort_column}"')
    code = """
        <table id="extensions">
            <colgroup>
                <col style="width: 1%; background: var(--table-border-color)">
                <col style="width: 20%; background: var(--table-border-color)">
                <col style="width: 59%;">
                <col style="width: 5%; background: var(--panel-background-fill)">
                <col style="width: 10%; background: var(--panel-background-fill)">
                <col style="width: 5%; background: var(--table-border-color)">
            </colgroup>
            <thead style="font-size: 110%; border-style: solid; border-bottom: 1px var(--button-primary-border-color) solid">
            <tr>
                <th>Enabled</th>
                <th>Extension</th>
                <th>Description</th>
                <th>Type</th>
                <th>Current version</th>
                <th></th>
            </tr>
        </thead>
        <tbody>"""
    for ext in extensions_list:
        extension = [extension for extension in extensions.extensions if extension.git_name == ext['name'] or extension.name == ext['name']]
        if len(extension) > 0:
            extension[0].read_info_from_repo()
        ext['installed'] = len(extension) > 0
        ext['commit_date'] = extension[0].commit_date if len(extension) > 0 else 1577836800
        ext['is_builtin'] = extension[0].is_builtin if len(extension) > 0 else False
        ext['version'] = extension[0].version if len(extension) > 0 else ''
        ext['enabled'] = extension[0].enabled if len(extension) > 0 else ''
        ext['remote'] = extension[0].remote if len(extension) > 0 else None
        ext['path'] = extension[0].path if len(extension) > 0 else ''
        ext['sort_default'] = f"{'1' if ext['is_builtin'] else '0'}{'1' if ext['installed'] else '0'}{ext.get('updated', '2000-01-01T00:00')}"
    sort_reverse, sort_function = sort_ordering[sort_column]

    def dt(x: str):
        val = ext.get(x, None)
        if val is not None:
            return datetime.fromisoformat(val[:-1]).strftime('%a %b%d %Y %H:%M')
        else:
            return "N/A"

    for ext in sorted(extensions_list, key=sort_function, reverse=sort_reverse):
        name = ext.get("name", "unknown")
        added = dt('added')
        created = dt('created')
        pushed = dt('pushed')
        updated = dt('updated')
        url = ext.get('url', None)
        size = ext.get('size', 0)
        stars = ext.get('stars', 0)
        issues = ext.get('issues', 0)
        commits = ext.get('commits', 0)
        description = ext.get("description", "")
        installed = ext.get("installed", False)
        enabled = ext.get("enabled", False)
        path = ext.get("path", "")
        remote = ext.get("remote", None)
        commit_date = ext.get("commit_date", 1577836800) or 1577836800
        update_available = (remote is not None) & (installed) & (datetime.utcfromtimestamp(commit_date + 60 * 60) < datetime.fromisoformat(ext.get('updated', '2000-01-01T00:00:00.000Z')[:-1]))
        ext['sort_user'] = f"{'0' if ext['is_builtin'] else '1'}{'1' if ext['installed'] else '0'}{ext.get('name', '')}"
        ext['sort_enabled'] = f"{'0' if ext['enabled'] else '1'}{'1' if ext['is_builtin'] else '0'}{'1' if ext['installed'] else '0'}{ext.get('updated', '2000-01-01T00:00')}"
        ext['sort_update'] = f"{'1' if update_available else '0'}{'1' if ext['installed'] else '0'}{ext.get('updated', '2000-01-01T00:00')}"
        tags = ext.get("tags", [])
        tags_string = ' '.join(tags)
        tags = tags + ["installed"] if installed else tags
        if len([x for x in tags if x in hide_tags]) > 0:
            continue
        if search_text and search_text.strip():
            if search_text.lower() not in html.escape(name).lower() and search_text.lower() not in html.escape(description).lower() and search_text.lower() not in html.escape(tags_string).lower():
                continue
        version_code = ''
        type_code = ''
        install_code = ''
        enabled_code = ''
        if installed:
            type_code = f"""<div class="type">{"SYSTEM" if ext['is_builtin'] else 'USER'}</div>"""
            version_code = f"""<div class="version" style="background: {"--input-border-color-focus" if update_available else "inherit"}">{ext['version']}</div>"""
            enabled_code = f"""<input class="gr-check-radio gr-checkbox" name="enable_{html.escape(name)}" type="checkbox" {'checked="checked"' if enabled else ''}>"""
            masked_path = html.escape(path.replace('\\', '/'))
            if not ext['is_builtin']:
                install_code = f"""<button onclick="uninstall_extension(this, '{masked_path}')" class="lg secondary gradio-button custom-button extension-button">uninstall</button>"""
            if update_available:
                install_code += f"""<button onclick="update_extension(this, '{masked_path}')" class="lg secondary gradio-button custom-button extension-button">update</button>"""
        else:
            install_code = f"""<button onclick="install_extension(this, '{html.escape(url)}')" class="lg secondary gradio-button custom-button extension-button">install</button>"""
        tags_text = ", ".join([f"<span class='extension-tag'>{x}</span>" for x in tags])
        code += f"""
            <tr>
                <td{' class="extension_status"' if ext['installed'] else ''}>{enabled_code}</td>
                <td><a href="{html.escape(url)}" target="_blank" class="name">{html.escape(name)}</a><br>{tags_text}</td>
                <td>{html.escape(description)}
                    <p class="info"><span class="date">Created {html.escape(created)} | Added {html.escape(added)} | Pushed {html.escape(pushed)} | Updated {html.escape(updated)}</span></p>
                    <p class="info"><span class="date">Stars {html.escape(str(stars))} | Size {html.escape(str(size))} | Commits {html.escape(str(commits))} | Issues {html.escape(str(issues))}</span></p>
                </td>
                <td>{type_code}</td>
                <td>{version_code}</td>
                <td>{install_code}</td>
            </tr>"""
    code += "</tbody></table>"
    return code


def create_ui():
    import modules.ui
    with gr.Blocks(analytics_enabled=False) as ui:
        extensions_disable_all = gr.Radio(label="Disable all extensions", choices=["none", "user", "all"], value=shared.opts.disable_all_extensions, elem_id="extensions_disable_all", visible=False)
        extensions_disabled_list = gr.Text(elem_id="extensions_disabled_list", visible=False).style(container=False)
        extensions_update_list = gr.Text(elem_id="extensions_update_list", visible=False).style(container=False)
        with gr.Tabs(elem_id="tabs_extensions"):
            with gr.TabItem("Manage Extensions", id="manage"):
                with gr.Row(elem_id="extensions_installed_top"):
                    extension_to_install = gr.Text(elem_id="extension_to_install", visible=False)
                    install_extension_button = gr.Button(elem_id="install_extension_button", visible=False)
                    uninstall_extension_button = gr.Button(elem_id="uninstall_extension_button", visible=False)
                    update_extension_button = gr.Button(elem_id="update_extension_button", visible=False)
                    with gr.Column(scale=4):
                        search_text = gr.Text(label="Search")
                        info = gr.HTML('Note: After any operation such as install/uninstall or enable/disable, please restart the server')
                    with gr.Column(scale=1):
                        sort_column = gr.Dropdown(value="default", label="Sort by", choices=list(sort_ordering.keys()), multiselect=False)
                    with gr.Column(scale=1):
                        refresh_extensions_button = gr.Button(value="Refresh extension list", variant="primary")
                        check = gr.Button(value="Update installed extensions", variant="primary")
                        apply = gr.Button(value="Apply changes & restart server", variant="primary")
                update_extension_list()
                extensions_table = gr.HTML(refresh_extensions_list_from_data(search_text.value, sort_column.value))
                check.click(
                    fn=wrap_gradio_gpu_call(check_updates, extra_outputs=[gr.update()]),
                    _js="extensions_check",
                    inputs=[info, extensions_disabled_list, search_text, sort_column],
                    outputs=[extensions_table, info],
                )
                apply.click(
                    fn=apply_and_restart,
                    _js="extensions_apply",
                    inputs=[extensions_disabled_list, extensions_update_list, extensions_disable_all],
                    outputs=[],
                )
                refresh_extensions_button.click(
                    fn=modules.ui.wrap_gradio_call(refresh_extensions_list, extra_outputs=[gr.update(), gr.update()]),
                    inputs=[search_text, sort_column],
                    outputs=[extensions_table, info],
                )
                install_extension_button.click(
                    fn=modules.ui.wrap_gradio_call(install_extension, extra_outputs=[gr.update(), gr.update(), gr.update()]),
                    inputs=[extension_to_install, search_text, sort_column],
                    outputs=[extensions_table, info],
                )
                uninstall_extension_button.click(
                    fn=modules.ui.wrap_gradio_call(uninstall_extension, extra_outputs=[gr.update(), gr.update(), gr.update()]),
                    inputs=[extension_to_install, search_text, sort_column],
                    outputs=[extensions_table, info],
                )
                update_extension_button.click(
                    fn=modules.ui.wrap_gradio_call(update_extension, extra_outputs=[gr.update(), gr.update(), gr.update()]),
                    inputs=[extension_to_install, search_text, sort_column],
                    outputs=[extensions_table, info],
                )
                search_text.change(
                    fn=modules.ui.wrap_gradio_call(search_extensions, extra_outputs=[gr.update(), gr.update()]),
                    inputs=[search_text, sort_column],
                    outputs=[extensions_table, info],
                )
                sort_column.change(
                    fn=modules.ui.wrap_gradio_call(search_extensions, extra_outputs=[gr.update(), gr.update()]),
                    inputs=[search_text, sort_column],
                    outputs=[extensions_table, info],
                )
            with gr.TabItem("Manual install", id="install_from_url"):
                install_url = gr.Text(label="URL for extension's git repository")
                install_branch = gr.Text(label="Specific branch name", placeholder="Leave empty for default main branch")
                install_dirname = gr.Text(label="Local directory name", placeholder="Leave empty for auto")
                install_button = gr.Button(value="Install", variant="primary")
                info = gr.HTML(elem_id="extension_info")
                install_button.click(
                    fn=modules.ui.wrap_gradio_call(install_extension_from_url, extra_outputs=[gr.update()]),
                    inputs=[install_dirname, install_url, install_branch, search_text, sort_column],
                    outputs=[extensions_table, info],
                )
    return ui
