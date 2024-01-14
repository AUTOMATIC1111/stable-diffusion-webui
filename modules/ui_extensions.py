import os
import json
import shutil
import errno
import html
from datetime import datetime, timedelta
import git
import gradio as gr
from modules import extensions, shared, paths, errors


extensions_index = "https://vladmandic.github.io/sd-data/pages/extensions.json"
hide_tags = ["localization"]
extensions_list = []
sort_ordering = {
    "default": (True, lambda x: x.get('sort_default', '')),
    "user extensions": (True, lambda x: x.get('sort_user', '')),
    "trending": (True, lambda x: x.get('sort_trending', -1)),
    "update available": (True, lambda x: x.get('sort_update', '')),
    "updated date": (True, lambda x: x.get('updated', '2000-01-01T00:00')),
    "created date": (True, lambda x: x.get('created', '2000-01-01T00:00')),
    "name": (False, lambda x: x.get('name', '').lower()),
    "enabled": (False, lambda x: x.get('sort_enabled', '').lower()),
    "size": (True, lambda x: x.get('size', 0)),
    "stars": (True, lambda x: x.get('stars', 0)),
    "commits": (True, lambda x: x.get('commits', 0)),
    "issues": (True, lambda x: x.get('issues', 0)),
}


def get_installed(ext) -> extensions.Extension:
    installed: extensions.Extension = [e for e in extensions.extensions if (e.remote or '').startswith(ext['url'].replace('.git', ''))]
    return installed[0] if len(installed) > 0 else None


def list_extensions():
    global extensions_list # pylint: disable=global-statement
    fn = os.path.join(paths.script_path, "html", "extensions.json")
    extensions_list = shared.readfile(fn) or []
    if type(extensions_list) != list:
        shared.log.warning(f'Invalid extensions list: file={fn}')
        extensions_list = []
    found = []
    for ext in extensions.extensions:
        ext.read_info()
    for ext in extensions_list:
        installed = get_installed(ext)
        if installed:
            found.append(installed)
    for ext in [e for e in extensions.extensions if e not in found]: # installed but not in index
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


def apply_changes(disable_list, update_list, disable_all):
    check_access()
    shared.log.debug(f'Extensions apply: disable={disable_list} update={update_list}')
    disabled = json.loads(disable_list)
    assert type(disabled) == list, f"wrong disable_list data for apply_changes: {disable_list}"
    update = json.loads(update_list)
    assert type(update) == list, f"wrong update_list data for apply_changes: {update_list}"
    update = set(update)
    for ext in extensions.extensions:
        if ext.name not in update:
            continue
        try:
            ext.git_fetch()
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
                ext.git_fetch()
                ext.read_info()
                commit_date = ext.commit_date or 1577836800
                shared.log.info(f'Extensions updated: {ext.name} {ext.commit_hash[:8]} {datetime.utcfromtimestamp(commit_date)}')
            else:
                commit_date = ext.commit_date or 1577836800
                shared.log.debug(f'Extensions no update available: {ext.name} {ext.commit_hash[:8]} {datetime.utcfromtimestamp(commit_date)}')
        except FileNotFoundError as e:
            if 'FETCH_HEAD' not in str(e):
                raise
        except Exception as e:
            errors.display(e, f'extensions check update: {ext.name}')
        shared.state.nextjob()
    return create_html(search_text, sort_column), "Extension update complete | Restart required"


def make_commit_link(commit_hash, remote, text=None):
    if text is None:
        text = commit_hash[:8]
    if remote.startswith("https://github.com/"):
        if remote.endswith(".git"):
            remote = remote[:-4]
        href = remote + "/commit/" + commit_hash
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
    if url.endswith('.git'):
        url = url.replace('.git', '')
    try:
        shutil.rmtree(tmpdir, True)
        if not branch_name: # if no branch is specified, use the default branch
            with git.Repo.clone_from(url, tmpdir, filter=['blob:none']) as repo:
                repo.remote().fetch()
                for submodule in repo.submodules:
                    submodule.update()
        else:
            with git.Repo.clone_from(url, tmpdir, filter=['blob:none'], branch=branch_name) as repo:
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
        return [create_html(search_text, sort_column), html.escape(f"Extension installed: {target_dir} | Restart required")]
    except Exception as e:
        shared.log.error(f'Error installing extension: {url} {e}')
    finally:
        shutil.rmtree(tmpdir, True)
    return []


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

    found = [extension for extension in extensions.extensions if os.path.abspath(extension.path) == os.path.abspath(extension_path)]
    if len(found) > 0 and os.path.isdir(extension_path):
        found = found[0]
        try:
            shutil.rmtree(found.path, ignore_errors=False, onerror=errorRemoveReadonly)
            # extensions.extensions = [extension for extension in extensions.extensions if os.path.abspath(found.path) != os.path.abspath(extension_path)]
        except Exception as e:
            shared.log.warning(f'Extension uninstall failed: {found.path} {e}')
        list_extensions()
        global extensions_list # pylint: disable=global-statement
        extensions_list = [ext for ext in extensions_list if ext['name'] != found.name]
        shared.log.info(f'Extension uninstalled: {found.path}')
        code = create_html(search_text, sort_column)
        return code, f"Extension uninstalled: {found.path} | Restart required"
    else:
        shared.log.warning(f'Extension uninstall cannot find extension: {extension_path}')
        code = create_html(search_text, sort_column)
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
                ext.git_fetch()
                ext.read_info()
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
    return create_html(search_text, sort_column), f"Extension updated | {extension_path} | Restart required"


def refresh_extensions_list(search_text, sort_column):
    global extensions_list # pylint: disable=global-statement
    import urllib.request
    try:
        shared.log.debug(f'Updating extensions list: url={extensions_index}')
        with urllib.request.urlopen(extensions_index, timeout=3.0) as response:
            text = response.read()
        extensions_list = json.loads(text)
        with open(os.path.join(paths.script_path, "html", "extensions.json"), "w", encoding="utf-8") as outfile:
            json_object = json.dumps(extensions_list, indent=2)
            outfile.write(json_object)
            shared.log.info(f'Updated extensions list: items={len(extensions_list)} url={extensions_index}')
    except Exception as e:
        shared.log.warning(f'Updated extensions list failed: {extensions_index} {e}')
    list_extensions()
    code = create_html(search_text, sort_column)
    return code, f'Extensions | {len(extensions.extensions)} registered | {len(extensions_list)} available'


def search_extensions(search_text, sort_column):
    code = create_html(search_text, sort_column)
    return code, f'Search | {search_text} | {sort_column}'


def create_html(search_text, sort_column):
    # shared.log.debug(f'Extensions manager: refresh list search="{search_text}" sort="{sort_column}"')
    code = """
        <table id="extensions">
            <colgroup>
                <col style="width: 1%; background: var(--table-border-color)">
                <col style="width: 1%; background: var(--table-border-color)">
                <col style="width: 20%; background: var(--table-border-color)">
                <col style="width: 59%;">
                <col style="width: 5%; background: var(--panel-background-fill)">
                <col style="width: 10%; background: var(--panel-background-fill)">
                <col style="width: 5%; background: var(--table-border-color)">
            </colgroup>
            <thead style="font-size: 110%; border-style: solid; border-bottom: 1px var(--button-primary-border-color) solid">
            <tr>
                <th>Status</th>
                <th>Enabled</th>
                <th>Extension</th>
                <th>Description</th>
                <th>Type</th>
                <th>Current version</th>
                <th></th>
            </tr>
            </thead>
        <tbody>"""
    if len(extensions_list) == 0:
        list_extensions()
    for ext in extensions_list:
        installed = get_installed(ext)
        ext['installed'] = installed is not None
        ext['commit_date'] = installed.commit_date if installed is not None else 1577836800
        ext['is_builtin'] = installed.is_builtin if installed is not None else False
        ext['version'] = installed.version if installed is not None else ''
        ext['enabled'] = installed.enabled if installed is not None else ''
        ext['remote'] = installed.remote if installed is not None else None
        ext['path'] = installed.path if installed is not None else ''
        ext['sort_default'] = f"{'1' if ext['is_builtin'] else '0'}{'1' if ext['installed'] else '0'}{ext.get('updated', '2000-01-01T00:00')}"
    sort_reverse, sort_function = sort_ordering[sort_column]

    def dt(x: str):
        val = ext.get(x, None)
        try:
            return datetime.fromisoformat(val[:-1]).strftime('%a %b%d %Y %H:%M') if val is not None else "N/A"
        except Exception:
            return 'N/A'

    stats = { 'processed': 0, 'enabled': 0, 'hidden': 0, 'installed': 0 }
    for ext in sorted(extensions_list, key=sort_function, reverse=sort_reverse):
        installed = get_installed(ext)
        author = ''
        updated = datetime.timestamp(datetime.now())
        try:
            if 'github' in ext['url']:
                author = ext['url'].split('/')[-2].split(':')[-1] if '/' in ext['url'] else ext['url'].split(':')[1].split('/')[0]
                author = f"Author: {author}"
                updated = datetime.timestamp(datetime.fromisoformat(ext.get('updated', '2000-01-01T00:00:00.000Z').rstrip('Z')))
        except Exception:
            updated = datetime.timestamp(datetime.now())
        update_available = (installed is not None) and (ext['remote'] is not None) and (ext['commit_date'] + 60 * 60 < updated)
        ext['sort_user'] = f"{'0' if ext['is_builtin'] else '1'}{'1' if ext['installed'] else '0'}{ext.get('name', '')}"
        ext['sort_enabled'] = f"{'0' if ext['enabled'] else '1'}{'1' if ext['is_builtin'] else '0'}{'1' if ext['installed'] else '0'}{ext.get('updated', '2000-01-01T00:00')}"
        ext['sort_update'] = f"{'1' if update_available else '0'}{'1' if ext['installed'] else '0'}{ext.get('updated', '2000-01-01T00:00')}"
        delta = datetime.now() - datetime.fromisoformat(ext.get('created', '2000-01-01T00:00Z')[:-1])
        ext['sort_trending'] = round(ext.get('stars', 0) / max(delta.days, 5), 1)
        tags = ext.get("tags", [])
        if not isinstance(tags, list):
            tags = tags.split(' ')
        tags_string = ' '.join(tags)
        tags = tags + ["installed"] if installed else tags
        tags = [t for t in tags if t.strip() != '']
        if len([x for x in tags if x in hide_tags]) > 0:
            continue
        visible = 'table-row'
        if search_text:
            s = search_text.strip().lower()
            if s not in html.escape(ext.get("name", "unknown")).lower() and s not in html.escape(ext.get("description", "")).lower() and s not in html.escape(tags_string).lower() and s not in author.lower():
                stats['hidden'] += 1
                visible = 'none'
        stats['processed'] += 1
        version_code = ''
        type_code = ''
        install_code = ''
        enabled_code = ''
        if installed:
            stats['installed'] += 1
            if ext.get("enabled", False):
                stats['enabled'] += 1
            type_code = f"""<div class="type">{"SYSTEM" if ext['is_builtin'] else 'USER'}</div>"""
            version_code = f"""<div class="version" style="background: {"--input-border-color-focus" if update_available else "inherit"}">{ext['version']}</div>"""
            enabled_code = f"""<input class="gr-check-radio gr-checkbox" name="enable_{html.escape(ext.get("name", "unknown"))}" type="checkbox" {'checked="checked"' if ext.get("enabled", False) else ''}>"""
            masked_path = html.escape(ext.get("path", "").replace('\\', '/'))
            if not ext['is_builtin']:
                install_code = f"""<button onclick="uninstall_extension(this, '{masked_path}')" class="lg secondary gradio-button custom-button extension-button">uninstall</button>"""
            if update_available:
                install_code += f"""<button onclick="update_extension(this, '{masked_path}')" class="lg secondary gradio-button custom-button extension-button">update</button>"""
        else:
            install_code = f"""<button onclick="install_extension(this, '{html.escape(ext.get('url', ''))}')" class="lg secondary gradio-button custom-button extension-button">install</button>"""
        tags_text = ", ".join([f"<span class='extension-tag'>{x}</span>" for x in tags])
        if ext.get('status', None) is None or type(ext['status']) == str: # old format
            ext['status'] = 0
        if ext['url'] is None or ext['url'] == '':
            status = "<span style='cursor:pointer;color:#00C0FD' title='Local'>⬤</span>"
        elif ext['status'] > 0:
            if ext['status'] == 1:
                status = "<span style='cursor:pointer;color:#00FD9C ' title='Verified'>⬤</span>"
            elif ext['status'] == 2:
                status = "<span style='cursor:pointer;color:#FFC300' title='Supported only with backend:Original'>⬤</span>"
            elif ext['status'] == 3:
                status = "<span style='cursor:pointer;color:#FFC300' title='Supported only with backend:Diffusers'>⬤</span>"
            elif ext['status'] == 4:
                status = f"<span style='cursor:pointer;color:#4E22FF' title=\"{ext.get('note', 'custom value')}\">⬤</span>"
            elif ext['status'] == 5:
                status = "<span style='cursor:pointer;color:#CE0000' title='Not supported'>⬤</span>"
            elif ext['status'] == 6:
                status = "<span style='cursor:pointer;color:#AEAEAE' title='Just discovered'>⬤</span>"
            else:
                status = "<span style='cursor:pointer;color:#008EBC' title='Unknown status'>⬤</span>"
        else:
            if updated < datetime.timestamp(datetime.now() - timedelta(6*30)):
                status = "<span style='cursor:pointer;color:#C000CF' title='Unmaintained'>⬤</span>"
            else:
                status = "<span style='cursor:pointer;color:#7C7C7C' title='No info'>⬤</span>"

        code += f"""
            <tr style="display: {visible}">
                <td>{status}</td>
                <td{' class="extension_status"' if ext['installed'] else ''}>{enabled_code}</td>
                <td><a href="{html.escape(ext.get('url', ''))}" title={html.escape(ext.get('note', ''))} target="_blank" class="name">{html.escape(ext.get("name", "unknown"))}</a><br>{tags_text}</td>
                <td>{html.escape(ext.get("description", ""))}
                    <p class="info"><span class="date">Created {html.escape(dt('created'))} | Added {html.escape(dt('added'))} | Pushed {html.escape(dt('pushed'))} | Updated {html.escape(dt('updated'))}</span></p>
                    <p class="info"><span class="date">{author} | Stars {html.escape(str(ext.get('stars', 0)))} | Size {html.escape(str(ext.get('size', 0)))} | Commits {html.escape(str(ext.get('commits', 0)))} | Issues {html.escape(str(ext.get('issues', 0)))} | Trending {html.escape(str(ext['sort_trending']))}</span></p>
                </td>
                <td>{type_code}</td>
                <td>{version_code}</td>
                <td>{install_code}</td>
            </tr>"""
    code += "</tbody></table>"
    shared.log.debug(f'Extension list: processed={stats["processed"]} installed={stats["installed"]} enabled={stats["enabled"]} disabled={stats["installed"] - stats["enabled"]} visible={stats["processed"] - stats["hidden"]} hidden={stats["hidden"]}')
    return code


def create_ui():
    import modules.ui
    with gr.Blocks(analytics_enabled=False) as ui:
        extensions_disable_all = gr.Radio(label="Disable all extensions", choices=["none", "user", "all"], value=shared.opts.disable_all_extensions, elem_id="extensions_disable_all", visible=False)
        extensions_disabled_list = gr.Text(elem_id="extensions_disabled_list", visible=False, container=False)
        extensions_update_list = gr.Text(elem_id="extensions_update_list", visible=False, container=False)
        with gr.Tabs(elem_id="tabs_extensions"):
            with gr.TabItem("Manage extensions", id="manage"):
                with gr.Row(elem_id="extensions_installed_top"):
                    extension_to_install = gr.Text(elem_id="extension_to_install", visible=False)
                    install_extension_button = gr.Button(elem_id="install_extension_button", visible=False)
                    uninstall_extension_button = gr.Button(elem_id="uninstall_extension_button", visible=False)
                    update_extension_button = gr.Button(elem_id="update_extension_button", visible=False)
                    with gr.Column(scale=4):
                        search_text = gr.Text(label="Search")
                    with gr.Column(scale=1):
                        sort_column = gr.Dropdown(value="default", label="Sort by", choices=list(sort_ordering.keys()), multiselect=False)
                    with gr.Column(scale=1):
                        refresh_extensions_button = gr.Button(value="Refresh extension list", variant="primary")
                        check = gr.Button(value="Update all installed", variant="primary")
                        apply = gr.Button(value="Apply changes", variant="primary")
                list_extensions()
                gr.HTML('<span style="color: var(--body-text-color)"><h2>Extension list</h2>⯀ Refesh extension list to download latest list with status<br>⯀ Check status of an extension by looking at status icon before installing it<br>⯀ After any operation such as install/uninstall or enable/disable, please restart the server<br></span>')
                gr.HTML('')
                info = gr.HTML('')
                extensions_table = gr.HTML(create_html(search_text.value, sort_column.value))
                check.click(
                    fn=modules.ui.wrap_gradio_call(check_updates, extra_outputs=[gr.update()]),
                    _js="extensions_check",
                    inputs=[info, extensions_disabled_list, search_text, sort_column],
                    outputs=[extensions_table, info],
                )
                apply.click(
                    fn=apply_changes,
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
                install_url = gr.Text(label="Extension GIT repository URL")
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
