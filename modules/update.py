from types import SimpleNamespace
import gradio as gr
import installer as i


version = SimpleNamespace(**{
    'url': '',
    'branch': '',
    'current': '0000-00-00',
    'chash': '0000000',
    'latest': '0000-00-00',
    'lhash': '0000000',
})


def get_version():
    # try:
    origin = i.git('remote get-url origin')
    origin = origin.splitlines()[0]
    version.branch = i.git('rev-parse --abbrev-ref HEAD')
    version.branch = version.branch.splitlines()[0]
    version.url = origin + '/tree/' + version.branch

    ver = i.git('log --pretty=format:"%h %ad" -1 --date=short')
    ver = ver.splitlines()[0]
    version.chash, version.current = ver.split(' ')

    ver = i.git(f'log origin/{version.branch} --pretty=format:"%h %ad" -1 --date=short')
    ver = ver.splitlines()[0]
    version.lhash, version.latest = ver.split(' ')

    # except Exception as e:
    #    i.log.error(f'Version check failed: {e}')
    i.log.info(f'Version: {vars(version)}')
    html = f'''
        <div>URL: <a href="{version.url}" target="_blank">{version.url}</a></div>
        <div>Current branch: <span style="color: var(--highlight-color)">{version.branch}</span></div>
        <div>Current version: <span style="color: var(--highlight-color)">{version.current}</span> hash <span style="color: var(--highlight-color)">{version.chash}</span></div>
        <div>Latest version: <span style="color: var(--highlight-color)">{version.latest}</span> hash <span style="color: var(--highlight-color)">{version.lhash}</span></div>
    '''
    return html


def apply_update(update_rebase, update_submodules, update_extensions):
    html = [
        'Updating...',
        f'Core rebase: {update_rebase} | Submodules: {update_submodules} | Extensions: {update_extensions}',
        f'<div>Current version: <span style="color: var(--highlight-color)">{version.current}</span> hash <span style="color: var(--highlight-color)">{version.chash}</span></div>',
    ]
    get_version()
    phash = version.chash
    try:
        if update_rebase:
            i.git('add .')
            i.git('stash')
        res = i.update('.', current_branch=True, rebase=update_rebase)
        html.append(res.replace('\n', '<br>'))
    except Exception as e:
        html.append(f'Error during repository upgrade: {e}')
        i.log.error(f'Error during repository upgrade: {e}')
    if update_submodules:
        try:
            res = i.install_submodules(force=True)
            html.append(res.replace('\n', '<br>'))
        except Exception as e:
            html.append(f'Error during submodule upgrade: {e}')
            i.log.error(f'Error during submodule upgrade: {e}')
    if update_extensions:
        try:
            res = i.install_extensions(force=True)
            html.append(res.replace('\n', '<br>'))
        except Exception as e:
            html.append(f'Error during extension upgrade: {e}')
            i.log.error(f'Error during extension upgrade: {e}')
    res = get_version()
    html.append('')
    html.append(res)
    if phash != version.chash:
        html.append('<span style="color: var(--highlight-color)">Update successful!<br>Perform full server restart to apply changes</span>')
    else:
        html.append('<span style="color: var(--highlight-color)">No changes</span>')
    return '<br>'.join(html)

def create_ui():
    with gr.Row():
        update_check = gr.Button(value='Check for updates', elem_id="ui_update_check", variant="primary")
        update_apply = gr.Button(value='Download updates', elem_id="ui_update_apply", variant="primary")
    with gr.Row():
        update_rebase = gr.Checkbox(label='Rebase', elem_id="ui_update_rebase", value=True)
    with gr.Row():
        update_submodules = gr.Checkbox(label='Submodules', elem_id="ui_update_submodules", value=True)
    with gr.Row():
        update_extensions = gr.Checkbox(label='Extensions', elem_id="ui_update_extensions", value=True)
    with gr.Row():
        update_status = gr.HTML("", elem_id="ui_update_status", elem_classes=['update-status'])
    update_check.click(fn=get_version, inputs=[], outputs=[update_status])
    update_apply.click(fn=apply_update, inputs=[update_rebase, update_submodules, update_extensions], outputs=[update_status])
