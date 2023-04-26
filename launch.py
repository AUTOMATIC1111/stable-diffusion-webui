import subprocess
import os
import sys
import shlex
import logging
import setup
import modules.paths_internal
import modules.cmd_args

setup.ensure_base_requirements()
from rich import print # pylint: disable=redefined-builtin,wrong-import-order

### majority of this file is superflous, but used by some extensions as helpers during extension installation

commandline_args = os.environ.get('COMMANDLINE_ARGS', "")
sys.argv += shlex.split(commandline_args)
setup.extensions_preload(force=False)
setup.parse_args()
args, _ = modules.cmd_args.parser.parse_known_args()
script_path = modules.paths_internal.script_path
extensions_dir = modules.paths_internal.extensions_dir
git = os.environ.get('GIT', "git")
index_url = os.environ.get('INDEX_URL', "")
stored_commit_hash = None
dir_repos = "repositories"
python = sys.executable # used by some extensions to run python
skip_install = False # parsed by some extensions


def commit_hash():
    global stored_commit_hash # pylint: disable=global-statement
    if stored_commit_hash is not None:
        return stored_commit_hash
    try:
        stored_commit_hash = run(f"{git} rev-parse HEAD").strip()
    except Exception:
        stored_commit_hash = "<none>"
    return stored_commit_hash


def run(command, desc=None, errdesc=None, custom_env=None, live=False):
    if desc is not None:
        print(desc)
    if live:
        result = subprocess.run(command, check=False, shell=True, env=os.environ if custom_env is None else custom_env)
        if result.returncode != 0:
            raise RuntimeError(f"""{errdesc or 'Error running command'} Command: {command} Error code: {result.returncode}""")
        return ''
    result = subprocess.run(command, stdout=subprocess.PIPE, check=False, stderr=subprocess.PIPE, shell=True, env=os.environ if custom_env is None else custom_env)
    if result.returncode != 0:
        raise RuntimeError(f"""{errdesc or 'Error running command'}: {command} code: {result.returncode}
{result.stdout.decode(encoding="utf8", errors="ignore") if len(result.stdout)>0 else ''}
{result.stderr.decode(encoding="utf8", errors="ignore") if len(result.stderr)>0 else ''}
""")
    return result.stdout.decode(encoding="utf8", errors="ignore")


def check_run(command):
    result = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    return result.returncode == 0


def is_installed(package):
    return setup.installed(package)


def repo_dir(name):
    return os.path.join(script_path, dir_repos, name)


def run_python(code, desc=None, errdesc=None):
    return run(f'"{sys.executable}" -c "{code}"', desc, errdesc)


def run_pip(pkg, desc=None):
    if desc is None:
        desc = pkg
    index_url_line = f' --index-url {index_url}' if index_url != '' else ''
    return run(f'"{sys.executable}" -m pip {pkg} --prefer-binary{index_url_line}', desc=f"Installing {desc}", errdesc=f"Couldn't install {desc}")


def check_run_python(code):
    return check_run(f'"{sys.executable}" -c "{code}"')


def git_clone(url, tgt, _name, commithash=None):
    setup.clone(url, tgt, commithash)


def run_extension_installer(ext_dir):
    setup.run_extension_installer(ext_dir)

if __name__ == "__main__":
    setup.run_setup()
    setup.extensions_preload(force=True)
    setup.log.info(f"Server arguments: {sys.argv[1:]}")
    setup.log.debug('Starting WebUI')
    logging.disable(logging.INFO)
    import webui
    webui.webui()
