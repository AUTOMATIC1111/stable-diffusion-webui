import os
import sys
import time
import shlex
import logging
import subprocess

commandline_args = os.environ.get('COMMANDLINE_ARGS', "")
sys.argv += shlex.split(commandline_args)

import installer
installer.ensure_base_requirements()
installer.add_args()
installer.parse_args()
installer.setup_logging(False)
installer.extensions_preload(force=False)

import modules.cmd_args
args, _ = modules.cmd_args.parser.parse_known_args()
import modules.paths_internal
script_path = modules.paths_internal.script_path
extensions_dir = modules.paths_internal.extensions_dir
git = os.environ.get('GIT', "git")
index_url = os.environ.get('INDEX_URL', "")
stored_commit_hash = None
dir_repos = "repositories"
python = sys.executable # used by some extensions to run python
skip_install = False # parsed by some extensions


def commit_hash(): # compatbility function
    global stored_commit_hash # pylint: disable=global-statement
    if stored_commit_hash is not None:
        return stored_commit_hash
    try:
        stored_commit_hash = run(f"{git} rev-parse HEAD").strip()
    except Exception:
        stored_commit_hash = "<none>"
    return stored_commit_hash


def run(command, desc=None, errdesc=None, custom_env=None, live=False): # compatbility function
    if desc is not None:
        installer.log.info(desc)
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


def check_run(command): # compatbility function
    result = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    return result.returncode == 0


def is_installed(package): # compatbility function
    return installer.installed(package)


def repo_dir(name): # compatbility function
    return os.path.join(script_path, dir_repos, name)


def run_python(code, desc=None, errdesc=None): # compatbility function
    return run(f'"{sys.executable}" -c "{code}"', desc, errdesc)


def run_pip(pkg, desc=None): # compatbility function
    if desc is None:
        desc = pkg
    index_url_line = f' --index-url {index_url}' if index_url != '' else ''
    return run(f'"{sys.executable}" -m pip {pkg} --prefer-binary{index_url_line}', desc=f"Installing {desc}", errdesc=f"Couldn't install {desc}")


def check_run_python(code): # compatbility function
    return check_run(f'"{sys.executable}" -c "{code}"')


def git_clone(url, tgt, _name, commithash=None): # compatbility function
    installer.clone(url, tgt, commithash)


def run_extension_installer(ext_dir): # compatbility function
    installer.run_extension_installer(ext_dir)


def get_memory_stats():
    import psutil
    def gb(val: float):
        return round(val / 1024 / 1024 / 1024, 2)
    process = psutil.Process(os.getpid())
    res = process.memory_info()
    ram_total = 100 * res.rss / process.memory_percent()
    return f'used: {gb(res.rss)} total: {gb(ram_total)}'


def start_server(immediate=True, server=None):
    if args.profile:
        import cProfile
        pr = cProfile.Profile()
        pr.enable()
    import gc
    import importlib.util
    collected = 0
    if server is not None:
        server = None
        collected = gc.collect()
    if not immediate:
        time.sleep(3)
    installer.log.debug(f'Memory {get_memory_stats()} Collected {collected}')
    module_spec = importlib.util.spec_from_file_location('webui', 'webui.py')
    server = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(server)
    if args.test:
        installer.log.info("Test only")
        server.wants_restart = False
    else:
        if args.api_only:
            server = server.api_only()
        else:
            server = server.webui()
    if args.profile:
        installer.print_profile(pr, 'WebUI')
    return server


if __name__ == "__main__":
    if args.version:
        installer.add_args()
        installer.log.info('SD.Next version information')
        installer.check_python()
        installer.check_version()
        installer.check_torch()
        exit(0)
    installer.run_setup()
    installer.extensions_preload(force=True)
    installer.log.info(f"Server arguments: {sys.argv[1:]}")
    installer.log.debug('Starting WebUI')
    logging.disable(logging.NOTSET if args.debug else logging.DEBUG)

    instance = start_server(immediate=True, server=None)
    while True:
        try:
            alive = instance.thread.is_alive()
        except:
            alive = False
        if round(time.time()) % 120 == 0:
            installer.log.debug(f'Server alive: {alive} Memory {get_memory_stats()}')
        if not alive:
            if instance.wants_restart:
                installer.log.info('Server restarting...')
                instance = start_server(immediate=False, server=instance)
            else:
                installer.log.info('Exiting...')
                break
        time.sleep(1)
