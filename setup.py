import os
import sys
import json
import time
import shutil
import logging
import platform
import subprocess

try:
    from modules.cmd_args import parser
except:
    import argparse
    parser = argparse.ArgumentParser(description="Stable Diffusion", conflict_handler='resolve', formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=55, indent_increment=2, width=200))


class Dot(dict): # dot notation access to dictionary attributes
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


log = logging.getLogger("sd")
args = Dot({ 'debug': False, 'upgrade': False, 'noupdate': False, 'nodirectml': False, 'skip-extensions': False, 'skip-requirements': False, 'reset': False })
quick_allowed = True
errors = 0
opts = {}


# setup console and file logging
def setup_logging(clean=False):
    try:
        if clean and os.path.isfile('setup.log'):
            os.remove('setup.log')
        time.sleep(0.1) # prevent race condition
    except:
        pass
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s | %(levelname)s | %(pathname)s | %(message)s', filename='setup.log', filemode='a', encoding='utf-8', force=True)
    from rich.theme import Theme
    from rich.logging import RichHandler
    from rich.console import Console
    from rich.pretty import install as pretty_install
    from rich.traceback import install as traceback_install
    console = Console(log_time=True, log_time_format='%H:%M:%S-%f', theme=Theme({
        "traceback.border": "black",
        "traceback.border.syntax_error": "black",
        "inspect.value.border": "black",
    }))
    pretty_install(console=console)
    traceback_install(console=console, extra_lines=1, width=console.width, word_wrap=False, indent_guides=False, suppress=[])
    rh = RichHandler(show_time=True, omit_repeated_times=False, show_level=True, show_path=False, markup=False, rich_tracebacks=True, log_time_format='%H:%M:%S-%f', level=logging.DEBUG if args.debug else logging.INFO, console=console)
    rh.set_name(logging.DEBUG if args.debug else logging.INFO)
    log.addHandler(rh)


# check if package is installed
def installed(package, friendly: str = None):
    import pkg_resources
    ok = True
    try:
        if friendly:
            pkgs = friendly.split()
        else:
            pkgs = [p for p in package.split() if not p.startswith('-') and not p.startswith('=')]
            pkgs = [p.split('/')[-1] for p in pkgs] # get only package name if installing from url
        for pkg in pkgs:
            if '>=' in pkg:
                p = pkg.split('>=')
            else:
                p = pkg.split('==')
            spec = pkg_resources.working_set.by_key.get(p[0], None) # more reliable than importlib
            if spec is None:
                spec = pkg_resources.working_set.by_key.get(p[0].lower(), None) # check name variations
            if spec is None:
                spec = pkg_resources.working_set.by_key.get(p[0].replace('_', '-'), None) # check name variations
            ok = ok and spec is not None
            if ok:
                version = pkg_resources.get_distribution(p[0]).version
                log.debug(f"Package version found: {p[0]} {version}")
                if len(p) > 1:
                    ok = ok and version == p[1]
                    if not ok:
                        log.warning(f"Package wrong version: {p[0]} {version} required {p[1]}")
            else:
                log.debug(f"Package version not found: {p[0]}")
        return ok
    except ModuleNotFoundError:
        log.debug(f"Package not installed: {pkgs}")
        return False


# install package using pip if not already installed
def install(package, friendly: str = None, ignore: bool = False):
    def pip(arg: str):
        arg = arg.replace('>=', '==')
        log.info(f'Installing package: {arg.replace("install", "").replace("--upgrade", "").replace("--no-deps", "").replace("  ", " ").strip()}')
        log.debug(f"Running pip: {arg}")
        result = subprocess.run(f'"{sys.executable}" -m pip {arg}', shell=True, check=False, env=os.environ, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        txt = result.stdout.decode(encoding="utf8", errors="ignore")
        if len(result.stderr) > 0:
            txt += ('\n' if len(txt) > 0 else '') + result.stderr.decode(encoding="utf8", errors="ignore")
        txt = txt.strip()
        if result.returncode != 0 and not ignore:
            global errors # pylint: disable=global-statement
            errors += 1
            log.error(f'Error running pip: {arg}')
            log.debug(f'Pip output: {txt}')
        return txt

    if not installed(package, friendly):
        pip(f"install --upgrade {package}")


# execute git command
def git(arg: str, folder: str = None, ignore: bool = False):
    if args.skip_git:
        return ''
    git_cmd = os.environ.get('GIT', "git")
    result = subprocess.run(f'"{git_cmd}" {arg}', check=False, shell=True, env=os.environ, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=folder or '.')
    txt = result.stdout.decode(encoding="utf8", errors="ignore")
    if len(result.stderr) > 0:
        txt += ('\n' if len(txt) > 0 else '') + result.stderr.decode(encoding="utf8", errors="ignore")
    txt = txt.strip()
    if result.returncode != 0 and not ignore:
        global errors # pylint: disable=global-statement
        errors += 1
        log.error(f'Error running git: {folder} / {arg}')
        if 'or stash them' in txt:
            log.error('Local changes detected: check setup.log for details')
        log.debug(f'Git output: {txt}')
    return txt


# update switch to main branch as head can get detached and update repository
def update(folder):
    if not os.path.exists(os.path.join(folder, '.git')):
        return
    branch = git('branch', folder)
    if 'main' in branch:
        branch = 'main'
    elif 'master' in branch:
        branch = 'master'
    else:
        branch = branch.split('\n')[0].replace('*', '').strip()
    log.debug(f'Setting branch: {folder} / {branch}')
    git(f'checkout {branch}', folder)
    if branch is None:
        git('pull --autostash --rebase', folder)
    else:
        git(f'pull origin {branch} --autostash --rebase', folder)
    # branch = git('branch', folder)


# clone git repository
def clone(url, folder, commithash=None):
    if os.path.exists(folder):
        if commithash is None:
            return
        current_hash = git('rev-parse HEAD', folder).strip()
        if current_hash != commithash:
            git('fetch', folder)
            git(f'checkout {commithash}', folder)
            return
    else:
        git(f'clone "{url}" "{folder}"')
        if commithash is not None:
            git(f'-C "{folder}" checkout {commithash}')


# check python version
def check_python():
    import platform
    supported_minors = [9, 10]
    if args.experimental:
        supported_minors.append(11)
    log.info(f'Python {platform.python_version()} on {platform.system()}')
    if not (int(sys.version_info.major) == 3 and int(sys.version_info.minor) in supported_minors):
        log.error(f"Incompatible Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} required 3.{supported_minors}")
        exit(1)
    git_cmd = os.environ.get('GIT', "git")
    if shutil.which(git_cmd) is None:
        log.error('Git not found')
        exit(1)
    else:
        git_version = git('--version', folder=None, ignore=False)
        log.debug(f'Git {git_version.replace("git version", "").strip()}')


# check torch version
def check_torch():
    if shutil.which('nvidia-smi') is not None or os.path.exists(os.path.join(os.environ.get('SystemRoot') or r'C:\Windows', 'System32', 'nvidia-smi.exe')):
        log.info('nVidia toolkit detected')
        torch_command = os.environ.get('TORCH_COMMAND', 'torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu118')
        xformers_package = os.environ.get('XFORMERS_PACKAGE', 'xformers==0.0.17' if opts.get('cross_attention_optimization', '') == 'xFormers' else 'none')
    elif shutil.which('rocminfo') is not None or os.path.exists('/opt/rocm/bin/rocminfo'):
        log.info('AMD toolkit detected')
        os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '10.3.0')
        torch_command = os.environ.get('TORCH_COMMAND', 'torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2')
        xformers_package = os.environ.get('XFORMERS_PACKAGE', 'none')
    else:
        machine = platform.machine()
        if 'arm' not in machine and 'aarch' not in machine and not args.nodirectml: # torch-directml is available on AMD64
            log.info('Using DirectML Backend')
            torch_command = os.environ.get('TORCH_COMMAND', 'torch==2.0.0 torchvision torch-directml')
            xformers_package = os.environ.get('XFORMERS_PACKAGE', 'none')
        else:
            log.info('Using CPU-only Torch')
            torch_command = os.environ.get('TORCH_COMMAND', 'torch torchaudio torchvision')
            xformers_package = os.environ.get('XFORMERS_PACKAGE', 'none')
    if 'torch' in torch_command:
        install(torch_command, 'torch torchvision torchaudio')
    try:
        import torch
        log.info(f'Torch {torch.__version__}')
        if torch.cuda.is_available():
            if torch.version.cuda:
                log.info(f'Torch backend: nVidia CUDA {torch.version.cuda} cuDNN {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "N/A"}')
            elif torch.version.hip:
                log.info(f'Torch backend: AMD ROCm HIP {torch.version.hip}')
            else:
                log.warning('Unknown Torch backend')
            for device in [torch.cuda.device(i) for i in range(torch.cuda.device_count())]:
                log.info(f'Torch detected GPU: {torch.cuda.get_device_name(device)} VRAM {round(torch.cuda.get_device_properties(device).total_memory / 1024 / 1024)} Arch {torch.cuda.get_device_capability(device)} Cores {torch.cuda.get_device_properties(device).multi_processor_count}')
        else:
            try:
                import torch_directml
                import pkg_resources
                version = pkg_resources.get_distribution("torch-directml")
                log.info(f'Torch backend: DirectML ({version})')
                for i in range(0, torch_directml.device_count()):
                    log.info(f'Torch detected GPU: {torch_directml.device_name(i)}')
                log.info(f'DirectML default device: {torch_directml.device_name(torch_directml.default_device())}')
            except:
                log.warning("Torch repoorts CUDA not available")
    except Exception as e:
        log.error(f'Could not load torch: {e}')
        exit(1)
    try:
        if 'xformers' in xformers_package:
            install(f'--no-deps {xformers_package}', ignore=True)
    except Exception as e:
        log.debug(f'Cannot install xformers package: {e}')
    try:
        tensorflow_package = os.environ.get('TENSORFLOW_PACKAGE', 'tensorflow==2.12.0')
        install(tensorflow_package, 'tensorflow', ignore=True)
    except Exception as e:
        log.debug(f'Cannot install tensorflow package: {e}')


# install required packages
def install_packages():
    log.info('Installing packages')
    # gfpgan_package = os.environ.get('GFPGAN_PACKAGE', "git+https://github.com/TencentARC/GFPGAN.git@8d2447a2d918f8eba5a4a01463fd48e45126a379")
    # openclip_package = os.environ.get('OPENCLIP_PACKAGE', "git+https://github.com/mlfoundations/open_clip.git@bb6e834e9c70d9c27d0dc3ecedeebeaeb1ffad6b")
    # install(gfpgan_package, 'gfpgan')
    # install(openclip_package, 'open-clip-torch')
    clip_package = os.environ.get('CLIP_PACKAGE', "git+https://github.com/openai/CLIP.git@d50d76daa670286dd6cacf3bcd80b5e4823fc8e1")
    install(clip_package, 'clip')


# clone required repositories
def install_repositories():
    def d(name):
        return os.path.join(os.path.dirname(__file__), 'repositories', name)
    log.info('Installing repositories')
    os.makedirs(os.path.join(os.path.dirname(__file__), 'repositories'), exist_ok=True)
    stable_diffusion_repo = os.environ.get('STABLE_DIFFUSION_REPO', "https://github.com/Stability-AI/stablediffusion.git")
    stable_diffusion_commit = os.environ.get('STABLE_DIFFUSION_COMMIT_HASH', "cf1d67a6fd5ea1aa600c4df58e5b47da45f6bdbf")
    clone(stable_diffusion_repo, d('stable-diffusion-stability-ai'), stable_diffusion_commit)
    taming_transformers_repo = os.environ.get('TAMING_TRANSFORMERS_REPO', "https://github.com/CompVis/taming-transformers.git")
    taming_transformers_commit = os.environ.get('TAMING_TRANSFORMERS_COMMIT_HASH', "3ba01b241669f5ade541ce990f7650a3b8f65318")
    clone(taming_transformers_repo, d('taming-transformers'), taming_transformers_commit)
    k_diffusion_repo = os.environ.get('K_DIFFUSION_REPO', 'https://github.com/crowsonkb/k-diffusion.git')
    k_diffusion_commit = os.environ.get('K_DIFFUSION_COMMIT_HASH', "b43db16749d51055f813255eea2fdf1def801919")
    clone(k_diffusion_repo, d('k-diffusion'), k_diffusion_commit)
    codeformer_repo = os.environ.get('CODEFORMER_REPO', 'https://github.com/sczhou/CodeFormer.git')
    codeformer_commit = os.environ.get('CODEFORMER_COMMIT_HASH', "c5b4593074ba6214284d6acd5f1719b6c5d739af")
    clone(codeformer_repo, d('CodeFormer'), codeformer_commit)
    blip_repo = os.environ.get('BLIP_REPO', 'https://github.com/salesforce/BLIP.git')
    blip_commit = os.environ.get('BLIP_COMMIT_HASH', "48211a1594f1321b00f14c9f7a5b4813144b2fb9")
    clone(blip_repo, d('BLIP'), blip_commit)


# run extension installer
def run_extension_installer(folder):
    path_installer = os.path.join(folder, "install.py")
    if not os.path.isfile(path_installer):
        return
    try:
        log.debug(f"Running extension installer: {folder} / {path_installer}")
        env = os.environ.copy()
        env['PYTHONPATH'] = os.path.abspath(".")
        result = subprocess.run(f'"{sys.executable}" "{path_installer}"', shell=True, env=env, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=folder)
        if result.returncode != 0:
            global errors # pylint: disable=global-statement
            errors += 1
            txt = result.stdout.decode(encoding="utf8", errors="ignore")
            if len(result.stderr) > 0:
                txt = txt + '\n' + result.stderr.decode(encoding="utf8", errors="ignore")
            log.error(f'Error running extension installer: {path_installer}')
            log.debug(txt)
    except Exception as e:
        log.error(f'Exception running extension installer: {e}')

# get list of all enabled extensions
def list_extensions(folder):
    if opts.get('disable_all_extensions', 'none') != 'none':
        log.debug('Disabled extensions: all')
        return []
    disabled_extensions = set(opts.get('disabled_extensions', []))
    if len(disabled_extensions) > 0:
        log.debug(f'Disabled extensions: {disabled_extensions}')
    return [x for x in os.listdir(folder) if x not in disabled_extensions and not x.startswith('.')]


# run installer for each installed and enabled extension and optionally update them
def install_extensions():
    from modules.paths_internal import extensions_builtin_dir, extensions_dir
    for folder in [extensions_builtin_dir, extensions_dir]:
        if not os.path.isdir(folder):
            continue
        extensions = list_extensions(folder)
        log.info(f'Extensions enabled: {extensions}')
        for ext in extensions:
            if not args.noupdate:
                try:
                    update(os.path.join(folder, ext))
                except:
                    log.error(f'Error updating extension: {os.path.join(folder, ext)}')
            if not args.skip_extensions:
                run_extension_installer(os.path.join(folder, ext))


# initialize and optionally update submodules
def install_submodules():
    log.info('Installing submodules')
    txt = git('submodule')
    log.debug(f'Submodules list: {txt}')
    if 'no submodule mapping found' in txt:
        log.warning('Attempting repository recover')
        git('add .')
        git('stash')
        git('merge --abort', folder=None, ignore=True)
        git('fetch --all')
        git('reset --hard origin/master')
        git('checkout master')
        log.info('Continuing setup')
    txt = git('submodule --quiet update --init --recursive')
    if not args.noupdate:
        log.info('Updating submodules')
        submodules = git('submodule').splitlines()
        for submodule in submodules:
            try:
                name = submodule.split()[1].strip()
                update(name)
            except:
                log.error(f'Error updating submodule: {submodule}')


def ensure_package(pkg):
    try:
        import pkg # type: ignore
    except ImportError:
        install(pkg)


def ensure_base_requirements():
    ensure_package('rich')


def install_requirements():
    if args.skip_requirements:
        return
    log.info('Verifying requirements')
    with open('requirements.txt', 'r', encoding='utf8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip() != '' and not line.startswith('#') and line is not None]
        for line in lines:
            install(line)


# set environment variables controling the behavior of various libraries
def set_environment():
    log.info('Setting environment tuning')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    os.environ.setdefault('ACCELERATE', 'True')
    os.environ.setdefault('FORCE_CUDA', '1')
    os.environ.setdefault('ATTN_PRECISION', 'fp16')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'garbage_collection_threshold:0.9,max_split_size_mb:512')
    os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '0')
    os.environ.setdefault('CUDA_CACHE_DISABLE', '0')
    os.environ.setdefault('CUDA_AUTO_BOOST', '1')
    os.environ.setdefault('CUDA_MODULE_LOADING', 'LAZY')
    os.environ.setdefault('CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT', '0')
    os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
    os.environ.setdefault('SAFETENSORS_FAST_GPU', '1')
    os.environ.setdefault('NUMEXPR_MAX_THREADS', '16')
    if sys.platform == 'darwin':
        os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')


def check_extensions():
    newest_all = os.path.getmtime('requirements.txt')
    from modules.paths_internal import extensions_builtin_dir, extensions_dir
    for folder in [extensions_builtin_dir, extensions_dir]:
        if not os.path.isdir(folder):
            continue
        extensions = list_extensions(folder)
        for ext in extensions:
            newest = 0
            extension_dir = os.path.join(folder, ext)
            if not os.path.isdir(extension_dir):
                log.debug(f'Extension listed as installed but folder missing: {extension_dir}')
                continue
            for f in os.listdir(extension_dir):
                if '.json' in f or '.csv' in f or '__pycache__' in f:
                    continue
                ts = os.path.getmtime(os.path.join(extension_dir, f))
                newest = max(newest, ts)
            newest_all = max(newest_all, newest)
            log.debug(f'Extension version: {time.ctime(newest)} {folder}{os.pathsep}{ext}')
    return round(newest_all)


# check version of the main repo and optionally upgrade it
def check_version():
    if not os.path.exists('.git'):
        log.error('Not a git repository')
        exit(1)
    _status = git('status')
    # if 'branch' not in status:
    #    log.error('Cannot get git repository status')
    #    exit(1)
    ver = git('log -1 --pretty=format:"%h %ad"')
    log.info(f'Version: {ver}')
    commit = git('rev-parse HEAD')
    try:
        import requests
    except ImportError:
        return
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    commits = None
    try:
        commits = requests.get('https://api.github.com/repos/vladmandic/automatic/branches/master', timeout=10).json()
        if commits['commit']['sha'] != commit:
            if args.upgrade:
                global quick_allowed # pylint: disable=global-statement
                quick_allowed = False
                try:
                    git('add .')
                    git('stash')
                    update('.')
                    # git('git stash pop')
                    ver = git('log -1 --pretty=format:"%h %ad"')
                    log.info(f'Upgraded to version: {ver}')
                except:
                    log.error('Error upgrading repository')
            else:
                log.info(f'Latest published version: {commits["commit"]["sha"]} {commits["commit"]["commit"]["author"]["date"]}')
    except Exception as e:
        log.error(f'Failed to check version: {e} {commits}')


def update_wiki():
    if not args.noupdate:
        log.info('Updating Wiki')
        try:
            update(os.path.join(os.path.dirname(__file__), "wiki"))
            update(os.path.join(os.path.dirname(__file__), "wiki", "origin-wiki"))
        except:
            log.error('Error updating wiki')


# check if we can run setup in quick mode
def check_timestamp():
    if not quick_allowed or not os.path.isfile('setup.log'):
        return False
    if args.skip_git:
        return True
    ok = True
    setup_time = -1
    with open('setup.log', 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            if 'Setup complete without errors' in line:
                setup_time = int(line.split(' ')[-1])
    try:
        version_time = int(git('log -1 --pretty=format:"%at"'))
    except Exception as e:
        log.error(f'Error getting local repository version: {e}')
        exit(1)
    log.debug(f'Repository update time: {time.ctime(int(version_time))}')
    if setup_time == -1:
        return False
    log.debug(f'Previous setup time: {time.ctime(setup_time)}')
    if setup_time < version_time:
        ok = False
    extension_time = check_extensions()
    log.debug(f'Latest extensions time: {time.ctime(extension_time)}')
    if setup_time < extension_time:
        ok = False
    log.debug(f'Timestamps: version:{version_time} setup:{setup_time} extension:{extension_time}')
    return ok


def add_args():
    parser.add_argument('--debug', default = False, action='store_true', help = "Run installer with debug logging, default: %(default)s")
    parser.add_argument('--reset', default = False, action='store_true', help = "Reset main repository to latest version, default: %(default)s")
    parser.add_argument('--upgrade', default = False, action='store_true', help = "Upgrade main repository to latest version, default: %(default)s")
    parser.add_argument('--noupdate', default = False, action='store_true', help = "Skip update of extensions and submodules, default: %(default)s")
    parser.add_argument('--nodirectml', default = False, action='store_true', help = "Although nVidia and AMD toolkit aren't detected, use CPU not DirectML, default: %(default)s")
    parser.add_argument('--skip-requirements', default = False, action='store_true', help = "Skips checking and installing requirements, default: %(default)s")
    parser.add_argument('--skip-extensions', default = False, action='store_true', help = "Skips running individual extension installers, default: %(default)s")
    parser.add_argument('--skip-git', default = False, action='store_true', help = "Skips running all GIT operations, default: %(default)s")
    parser.add_argument('--experimental', default = False, action='store_true', help = "Allow unsupported versions of libraries, default: %(default)s")
    parser.add_argument('--test', default = False, action='store_true', help = "Run test only, default: %(default)s")


def parse_args():
    # command line args
    global args # pylint: disable=global-statement
    args = parser.parse_args()


def extensions_preload(force = False):
    setup_time = 0
    if os.path.isfile('setup.log'):
        with open('setup.log', 'r', encoding='utf8') as f:
            lines = f.readlines()
            for line in lines:
                if 'Setup complete without errors' in line:
                    setup_time = int(line.split(' ')[-1])
    if setup_time > 0 or force:
        log.info('Running extension preloading')
        from modules.script_loading import preload_extensions
        from modules.paths_internal import extensions_builtin_dir, extensions_dir
        for ext_dir in [extensions_builtin_dir, extensions_dir]:
            preload_extensions(ext_dir, parser)


def git_reset():
    log.warning('Running GIT reset')
    global quick_allowed # pylint: disable=global-statement
    quick_allowed = False
    git('merge --abort')
    git('fetch --all')
    git('reset --hard origin/master')
    git('checkout master')
    log.info('GIT reset complete')


def read_options():
    global opts # pylint: disable=global-statement
    if os.path.isfile(args.ui_settings_file):
        with open(args.ui_settings_file, "r", encoding="utf8") as file:
            opts = json.load(file)


# entry method when used as module
def run_setup():
    setup_logging(args.upgrade)
    read_options()
    check_python()
    if args.reset:
        git_reset()
    if args.skip_git:
        log.info('Skipping GIT operations')
    check_version()
    set_environment()
    check_torch()
    install_requirements()
    if check_timestamp():
        log.info('No changes detected: Quick launch active')
        return
    log.info("Running setup")
    log.debug(f"Args: {vars(args)}")
    install_packages()
    install_repositories()
    install_submodules()
    install_extensions()
    update_wiki()
    if errors == 0:
        log.debug(f'Setup complete without errors: {round(time.time())}')
    else:
        log.warning(f'Setup complete with errors ({errors})')
        log.warning('See log file for more details: setup.log')


if __name__ == "__main__":
    parse_args()
    run_setup()
