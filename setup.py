import os
import sys
import json
import time
import subprocess
import argparse


import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s | %(levelname)s | %(message)s', filename='setup.log', filemode='w')
log = logging.getLogger("sd-setup")
try:
    from rich.logging import RichHandler
    from rich.console import Console
    from rich.pretty import install as pretty_install
    from rich.traceback import install as traceback_install
    rh = RichHandler(show_time=True, omit_repeated_times=False, show_level=True, show_path=True, markup=True, rich_tracebacks=True, log_time_format='%H:%M:%S-%f')
    rh.setLevel(logging.INFO)
    log.addHandler(rh)
    # logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[RichHandler(show_time=True, omit_repeated_times=False, show_level=True, show_path=True, markup=True, rich_tracebacks=True, log_time_format='%H:%M:%S-%f')])
    console = Console(log_time=True, log_time_format='%H:%M:%S-%f')
    pretty_install(console=console)
    traceback_install(console=console, extra_lines=1, width=console.width, word_wrap=False, indent_guides=False, suppress=[])
except:
    pass
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    log.addHandler(sh)

parser = argparse.ArgumentParser(description = 'Setup for SD WebUI')
parser.add_argument('--quick', default = False, action='store_true', help = "Skip installing if setup.log is newer than repo timestamp, default: %(default)s")
parser.add_argument('--upgrade', default = False, action='store_true', help = "Upgrade main repository to latest version, default: %(default)s")
parser.add_argument('--update', default = False, action='store_true', help = "Update all extensions and submodules, default: %(default)s")
parser.add_argument('--skip-extensions', default = False, action='store_true', help = "Skips running individual extension installers, default: %(default)s")
args = parser.parse_args()


def install(package):
    def pip(args: str):
        log.debug(f"Running pip: {args}")
        result = subprocess.run(f'"{sys.executable}" -m pip {args}', shell=True, env=os.environ, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        txt = result.stdout.decode(encoding="utf8", errors="ignore")
        if len(result.stderr) > 0:
            txt = txt + '\n' + result.stderr.decode(encoding="utf8", errors="ignore")
        if result.returncode != 0:
            log.error(f'Error running pip with args: {args}')
            log.debug(f'Pip output: {txt}')
        return txt

    def installed():
        import pkg_resources
        ok = True
        try:
            pkgs = [p for p in package.split() if not p.startswith('-') and not p.startswith('git+') and not p.startswith('http') and not p.startswith('=')]
            for pkg in pkgs:
                p = pkg.split('==')
                spec = pkg_resources.working_set.by_key.get(p[0], None)
                if spec is None:
                    spec = pkg_resources.working_set.by_key.get(p[0].lower(), None)
                if spec is None:
                    spec = pkg_resources.working_set.by_key.get(p[0].replace('_', '-'), None)
                ok = ok and spec is not None
                version = pkg_resources.get_distribution(p[0]).version
                if ok and len(p) > 1:
                    ok = ok and version == p[1]
                    if not ok:
                        log.warning(f"Package wrong version found: {p[0]} {version} required {p[1]}")
                # if ok:
                #   log.debug(f"Package already installed: {p[0]} {version}")
                # else:
                #    log.debug(f"Package not installed: {p[0]} {version}")
            return ok
        except ModuleNotFoundError:
            log.debug(f"Package not installed: {pkgs}")
            return False

    if not installed():
        pip(f"install --upgrade {package}")


def git(args: str):
    # log.debug(f"Running git: {args}")
    git_cmd = os.environ.get('GIT', "git")
    result = subprocess.run(f'"{git_cmd}" {args}', shell=True, env=os.environ, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    txt = result.stdout.decode(encoding="utf8", errors="ignore")
    if len(result.stderr) > 0:
        txt = txt + '\n' + result.stderr.decode(encoding="utf8", errors="ignore")
    if result.returncode != 0:
        log.error(f'Error running git with args: {args}')
        log.debug(f'Git output: {txt}')
    return txt


def update(dir):
        branch = git(f'-C "{dir}" branch')
        if 'main' in branch:
            # log.debug(f'Using main branch {dir}')
            git(f'-C "{dir}" checkout main')
        elif 'master' in branch:
            # log.debug(f'Using master branch {dir}')
            git(f'-C "{dir}" checkout master')
        else:
            log.warning(f'Unknown branch for: {dir}')
        git(f'-C "{dir}" pull --rebase --autostash')


def clone(url, dir, commithash=None):

    if os.path.exists(dir):
        if commithash is None:
            return
        current_hash = git(f'-C "{dir}" rev-parse HEAD').strip()
        if current_hash != commithash:
            git(f'-C "{dir}" fetch')
            git(f'-C "{dir}" checkout {commithash}')
            return
    else:
        git(f'clone "{url}" "{dir}"')
        if commithash is not None:
            git(f'-C "{dir}" checkout {commithash}')


def parse_env():
    import shlex
    args = os.environ.get('COMMANDLINE_ARGS', "")
    sys.argv += shlex.split(args)


def check_python():
    import platform
    supported_minors = [10] if platform.system() != "Windows" else [9, 10, 11]
    log.info(f'Python {platform.python_version()} on {platform.system()}')
    if not (sys.version_info.major == 3 and sys.version_info.minor in supported_minors):
        raise RuntimeError(f"Incompatible Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} required 3.9-3.11")


def check_torch():
    install(f'torch torchaudio torchvision --extra-index-url https://download.pytorch.org/whl/cu118')
    try:
        import torch;
        log.info(f'Torch {torch.__version__}')
        if not torch.cuda.is_available():
            log.warning("Torch repoorts CUDA not available")
        else:
            log.info(f'Torch detected GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}')
            if torch.version.cuda:
                log.info(f'Torch backend: nVidia CUDA {torch.version.cuda} cuDNN {torch.backends.cudnn.version()}')
            elif torch.version.hip:
                log.info(f'Torch backend: AMD ROCm HIP {torch.version.hip}')
            else:
                log.warning(f'Unknown Torch backend')
    except:
        pass


def install_packages():
    log.info('Installing packages')
    gfpgan_package = os.environ.get('GFPGAN_PACKAGE', "git+https://github.com/TencentARC/GFPGAN.git@8d2447a2d918f8eba5a4a01463fd48e45126a379")
    clip_package = os.environ.get('CLIP_PACKAGE', "git+https://github.com/openai/CLIP.git@d50d76daa670286dd6cacf3bcd80b5e4823fc8e1")
    openclip_package = os.environ.get('OPENCLIP_PACKAGE', "git+https://github.com/mlfoundations/open_clip.git@bb6e834e9c70d9c27d0dc3ecedeebeaeb1ffad6b")
    xformers_package = os.environ.get('XFORMERS_PACKAGE', 'xformers==0.0.18')
    install(gfpgan_package)
    install(clip_package)
    install(openclip_package)
    install(f'--no-deps {xformers_package}')


def install_repositories():
    def dir(name):
        return os.path.join(os.path.dirname(__file__), 'repositories', name)

    log.info('Installing repositories')
    os.makedirs(os.path.join(os.path.dirname(__file__), 'repositories'), exist_ok=True)
    stable_diffusion_repo = os.environ.get('STABLE_DIFFUSION_REPO', "https://github.com/Stability-AI/stablediffusion.git")
    stable_diffusion_commit = os.environ.get('STABLE_DIFFUSION_COMMIT_HASH', "cf1d67a6fd5ea1aa600c4df58e5b47da45f6bdbf")
    clone(stable_diffusion_repo, dir('stable-diffusion-stability-ai'), stable_diffusion_commit)
    taming_transformers_repo = os.environ.get('TAMING_TRANSFORMERS_REPO', "https://github.com/CompVis/taming-transformers.git")
    taming_transformers_commit = os.environ.get('TAMING_TRANSFORMERS_COMMIT_HASH', "3ba01b241669f5ade541ce990f7650a3b8f65318")
    clone(taming_transformers_repo, dir('taming-transformers'), taming_transformers_commit)
    k_diffusion_repo = os.environ.get('K_DIFFUSION_REPO', 'https://github.com/crowsonkb/k-diffusion.git')
    k_diffusion_commit = os.environ.get('K_DIFFUSION_COMMIT_HASH', "b43db16749d51055f813255eea2fdf1def801919")
    clone(k_diffusion_repo, dir('k-diffusion'), k_diffusion_commit)
    codeformer_repo = os.environ.get('CODEFORMER_REPO', 'https://github.com/sczhou/CodeFormer.git')
    codeformer_commit = os.environ.get('CODEFORMER_COMMIT_HASH', "c5b4593074ba6214284d6acd5f1719b6c5d739af")
    clone(codeformer_repo, dir('CodeFormer'), codeformer_commit)
    blip_repo = os.environ.get('BLIP_REPO', 'https://github.com/salesforce/BLIP.git')
    blip_commit = os.environ.get('BLIP_COMMIT_HASH', "48211a1594f1321b00f14c9f7a5b4813144b2fb9")
    clone(blip_repo, dir('BLIP'), blip_commit)


def install_extensions():
    def list_extensions(dir):
        settings = {}
        if os.path.isfile('config.json'):
            with open('config.json', "r", encoding="utf8") as file:
                settings = json.load(file)
        if settings.get('disable_all_extensions', 'none') != 'none':
            log.debug(f'Disabled extensions: all')
            return []
        else:
            disabled_extensions = set(settings.get('disabled_extensions', []))
            if len(disabled_extensions) > 0:
                log.debug(f'Disabled extensions: {disabled_extensions}')
            return [x for x in os.listdir(dir) if x not in disabled_extensions and not x.startswith('.')]

    def run_extension_installer(extension_dir):
        path_installer = os.path.join(extension_dir, "install.py")
        if not os.path.isfile(path_installer):
            return
        try:
            log.debug(f"Running extension installer: {path_installer}")
            env = os.environ.copy()
            env['PYTHONPATH'] = os.path.abspath(".")
            result = subprocess.run(f'"{sys.executable}" "{path_installer}"', shell=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                txt = result.stdout.decode(encoding="utf8", errors="ignore")
                if len(result.stderr) > 0:
                    txt = txt + '\n' + result.stderr.decode(encoding="utf8", errors="ignore")
                log.error(f'Error running extension installer: {txt}')
        except Exception as e:
            log.error(f'Exception running extension installer: {e}')


    extensions_builtin_dir = os.path.join(os.path.dirname(__file__), 'extensions-builtin')
    extensions = list_extensions(extensions_builtin_dir)
    log.info(f'Built-in extensions: {extensions}')
    for ext in extensions:
        if args.update:
            update(os.path.join(extensions_builtin_dir, ext))
        if not args.skip_extensions:
            run_extension_installer(os.path.join(extensions_builtin_dir, ext))

    extensions_dir = os.path.join(os.path.dirname(__file__), 'extensions')
    extensions = list_extensions(extensions_dir)
    log.info(f'Enabled extensions: {extensions}')
    for ext in extensions:
        if args.update:
            update(os.path.join(extensions_dir, ext))
        if not args.skip_extensions:
            run_extension_installer(os.path.join(extensions_dir, ext))


def install_submodules():
    log.info('Installing submodules')
    git(f'submodule --quiet update --init --recursive')
    if args.update:
        log.info('Updating submodules')
        submodules = git('submodule').splitlines()
        for submodule in submodules:
            name = submodule.split()[1].strip()
            update(name)


def install_requirements():
    log.info('Installing requirements')
    f = open('requirements.txt', 'r')
    lines = [line.strip() for line in f.readlines() if line.strip() != '']
    for line in lines:
        install(line)


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


def check_version():
    ver = git('log -1 --pretty=format:"%h %ad"')
    log.info(f'Version: {ver}')
    hash = git('rev-parse HEAD')
    import requests
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    commits = requests.get('https://api.github.com/repos/vladmandic/automatic/branches/master').json()
    if commits['commit']['sha'] != hash:
        if args.upgrade:
            update('.')
            ver = git('log -1 --pretty=format:"%h %ad"')
            log.info(f'Updated to version: {ver}')
        else:
            log.warning(f'Latest available version: {commits["commit"]["commit"]["author"]["date"]}')
    if args.update:
        log.info('Updating Wiki')
        update(os.path.join(os.path.dirname(__file__), "wiki"))
        update(os.path.join(os.path.dirname(__file__), "wiki", "origin-wiki"))


def check_timestamp():
    if not os.path.isfile('setup.log'):
        return False
    setup_time = os.path.getmtime('setup.log')
    log.debug(f'Previous setup time: {time.ctime(setup_time)}')
    version_time = int(git('log -1 --pretty=format:"%at"'))
    log.debug(f'Repository update time: {time.ctime(int(version_time))}')
    return setup_time >= version_time

def run_setup(quick = False):
    parse_env()
    check_python()
    if (quick or args.quick) and check_timestamp():
        log.info('Attempting quick setup')
        return
    log.info("Running setup")
    install_requirements()
    check_version()
    install_packages()
    install_repositories()
    install_submodules()
    install_extensions()
    install_requirements()

if __name__ == "__main__":
    run_setup()
    set_environment()
    check_torch()
