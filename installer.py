import os
import sys
import json
import time
import shutil
import logging
import platform
import subprocess
import cProfile
import pkg_resources


class Dot(dict): # dot notation access to dictionary attributes
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


version = None
log = logging.getLogger("sd")
log_file = os.path.join(os.path.dirname(__file__), 'sdnext.log')
log_rolled = False
first_call = True
quick_allowed = True
errors = 0
opts = {}
args = Dot({
    'debug': False,
    'reset': False,
    'profile': False,
    'upgrade': False,
    'skip_extensions': False,
    'skip_requirements': False,
    'skip_git': False,
    'skip_torch': False,
    'use_directml': False,
    'use_ipex': False,
    'use_cuda': False,
    'use_rocm': False,
    'experimental': False,
    'test': False,
    'tls_selfsign': False,
    'reinstall': False,
    'version': False,
    'ignore': False,
})
git_commit = "unknown"


# setup console and file logging
def setup_logging():

    class RingBuffer(logging.StreamHandler):
        def __init__(self, capacity):
            super().__init__()
            self.capacity = capacity
            self.buffer = []
            self.formatter = logging.Formatter('{ "asctime":"%(asctime)s", "created":%(created)f, "facility":"%(name)s", "pid":%(process)d, "tid":%(thread)d, "level":"%(levelname)s", "module":"%(module)s", "func":"%(funcName)s", "msg":"%(message)s" }')

        def emit(self, record):
            msg = self.format(record)
            # self.buffer.append(json.loads(msg))
            self.buffer.append(msg)
            if len(self.buffer) > self.capacity:
                self.buffer.pop(0)

        def get(self):
            return self.buffer

    from functools import partial, partialmethod
    from logging.handlers import RotatingFileHandler
    from rich.theme import Theme
    from rich.logging import RichHandler
    from rich.console import Console
    from rich.pretty import install as pretty_install
    from rich.traceback import install as traceback_install

    if args.log:
        global log_file # pylint: disable=global-statement
        log_file = args.log

    logging.TRACE = 25
    logging.addLevelName(logging.TRACE, 'TRACE')
    logging.Logger.trace = partialmethod(logging.Logger.log, logging.TRACE)
    logging.trace = partial(logging.log, logging.TRACE)

    level = logging.DEBUG if args.debug else logging.INFO
    log.setLevel(logging.DEBUG) # log to file is always at level debug for facility `sd`
    console = Console(log_time=True, log_time_format='%H:%M:%S-%f', theme=Theme({
        "traceback.border": "black",
        "traceback.border.syntax_error": "black",
        "inspect.value.border": "black",
    }))
    logging.basicConfig(level=logging.ERROR, format='%(asctime)s | %(name)s | %(levelname)s | %(module)s | %(message)s', handlers=[logging.NullHandler()]) # redirect default logger to null
    pretty_install(console=console)
    traceback_install(console=console, extra_lines=1, max_frames=10, width=console.width, word_wrap=False, indent_guides=False, suppress=[])
    while log.hasHandlers() and len(log.handlers) > 0:
        log.removeHandler(log.handlers[0])

    # handlers
    rh = RichHandler(show_time=True, omit_repeated_times=False, show_level=True, show_path=False, markup=False, rich_tracebacks=True, log_time_format='%H:%M:%S-%f', level=level, console=console)
    rh.setLevel(level)
    log.addHandler(rh)

    fh = RotatingFileHandler(log_file, maxBytes=32*1024*1024, backupCount=9, encoding='utf-8', delay=True) # 10MB default for log rotation
    global log_rolled # pylint: disable=global-statement
    if not log_rolled and args.debug and not args.log:
        fh.doRollover()
        log_rolled = True

    fh.formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(module)s | %(message)s')
    fh.setLevel(logging.DEBUG)
    log.addHandler(fh)

    rb = RingBuffer(100) # 100 entries default in log ring buffer
    rb.setLevel(level)
    log.addHandler(rb)
    log.buffer = rb.buffer

    # overrides
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("diffusers").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)
    logging.getLogger("ControlNet").handlers = log.handlers
    logging.getLogger("lycoris").handlers = log.handlers
    # logging.getLogger("DeepSpeed").handlers = log.handlers

def get_logfile():
    log_size = os.path.getsize(log_file) if os.path.exists(log_file) else 0
    log.info(f'Logger: file="{log_file}" level={logging.getLevelName(logging.DEBUG if args.debug else logging.INFO)} size={log_size} mode={"append" if not log_rolled else "create"}')
    return log_file


def custom_excepthook(exc_type, exc_value, exc_traceback):
    import traceback
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    log.error(f"Uncaught exception occurred: type={exc_type} value={exc_value}")
    if exc_traceback:
        format_exception = traceback.format_tb(exc_traceback)
        for line in format_exception:
            log.error(repr(line))


def print_dict(d):
    return ' '.join([f'{k}={v}' for k, v in d.items()])


def print_profile(profiler: cProfile.Profile, msg: str):
    from modules.errors import profile
    profile(profiler, msg)


# check if package is installed
def installed(package, friendly: str = None, reload = False, quiet = False):
    ok = True
    try:
        if reload:
            import imp # pylint: disable=deprecated-module
            imp.reload(pkg_resources)
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
                package_version = pkg_resources.get_distribution(p[0]).version
                # log.debug(f"Package version found: {p[0]} {package_version}")
                if len(p) > 1:
                    exact = package_version == p[1]
                    ok = ok and (exact or args.experimental)
                    if not exact and not quiet:
                        if args.experimental:
                            log.warning(f"Package allowing experimental: {p[0]} {package_version} required {p[1]}")
                        else:
                            log.warning(f"Package version mismatch: {p[0]} {package_version} required {p[1]}")
            else:
                if not quiet:
                    log.debug(f"Package not found: {p[0]}")
        return ok
    except Exception as e:
        log.debug(f"Package error: {pkgs} {e}")
        return False


def uninstall(package):
    if installed(package, package, quiet=True):
        log.warning(f'Uninstalling: {package}')
        pip(f"uninstall {package} --yes --quiet", ignore=True, quiet=True)


def pip(arg: str, ignore: bool = False, quiet: bool = False):
    arg = arg.replace('>=', '==')
    if not quiet:
        log.info(f'Installing package: {arg.replace("install", "").replace("--upgrade", "").replace("--no-deps", "").replace("--force", "").replace("  ", " ").strip()}')
    env_args = os.environ.get("PIP_EXTRA_ARGS", "")
    log.debug(f"Running pip: {arg} {env_args}")
    result = subprocess.run(f'"{sys.executable}" -m pip {arg} {env_args}', shell=True, check=False, env=os.environ, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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


# install package using pip if not already installed
def install(package, friendly: str = None, ignore: bool = False):
    if args.reinstall or args.upgrade:
        global quick_allowed # pylint: disable=global-statement
        quick_allowed = False
    if args.reinstall or not installed(package, friendly):
        pip(f"install --upgrade {package}", ignore=ignore)


# execute git command
def git(arg: str, folder: str = None, ignore: bool = False):
    if args.skip_git:
        return ''
    git_cmd = os.environ.get('GIT', "git")
    if git_cmd != "git":
        git_cmd = os.path.abspath(git_cmd)
    result = subprocess.run(f'"{git_cmd}" {arg}', check=False, shell=True, env=os.environ, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=folder or '.')
    txt = result.stdout.decode(encoding="utf8", errors="ignore")
    if len(result.stderr) > 0:
        txt += ('\n' if len(txt) > 0 else '') + result.stderr.decode(encoding="utf8", errors="ignore")
    txt = txt.strip()
    if result.returncode != 0 and not ignore:
        if "couldn't find remote ref" in txt: # not a git repo
            return txt
        global errors # pylint: disable=global-statement
        errors += 1
        log.error(f'Error running git: {folder} / {arg}')
        if 'or stash them' in txt:
            log.error(f'Local changes detected: check log for details: {log_file}')
        log.debug(f'Git output: {txt}')
    return txt

# switch to main branch as head can get detached
def branch(folder):
    if args.experimental:
        return None
    if not os.path.exists(os.path.join(folder, '.git')):
        return None
    b = git('branch', folder)
    if 'main' in b:
        b = 'main'
    elif 'master' in b:
        b = 'master'
    else:
        b = b.split('\n')[0].replace('*', '').strip()
    log.debug(f'Submodule: {folder} / {b}')
    git(f'checkout {b}', folder, ignore=True)
    return b


# update git repository
def update(folder, current_branch = False):
    try:
        git('config rebase.Autostash true')
    except Exception:
        pass
    if current_branch:
        git('pull --rebase --force', folder)
        return
    b = branch(folder)
    if branch is None:
        git('pull --rebase --force', folder)
    else:
        git(f'pull origin {b} --rebase --force', folder)


# clone git repository
def clone(url, folder, commithash=None):
    if os.path.exists(folder):
        if commithash is None:
            update(folder)
        else:
            current_hash = git('rev-parse HEAD', folder).strip()
            if current_hash != commithash:
                git('fetch', folder)
                git(f'checkout {commithash}', folder)
                return
    else:
        log.info(f'Cloning repository: {url}')
        git(f'clone "{url}" "{folder}"')
        if commithash is not None:
            git(f'-C "{folder}" checkout {commithash}')


def get_platform():
    try:
        if platform.system() == 'Windows':
            release = platform.platform(aliased = True, terse = False)
        else:
            release = platform.release()
        return {
            # 'host': platform.node(),
            'arch': platform.machine(),
            'cpu': platform.processor(),
            'system': platform.system(),
            'release': release,
            # 'platform': platform.platform(aliased = True, terse = False),
            # 'version': platform.version(),
            'python': platform.python_version(),
        }
    except Exception as e:
        return { 'error': e }


# check python version
def check_python():
    supported_minors = [9, 10, 11]
    if args.quick:
        return
    log.info(f'Python {platform.python_version()} on {platform.system()}')
    if not (int(sys.version_info.major) == 3 and int(sys.version_info.minor) in supported_minors):
        log.error(f"Incompatible Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} required 3.{supported_minors}")
        if not args.ignore:
            sys.exit(1)
    if not args.skip_git:
        git_cmd = os.environ.get('GIT', "git")
        if shutil.which(git_cmd) is None:
            log.error('Git not found')
            if not args.ignore:
                sys.exit(1)
    else:
        git_version = git('--version', folder=None, ignore=False)
        log.debug(f'Git {git_version.replace("git version", "").strip()}')


# check torch version
def check_torch():
    if args.skip_torch:
        log.info('Skipping Torch tests')
        return
    if args.profile:
        pr = cProfile.Profile()
        pr.enable()
    allow_cuda = not (args.use_rocm or args.use_directml or args.use_ipex or args.use_openvino)
    allow_rocm = not (args.use_cuda or args.use_directml or args.use_ipex or args.use_openvino)
    allow_ipex = not (args.use_cuda or args.use_rocm or args.use_directml or args.use_openvino)
    allow_directml = not (args.use_cuda or args.use_rocm or args.use_ipex or args.use_openvino)
    allow_openvino = not (args.use_cuda or args.use_rocm or args.use_ipex or args.use_directml)
    log.debug(f'Torch overrides: cuda={args.use_cuda} rocm={args.use_rocm} ipex={args.use_ipex} diml={args.use_directml} openvino={args.use_openvino}')
    log.debug(f'Torch allowed: cuda={allow_cuda} rocm={allow_rocm} ipex={allow_ipex} diml={allow_directml} openvino={allow_openvino}')
    torch_command = os.environ.get('TORCH_COMMAND', '')
    xformers_package = os.environ.get('XFORMERS_PACKAGE', 'none')
    if not installed('onnxruntime', quiet=True) and not installed('onnxruntime-gpu', quiet=True): # allow either
        install('onnxruntime', 'onnxruntime', ignore=True)
    if torch_command != '':
        pass
    elif allow_cuda and (shutil.which('nvidia-smi') is not None or args.use_xformers or os.path.exists(os.path.join(os.environ.get('SystemRoot') or r'C:\Windows', 'System32', 'nvidia-smi.exe'))):
        log.info('nVidia CUDA toolkit detected: nvidia-smi present')
        if not args.use_xformers:
            torch_command = os.environ.get('TORCH_COMMAND', 'torch torchvision --index-url https://download.pytorch.org/whl/cu121')
        else:
            torch_command = os.environ.get('TORCH_COMMAND', 'torch torchvision --index-url https://download.pytorch.org/whl/cu118')
        xformers_package = os.environ.get('XFORMERS_PACKAGE', '--pre xformers' if opts.get('cross_attention_optimization', '') == 'xFormers' else 'none')
        install('onnxruntime-gpu', 'onnxruntime-gpu', ignore=True)
    elif allow_rocm and (shutil.which('rocminfo') is not None or os.path.exists('/opt/rocm/bin/rocminfo') or os.path.exists('/dev/kfd')):
        log.info('AMD ROCm toolkit detected')
        os.environ.setdefault('PYTORCH_HIP_ALLOC_CONF', 'garbage_collection_threshold:0.8,max_split_size_mb:512')
        os.environ.setdefault('TENSORFLOW_PACKAGE', 'tensorflow-rocm')
        try:
            command = subprocess.run('rocm_agent_enumerator', shell=True, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            amd_gpus = command.stdout.decode(encoding="utf8", errors="ignore").split('\n')
            amd_gpus = [x for x in amd_gpus if x and x != 'gfx000']
            log.debug(f'ROCm agents detected: {amd_gpus}')
        except Exception as e:
            log.debug(f'Run rocm_agent_enumerator failed: {e}')
            amd_gpus = []

        hip_visible_devices = [] # use the first available amd gpu by default
        for idx, gpu in enumerate(amd_gpus):
            if gpu in ['gfx1100', 'gfx1101', 'gfx1102']:
                hip_visible_devices.append((idx, gpu, 'navi3x'))
                break
            if gpu in ['gfx1030', 'gfx1031', 'gfx1032', 'gfx1034']: # experimental navi 2x support
                hip_visible_devices.append((idx, gpu, 'navi2x'))
                break
        if len(hip_visible_devices) > 0:
            idx, gpu, arch = hip_visible_devices[0]
            log.debug(f'ROCm agent used by default: idx={idx} gpu={gpu} arch={arch}')
            os.environ.setdefault('HIP_VISIBLE_DEVICES', str(idx))
            if arch == 'navi3x':
                os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')
                if os.environ.get('TENSORFLOW_PACKAGE') == 'tensorflow-rocm': # do not use tensorflow-rocm for navi 3x
                    os.environ['TENSORFLOW_PACKAGE'] = 'tensorflow==2.13.0'
            elif arch == 'navi2x':
                os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '10.3.0')
            else:
                log.debug(f'HSA_OVERRIDE_GFX_VERSION auto config is skipped for {gpu}')
        try:
            command = subprocess.run('hipconfig --version', shell=True, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            arr = command.stdout.decode(encoding="utf8", errors="ignore").split('.')
            if len(arr) >= 2:
                rocm_ver = f'{arr[0]}.{arr[1]}'
            log.debug(f'ROCm version detected: {rocm_ver}')
        except Exception as e:
            log.debug(f'ROCm hipconfig failed: {e}')
            rocm_ver = None
        if rocm_ver in {"5.7"}:
            torch_command = os.environ.get('TORCH_COMMAND', f'torch torchvision --pre --index-url https://download.pytorch.org/whl/nightly/rocm{rocm_ver}')
        elif rocm_ver in {"5.5", "5.6"}:
            torch_command = os.environ.get('TORCH_COMMAND', f'torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm{rocm_ver}')
        else:
            # ROCm 5.5 is oldest for PyTorch 2.1
            torch_command = os.environ.get('TORCH_COMMAND', 'torch torchvision --index-url https://download.pytorch.org/whl/rocm5.5')
        xformers_package = os.environ.get('XFORMERS_PACKAGE', 'none')
    elif allow_ipex and (args.use_ipex or shutil.which('sycl-ls') is not None or shutil.which('sycl-ls.exe') is not None or os.environ.get('ONEAPI_ROOT') is not None or os.path.exists('/opt/intel/oneapi') or os.path.exists("C:/Program Files (x86)/Intel/oneAPI") or os.path.exists("C:/oneAPI")):
        args.use_ipex = True # pylint: disable=attribute-defined-outside-init
        log.info('Intel OneAPI Toolkit detected')
        os.environ.setdefault('NEOReadDebugKeys', '1')
        os.environ.setdefault('ClDeviceGlobalMemSizeAvailablePercent', '100')
        if "linux" in sys.platform:
            torch_command = os.environ.get('TORCH_COMMAND', 'torch==2.1.0a0 torchvision==0.16.0a0 intel-extension-for-pytorch==2.1.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/')
            os.environ.setdefault('TENSORFLOW_PACKAGE', 'tensorflow==2.14.0 intel-extension-for-tensorflow[xpu]==2.14.0.1')
            install(os.environ.get('MKL_PACKAGE', 'mkl==2024.0.0'), 'mkl')
            install(os.environ.get('DPCPP_PACKAGE', 'mkl-dpcpp==2024.0.0'), 'mkl-dpcpp')
        else:
            if sys.version_info[1] == 11:
                pytorch_pip = 'https://github.com/Nuullll/intel-extension-for-pytorch/releases/download/v2.1.10%2Bxpu/torch-2.1.0a0+cxx11.abi-cp311-cp311-win_amd64.whl'
                torchvision_pip = 'https://github.com/Nuullll/intel-extension-for-pytorch/releases/download/v2.1.10%2Bxpu/torchvision-0.16.0a0+cxx11.abi-cp311-cp311-win_amd64.whl'
                ipex_pip = 'https://github.com/Nuullll/intel-extension-for-pytorch/releases/download/v2.1.10%2Bxpu/intel_extension_for_pytorch-2.1.10+xpu-cp311-cp311-win_amd64.whl'
            elif sys.version_info[1] == 10:
                pytorch_pip = 'https://github.com/Nuullll/intel-extension-for-pytorch/releases/download/v2.1.10%2Bxpu/torch-2.1.0a0+cxx11.abi-cp310-cp310-win_amd64.whl'
                torchvision_pip = 'https://github.com/Nuullll/intel-extension-for-pytorch/releases/download/v2.1.10%2Bxpu/torchvision-0.16.0a0+cxx11.abi-cp310-cp310-win_amd64.whl'
                ipex_pip = 'https://github.com/Nuullll/intel-extension-for-pytorch/releases/download/v2.1.10%2Bxpu/intel_extension_for_pytorch-2.1.10+xpu-cp310-cp310-win_amd64.whl'
            else:
                pytorch_pip = 'torch==2.1.0a0'
                torchvision_pip = 'torchvision==0.16.0a0'
                ipex_pip = 'intel-extension-for-pytorch==2.1.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/'
                install(os.environ.get('MKL_PACKAGE', 'mkl==2024.0.0'), 'mkl')
                install(os.environ.get('DPCPP_PACKAGE', 'mkl-dpcpp==2024.0.0'), 'mkl-dpcpp')
            torch_command = os.environ.get('TORCH_COMMAND', f'{pytorch_pip} {torchvision_pip} {ipex_pip}')
        install(os.environ.get('OPENVINO_PACKAGE', 'openvino==2023.2.0'), 'openvino', ignore=True)
        install('onnxruntime-openvino', 'onnxruntime-openvino', ignore=True)
    elif allow_openvino and args.use_openvino:
        log.info('Using OpenVINO')
        torch_command = os.environ.get('TORCH_COMMAND', 'torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu')
    else:
        machine = platform.machine()
        if sys.platform == 'darwin':
            torch_command = os.environ.get('TORCH_COMMAND', 'torch torchvision')
        elif allow_directml and args.use_directml and ('arm' not in machine and 'aarch' not in machine):
            log.info('Using DirectML Backend')
            torch_command = os.environ.get('TORCH_COMMAND', 'torch-directml')
            if 'torch' in torch_command and not args.version:
                install(torch_command, 'torch torchvision')
            install('onnxruntime-directml', 'onnxruntime-directml', ignore=True)
        else:
            log.info('Using CPU-only Torch')
            torch_command = os.environ.get('TORCH_COMMAND', 'torch torchvision')
    if 'torch' in torch_command and not args.version:
        # log.info('Installing torch - this may take a while')
        install(torch_command, 'torch torchvision')
    else:
        try:
            import torch
            log.info(f'Torch {torch.__version__}')
            if args.use_ipex and allow_ipex:
                import intel_extension_for_pytorch as ipex # pylint: disable=import-error, unused-import
                log.info(f'Torch backend: Intel IPEX {ipex.__version__}')
                if shutil.which('icpx') is not None:
                    log.info(f'{os.popen("icpx --version").read().rstrip()}')
                for device in range(torch.xpu.device_count()):
                    log.info(f'Torch detected GPU: {torch.xpu.get_device_name(device)} VRAM {round(torch.xpu.get_device_properties(device).total_memory / 1024 / 1024)} Compute Units {torch.xpu.get_device_properties(device).max_compute_units}')
            elif torch.cuda.is_available() and (allow_cuda or allow_rocm):
                # log.debug(f'Torch allocator: {torch.cuda.get_allocator_backend()}')
                if torch.version.cuda and allow_cuda:
                    log.info(f'Torch backend: nVidia CUDA {torch.version.cuda} cuDNN {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "N/A"}')
                elif torch.version.hip and allow_rocm:
                    log.info(f'Torch backend: AMD ROCm HIP {torch.version.hip}')
                else:
                    log.warning('Unknown Torch backend')
                for device in [torch.cuda.device(i) for i in range(torch.cuda.device_count())]:
                    log.info(f'Torch detected GPU: {torch.cuda.get_device_name(device)} VRAM {round(torch.cuda.get_device_properties(device).total_memory / 1024 / 1024)} Arch {torch.cuda.get_device_capability(device)} Cores {torch.cuda.get_device_properties(device).multi_processor_count}')
            else:
                try:
                    if args.use_directml and allow_directml:
                        import torch_directml # pylint: disable=import-error
                        dml_ver = pkg_resources.get_distribution("torch-directml")
                        log.info(f'Torch backend: DirectML ({dml_ver})')
                        for i in range(0, torch_directml.device_count()):
                            log.info(f'Torch detected GPU: {torch_directml.device_name(i)}')
                except Exception:
                    log.warning("Torch reports CUDA not available")
        except Exception as e:
            log.error(f'Could not load torch: {e}')
            if not args.ignore:
                sys.exit(1)
    if args.version:
        return
    try:
        if 'xformers' in xformers_package:
            install(f'--no-deps {xformers_package}', ignore=True)
            import torch
            import xformers
            if torch.__version__ != '2.0.1+cu118' and xformers.__version__ in ['0.0.22', '0.0.21', '0.0.20']:
                log.warning(f'Likely incompatible torch with: xformers=={xformers.__version__} installed: torch=={torch.__version__} required: torch==2.1.0+cu118 - build xformers manually or downgrade torch')
        elif not args.experimental and not args.use_xformers:
            uninstall('xformers')
    except Exception as e:
        log.debug(f'Cannot install xformers package: {e}')
    if opts.get('cuda_compile_backend', '') == 'hidet':
        install('hidet', 'hidet')
    if args.use_openvino or opts.get('cuda_compile_backend', '') == 'openvino_fx':
        install(os.environ.get('OPENVINO_PACKAGE', 'openvino==2023.2.0'), 'openvino')
        install('nncf==2.7.0', 'nncf')
        install('onnxruntime-openvino', 'onnxruntime-openvino', ignore=True) # TODO openvino: numpy version conflicts with tensorflow and doesn't support Python 3.11
        os.environ.setdefault('PYTORCH_TRACING_MODE', 'TORCHFX')
        os.environ.setdefault('NEOReadDebugKeys', '1')
        os.environ.setdefault('ClDeviceGlobalMemSizeAvailablePercent', '100')
    if args.profile:
        print_profile(pr, 'Torch')


# check modified files
def check_modified_files():
    if args.quick:
        return
    if args.skip_git:
        return
    try:
        res = git('status --porcelain')
        files = [x[2:].strip() for x in res.split('\n')]
        files = [x for x in files if len(x) > 0 and (not x.startswith('extensions')) and (not x.startswith('wiki')) and (not x.endswith('.json')) and ('.log' not in x)]
        deleted = [x for x in files if not os.path.exists(x)]
        if len(deleted) > 0:
            log.warning(f'Deleted files: {files}')
        files = [x for x in files if os.path.exists(x) and not os.path.isdir(x)]
        if len(files) > 0:
            log.warning(f'Modified files: {files}')
    except Exception:
        pass


# install required packages
def install_packages():
    if args.profile:
        pr = cProfile.Profile()
        pr.enable()
    log.info('Verifying packages')
    clip_package = os.environ.get('CLIP_PACKAGE', "git+https://github.com/openai/CLIP.git")
    install(clip_package, 'clip')
    invisiblewatermark_package = os.environ.get('INVISIBLEWATERMARK_PACKAGE', "git+https://github.com/patrickvonplaten/invisible-watermark.git@remove_onnxruntime_depedency")
    install(invisiblewatermark_package, 'invisible-watermark')
    install('pi-heif', 'pi_heif', ignore=True)
    tensorflow_package = os.environ.get('TENSORFLOW_PACKAGE', 'tensorflow==2.13.0')
    install(tensorflow_package, 'tensorflow-rocm' if 'rocm' in tensorflow_package else 'tensorflow', ignore=True)
    bitsandbytes_package = os.environ.get('BITSANDBYTES_PACKAGE', None)
    if bitsandbytes_package is not None:
        install(bitsandbytes_package, 'bitsandbytes', ignore=True)
    elif not args.experimental:
        uninstall('bitsandbytes')
    if args.profile:
        print_profile(pr, 'Packages')


# run extension installer
def run_extension_installer(folder):
    path_installer = os.path.realpath(os.path.join(folder, "install.py"))
    if not os.path.isfile(path_installer):
        return
    try:
        log.debug(f"Running extension installer: {path_installer}")
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
def list_extensions_folder(folder, quiet=False):
    name = os.path.basename(folder)
    disabled_extensions_all = opts.get('disable_all_extensions', 'none')
    if disabled_extensions_all != 'none':
        return []
    disabled_extensions = opts.get('disabled_extensions', [])
    enabled_extensions = [x for x in os.listdir(folder) if x not in disabled_extensions and not x.startswith('.')]
    if not quiet:
        log.info(f'Extensions: enabled={enabled_extensions} {name}')
    return enabled_extensions


# run installer for each installed and enabled extension and optionally update them
def install_extensions():
    if args.profile:
        pr = cProfile.Profile()
        pr.enable()
    pkg_resources._initialize_master_working_set() # pylint: disable=protected-access
    pkgs = [f'{p.project_name}=={p._version}' for p in pkg_resources.working_set] # pylint: disable=protected-access,not-an-iterable
    log.debug(f'Installed packages: {len(pkgs)}')
    from modules.paths import extensions_builtin_dir, extensions_dir
    extensions_duplicates = []
    extensions_enabled = []
    extension_folders = [extensions_builtin_dir] if args.safe else [extensions_builtin_dir, extensions_dir]
    for folder in extension_folders:
        if not os.path.isdir(folder):
            continue
        extensions = list_extensions_folder(folder, quiet=True)
        log.debug(f'Extensions all: {extensions}')
        for ext in extensions:
            if ext in extensions_enabled:
                extensions_duplicates.append(ext)
                continue
            extensions_enabled.append(ext)
            if args.upgrade:
                try:
                    update(os.path.join(folder, ext))
                except Exception:
                    log.error(f'Error updating extension: {os.path.join(folder, ext)}')
            if not args.skip_extensions:
                run_extension_installer(os.path.join(folder, ext))
            pkg_resources._initialize_master_working_set() # pylint: disable=protected-access
            try:
                updated = [f'{p.project_name}=={p._version}' for p in pkg_resources.working_set] # pylint: disable=protected-access,not-an-iterable
                diff = [x for x in updated if x not in pkgs]
                pkgs = updated
                if len(diff) > 0:
                    log.info(f'Extension installed packages: {ext} {diff}')
            except Exception as e:
                log.error(f'Extension installed unknown package: {e}')
    log.info(f'Extensions enabled: {extensions_enabled}')
    if len(extensions_duplicates) > 0:
        log.warning(f'Extensions duplicates: {extensions_duplicates}')
    if args.profile:
        print_profile(pr, 'Extensions')


# initialize and optionally update submodules
def install_submodules():
    if args.profile:
        pr = cProfile.Profile()
        pr.enable()
    log.info('Verifying submodules')
    txt = git('submodule')
    # log.debug(f'Submodules list: {txt}')
    if 'no submodule mapping found' in txt:
        log.warning('Attempting repository recover')
        git('add .')
        git('stash')
        git('merge --abort', folder=None, ignore=True)
        git('fetch --all')
        git('reset --hard origin/master')
        git('checkout master')
        txt = git('submodule')
        log.info('Continuing setup')
    git('submodule --quiet update --init --recursive')
    git('submodule --quiet sync --recursive')
    submodules = txt.splitlines()
    for submodule in submodules:
        try:
            name = submodule.split()[1].strip()
            if args.upgrade:
                update(name)
            else:
                branch(name)
        except Exception:
            log.error(f'Error updating submodule: {submodule}')
    if args.profile:
        print_profile(pr, 'Submodule')


def ensure_base_requirements():
    try:
        import setuptools # pylint: disable=unused-import
    except ImportError:
        install('setuptools', 'setuptools')
    try:
        import setuptools # pylint: disable=unused-import
    except ImportError:
        pass
    try:
        import rich # pylint: disable=unused-import
    except ImportError:
        install('rich', 'rich')
    try:
        import rich # pylint: disable=unused-import
    except ImportError:
        pass


def install_requirements():
    if args.profile:
        pr = cProfile.Profile()
        pr.enable()
    if args.skip_requirements and not args.requirements:
        return
    log.info('Verifying requirements')
    with open('requirements.txt', 'r', encoding='utf8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip() != '' and not line.startswith('#') and line is not None]
        for line in lines:
            install(line)
    if args.profile:
        print_profile(pr, 'Requirements')


# set environment variables controling the behavior of various libraries
def set_environment():
    log.debug('Setting environment tuning')
    os.environ.setdefault('ACCELERATE', 'True')
    os.environ.setdefault('ATTN_PRECISION', 'fp16')
    os.environ.setdefault('CUDA_AUTO_BOOST', '1')
    os.environ.setdefault('CUDA_CACHE_DISABLE', '0')
    os.environ.setdefault('CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT', '0')
    os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '0')
    os.environ.setdefault('CUDA_MODULE_LOADING', 'LAZY')
    os.environ.setdefault('TORCH_CUDNN_V8_API_ENABLED', '1')
    os.environ.setdefault('FORCE_CUDA', '1')
    os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
    os.environ.setdefault('HF_HUB_DISABLE_EXPERIMENTAL_WARNING', '1')
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
    os.environ.setdefault('K_DIFFUSION_USE_COMPILE', '0')
    os.environ.setdefault('NUMEXPR_MAX_THREADS', '16')
    os.environ.setdefault('PYTHONHTTPSVERIFY', '0')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'garbage_collection_threshold:0.8,max_split_size_mb:512')
    os.environ.setdefault('SAFETENSORS_FAST_GPU', '1')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
    os.environ.setdefault('USE_TORCH', '1')
    os.environ.setdefault('UVICORN_TIMEOUT_KEEP_ALIVE', '60')
    os.environ.setdefault('KINETO_LOG_LEVEL', '3')
    os.environ.setdefault('DO_NOT_TRACK', '1')
    os.environ.setdefault('HF_HUB_CACHE', opts.get('hfcache_dir', os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub')))
    log.debug(f'Cache folder: {os.environ.get("HF_HUB_CACHE")}')
    if sys.platform == 'darwin':
        os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')


def check_extensions():
    newest_all = os.path.getmtime('requirements.txt')
    from modules.paths import extensions_builtin_dir, extensions_dir
    extension_folders = [extensions_builtin_dir] if args.safe else [extensions_builtin_dir, extensions_dir]
    disabled_extensions_all = opts.get('disable_all_extensions', 'none')
    if disabled_extensions_all != 'none':
        log.info(f'Extensions: disabled={disabled_extensions_all}')
    else:
        log.info(f'Extensions: disabled={opts.get("disabled_extensions", [])}')
    for folder in extension_folders:
        if not os.path.isdir(folder):
            continue
        extensions = list_extensions_folder(folder)
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
            # log.debug(f'Extension version: {time.ctime(newest)} {folder}{os.pathsep}{ext}')
    return round(newest_all)


def get_version():
    global version # pylint: disable=global-statement
    if version is None:
        try:
            subprocess.run('git config log.showsignature false', stdout = subprocess.PIPE, stderr = subprocess.PIPE, shell=True, check=True)
        except Exception:
            pass
        try:
            res = subprocess.run('git log --pretty=format:"%h %ad" -1 --date=short', stdout = subprocess.PIPE, stderr = subprocess.PIPE, shell=True, check=True)
            ver = res.stdout.decode(encoding = 'utf8', errors='ignore') if len(res.stdout) > 0 else '  '
            githash, updated = ver.split(' ')
            res = subprocess.run('git remote get-url origin', stdout = subprocess.PIPE, stderr = subprocess.PIPE, shell=True, check=True)
            origin = res.stdout.decode(encoding = 'utf8', errors='ignore') if len(res.stdout) > 0 else ''
            res = subprocess.run('git rev-parse --abbrev-ref HEAD', stdout = subprocess.PIPE, stderr = subprocess.PIPE, shell=True, check=True)
            branch_name = res.stdout.decode(encoding = 'utf8', errors='ignore') if len(res.stdout) > 0 else ''
            version = {
                'app': 'sd.next',
                'updated': updated,
                'hash': githash,
                'url': origin.replace('\n', '') + '/tree/' + branch_name.replace('\n', '')
            }
        except Exception:
            version = { 'app': 'sd.next', 'version': 'unknown' }
    return version


# check version of the main repo and optionally upgrade it
def check_version(offline=False, reset=True): # pylint: disable=unused-argument
    if args.skip_all:
        return
    if not os.path.exists('.git'):
        log.error('Not a git repository')
        if not args.ignore:
            sys.exit(1)
    log.info(f'Version: {print_dict(get_version())}')
    if args.version or args.skip_git:
        return
    commit = git('rev-parse HEAD')
    global git_commit # pylint: disable=global-statement
    git_commit = commit[:7]
    if args.quick:
        return
    try:
        import requests
    except ImportError:
        return
    commits = None
    try:
        commits = requests.get('https://api.github.com/repos/vladmandic/automatic/branches/master', timeout=10).json()
        if commits['commit']['sha'] != commit:
            if args.upgrade:
                global quick_allowed # pylint: disable=global-statement
                quick_allowed = False
                log.info('Updating main repository')
                try:
                    git('add .')
                    git('stash')
                    update('.', current_branch=True)
                    # git('git stash pop')
                    ver = git('log -1 --pretty=format:"%h %ad"')
                    log.info(f'Upgraded to version: {ver}')
                except Exception:
                    if not reset:
                        log.error('Error during repository upgrade')
                    else:
                        log.warning('Retrying repository upgrade...')
                        git_reset()
                        check_version(offline=offline, reset=False)
            else:
                log.info(f'Latest published version: {commits["commit"]["sha"]} {commits["commit"]["commit"]["author"]["date"]}')
    except Exception as e:
        log.error(f'Failed to check version: {e} {commits}')


def update_wiki():
    if args.upgrade:
        log.info('Updating Wiki')
        try:
            update(os.path.join(os.path.dirname(__file__), "wiki"))
        except Exception:
            log.error('Error updating wiki')


# check if we can run setup in quick mode
def check_timestamp():
    if not quick_allowed or not os.path.isfile(log_file):
        return False
    if args.quick:
        return True
    if args.skip_git:
        return True
    ok = True
    setup_time = -1
    version_time = -1
    with open(log_file, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            if 'Setup complete without errors' in line:
                setup_time = int(line.split(' ')[-1])
    try:
        version_time = int(git('log -1 --pretty=format:"%at"'))
    except Exception as e:
        log.error(f'Error getting local repository version: {e}')
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
    if args.reinstall:
        ok = False
    return ok


def add_args(parser):
    group = parser.add_argument_group('Setup options')
    group.add_argument("--log", type=str, default=os.environ.get("SD_LOG", None), help="Set log file, default: %(default)s")
    group.add_argument('--debug', default = os.environ.get("SD_DEBUG",False), action='store_true', help = "Run installer with debug logging, default: %(default)s")
    group.add_argument('--reset', default = os.environ.get("SD_RESET",False), action='store_true', help = "Reset main repository to latest version, default: %(default)s")
    group.add_argument('--upgrade', default = os.environ.get("SD_UPGRADE",False), action='store_true', help = "Upgrade main repository to latest version, default: %(default)s")
    group.add_argument('--requirements', default = os.environ.get("SD_REQUIREMENTS",False), action='store_true', help = "Force re-check of requirements, default: %(default)s")
    group.add_argument('--quick', default = os.environ.get("SD_QUICK",False), action='store_true', help = "Bypass version checks, default: %(default)s")
    group.add_argument('--use-directml', default = os.environ.get("SD_USEDIRECTML",False), action='store_true', help = "Use DirectML if no compatible GPU is detected, default: %(default)s")
    group.add_argument("--use-openvino", default = os.environ.get("SD_USEOPENVINO",False), action='store_true', help="Use Intel OpenVINO backend, default: %(default)s")
    group.add_argument("--use-ipex", default = os.environ.get("SD_USEIPEX",False), action='store_true', help="Force use Intel OneAPI XPU backend, default: %(default)s")
    group.add_argument("--use-cuda", default = os.environ.get("SD_USECUDA",False), action='store_true', help="Force use nVidia CUDA backend, default: %(default)s")
    group.add_argument("--use-rocm", default = os.environ.get("SD_USEROCM",False), action='store_true', help="Force use AMD ROCm backend, default: %(default)s")
    group.add_argument("--use-xformers", default = os.environ.get("SD_USEXFORMERS",False), action='store_true', help="Force use xFormers cross-optimization, default: %(default)s")
    group.add_argument('--skip-requirements', default = os.environ.get("SD_SKIPREQUIREMENTS",False), action='store_true', help = "Skips checking and installing requirements, default: %(default)s")
    group.add_argument('--skip-extensions', default = os.environ.get("SD_SKIPEXTENSION",False), action='store_true', help = "Skips running individual extension installers, default: %(default)s")
    group.add_argument('--skip-git', default = os.environ.get("SD_SKIPGIT",False), action='store_true', help = "Skips running all GIT operations, default: %(default)s")
    group.add_argument('--skip-torch', default = os.environ.get("SD_SKIPTORCH",False), action='store_true', help = "Skips running Torch checks, default: %(default)s")
    group.add_argument('--skip-all', default = os.environ.get("SD_SKIPALL",False), action='store_true', help = "Skips running all checks, default: %(default)s")
    group.add_argument('--experimental', default = os.environ.get("SD_EXPERIMENTAL",False), action='store_true', help = "Allow unsupported versions of libraries, default: %(default)s")
    group.add_argument('--reinstall', default = os.environ.get("SD_REINSTALL",False), action='store_true', help = "Force reinstallation of all requirements, default: %(default)s")
    group.add_argument('--test', default = os.environ.get("SD_TEST",False), action='store_true', help = "Run test only and exit")
    group.add_argument('--version', default = False, action='store_true', help = "Print version information")
    group.add_argument('--ignore', default = os.environ.get("SD_IGNORE",False), action='store_true', help = "Ignore any errors and attempt to continue")
    group.add_argument('--safe', default = os.environ.get("SD_SAFE",False), action='store_true', help = "Run in safe mode with no user extensions")


def parse_args(parser):
    # command line args
    global args # pylint: disable=global-statement
    args = parser.parse_args()
    return args


def extensions_preload(parser):
    if args.profile:
        pr = cProfile.Profile()
        pr.enable()
    if args.safe:
        log.info('Running in safe mode without user extensions')
    try:
        from modules.script_loading import preload_extensions
        from modules.paths import extensions_builtin_dir, extensions_dir
        extension_folders = [extensions_builtin_dir] if args.safe else [extensions_builtin_dir, extensions_dir]
        preload_time = {}
        for ext_dir in extension_folders:
            t0 = time.time()
            preload_extensions(ext_dir, parser)
            t1 = time.time()
            preload_time[ext_dir] = round(t1 - t0, 2)
        log.info(f'Extension preload: {preload_time}')
    except Exception:
        log.error('Error running extension preloading')
    if args.profile:
        print_profile(pr, 'Preload')


def git_reset():
    log.warning('Running GIT reset')
    global quick_allowed # pylint: disable=global-statement
    quick_allowed = False
    git('add .')
    git('stash')
    git('merge --abort', folder=None, ignore=True)
    git('fetch --all')
    git('reset --hard origin/master')
    git('checkout master')
    git('submodule update --init --recursive')
    git('submodule sync --recursive')
    log.info('GIT reset complete')


def read_options():
    global opts # pylint: disable=global-statement
    if os.path.isfile(args.config):
        with open(args.config, "r", encoding="utf8") as file:
            try:
                opts = json.load(file)
                if type(opts) is str:
                    opts = json.loads(opts)
            except Exception as e:
                log.error(f'Error reading options file: {file} {e}')
