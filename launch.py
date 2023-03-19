# this scripts installs necessary requirements and launches main program in webui.py
import subprocess
import os
import sys
import importlib.util
import shlex
import platform
import argparse
import json
import warnings
from rich import print

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--ui-settings-file", type=str, default='config.json')
parser.add_argument("--data-dir", type=str, default=os.path.dirname(os.path.realpath(__file__)))
args, _ = parser.parse_known_args(sys.argv)

script_path = os.path.dirname(__file__)
data_path = os.getcwd()
dir_repos = "repositories"
dir_extensions = "extensions"
python = sys.executable
git = os.environ.get('GIT', "git")
index_url = os.environ.get('INDEX_URL', "")
stored_commit_hash = None
skip_install = False

warnings.filterwarnings(action="ignore", category=UserWarning)

def check_python_version():
    is_windows = platform.system() == "Windows"
    major = sys.version_info.major
    minor = sys.version_info.minor
    micro = sys.version_info.micro

    if is_windows:
        supported_minors = [10]
    else:
        supported_minors = [9, 10, 11]

    if not (major == 3 and minor in supported_minors):
        import modules.errors
        modules.errors.print_error_explanation(f"Incompatible Python version: {major}.{minor}.{micro} required 3.9-3.11")


def commit_hash():
    global stored_commit_hash

    if stored_commit_hash is not None:
        return stored_commit_hash

    try:
        stored_commit_hash = run(f"{git} rev-parse HEAD").strip()
    except Exception:
        stored_commit_hash = "<none>"

    return stored_commit_hash


def extract_arg(args, name):
    return [x for x in args if x != name], name in args


def extract_opt(args, name):
    opt = None
    is_present = False
    if name in args:
        is_present = True
        idx = args.index(name)
        del args[idx]
        if idx < len(args) and args[idx][0] != "-":
            opt = args[idx]
            del args[idx]
    return args, is_present, opt


def run(command, desc=None, errdesc=None, custom_env=None, live=False):
    if desc is not None:
        print(desc)

    if live:
        result = subprocess.run(command, shell=True, env=os.environ if custom_env is None else custom_env)
        if result.returncode != 0:
            raise RuntimeError(f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {result.returncode}""")

        return ""

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, env=os.environ if custom_env is None else custom_env)

    if result.returncode != 0:

        message = f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {result.returncode}
stdout: {result.stdout.decode(encoding="utf8", errors="ignore") if len(result.stdout)>0 else '<empty>'}
stderr: {result.stderr.decode(encoding="utf8", errors="ignore") if len(result.stderr)>0 else '<empty>'}
"""
        raise RuntimeError(message)

    return result.stdout.decode(encoding="utf8", errors="ignore")


def check_run(command):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    return result.returncode == 0


def is_installed(package):
    try:
        spec = importlib.util.find_spec(package)
    except ModuleNotFoundError:
        return False

    return spec is not None


def repo_dir(name):
    return os.path.join(script_path, dir_repos, name)


def run_python(code, desc=None, errdesc=None):
    return run(f'"{python}" -c "{code}"', desc, errdesc)


def run_pip(args, desc=None):
    if skip_install:
        return

    index_url_line = f' --index-url {index_url}' if index_url != '' else ''
    return run(f'"{python}" -m pip {args} --prefer-binary{index_url_line}', desc=f"Installing {desc}", errdesc=f"Couldn't install {desc}")


def check_run_python(code):
    return check_run(f'"{python}" -c "{code}"')


def git_clone(url, dir, name, commithash=None):
    # TODO clone into temporary dir and move if successful

    if os.path.exists(dir):
        if commithash is None:
            return

        current_hash = run(f'"{git}" -C "{dir}" rev-parse HEAD', None, f"Couldn't determine {name}'s hash: {commithash}").strip()
        if current_hash == commithash:
            return

        run(f'"{git}" -C "{dir}" fetch', f"Fetching updates for {name}...", f"Couldn't fetch {name}")
        run(f'"{git}" -C "{dir}" checkout {commithash}', f"Checking out commit for {name} with hash: {commithash}...", f"Couldn't checkout commit {commithash} for {name}")
        return

    run(f'"{git}" clone "{url}" "{dir}"', f"Cloning {name} into {dir}...", f"Couldn't clone {name}")

    if commithash is not None:
        run(f'"{git}" -C "{dir}" checkout {commithash}', None, "Couldn't checkout {name}'s hash: {commithash}")


def git_pull_recursive(dir):
    for subdir, _, _ in os.walk(dir):
        if os.path.exists(os.path.join(subdir, '.git')):
            try:
                output = subprocess.check_output([git, '-C', subdir, 'pull', '--autostash'])
                print(f"Pulled changes for repository in '{subdir}':\n{output.decode('utf-8').strip()}\n")
            except subprocess.CalledProcessError as e:
                print(f"Couldn't perform 'git pull' on repository in '{subdir}':\n{e.output.decode('utf-8').strip()}\n")


def run_extension_installer(extension_dir):
    path_installer = os.path.join(extension_dir, "install.py")
    if not os.path.isfile(path_installer):
        return

    try:
        env = os.environ.copy()
        env['PYTHONPATH'] = os.path.abspath(".")

        stdout = run(f'"{python}" "{path_installer}"', errdesc=f"Error running install.py for extension {extension_dir}", custom_env=env)
        if stdout is not None:
            print(stdout)
    except Exception as e:
        print(e, file=sys.stderr)


def list_extensions(settings_file):
    settings = {}

    try:
        if os.path.isfile(settings_file):
            with open(settings_file, "r", encoding="utf8") as file:
                settings = json.load(file)
    except Exception as e:
        print(e, file=sys.stderr)

    disabled_extensions = set(settings.get('disabled_extensions', []))

    return [x for x in os.listdir(os.path.join(data_path, dir_extensions)) if x not in disabled_extensions]


def run_extensions_installers(settings_file):
    if not os.path.isdir(dir_extensions):
        return

    for dirname_extension in list_extensions(settings_file):
        run_extension_installer(os.path.join(dir_extensions, dirname_extension))


def prepare_environment():
    global skip_install

    requirements_file = os.environ.get('REQS_FILE', "requirements_versions.txt")
    commandline_args = os.environ.get('COMMANDLINE_ARGS', "")

    gfpgan_package = os.environ.get('GFPGAN_PACKAGE', "git+https://github.com/TencentARC/GFPGAN.git@8d2447a2d918f8eba5a4a01463fd48e45126a379")
    clip_package = os.environ.get('CLIP_PACKAGE', "git+https://github.com/openai/CLIP.git@d50d76daa670286dd6cacf3bcd80b5e4823fc8e1")
    openclip_package = os.environ.get('OPENCLIP_PACKAGE', "git+https://github.com/mlfoundations/open_clip.git@bb6e834e9c70d9c27d0dc3ecedeebeaeb1ffad6b")

    stable_diffusion_repo = os.environ.get('STABLE_DIFFUSION_REPO', "https://github.com/Stability-AI/stablediffusion.git")
    taming_transformers_repo = os.environ.get('TAMING_TRANSFORMERS_REPO', "https://github.com/CompVis/taming-transformers.git")
    k_diffusion_repo = os.environ.get('K_DIFFUSION_REPO', 'https://github.com/crowsonkb/k-diffusion.git')
    codeformer_repo = os.environ.get('CODEFORMER_REPO', 'https://github.com/sczhou/CodeFormer.git')
    blip_repo = os.environ.get('BLIP_REPO', 'https://github.com/salesforce/BLIP.git')

    stable_diffusion_commit_hash = os.environ.get('STABLE_DIFFUSION_COMMIT_HASH', "47b6b607fdd31875c9279cd2f4f16b92e4ea958e")
    taming_transformers_commit_hash = os.environ.get('TAMING_TRANSFORMERS_COMMIT_HASH', "24268930bf1dce879235a7fddd0b2355b84d7ea6")
    # k_diffusion_commit_hash = os.environ.get('K_DIFFUSION_COMMIT_HASH', "5b3af030dd83e0297272d861c19477735d0317ec")
    k_diffusion_commit_hash = os.environ.get('K_DIFFUSION_COMMIT_HASH', "b43db16749d51055f813255eea2fdf1def801919")
    codeformer_commit_hash = os.environ.get('CODEFORMER_COMMIT_HASH', "c5b4593074ba6214284d6acd5f1719b6c5d739af")
    blip_commit_hash = os.environ.get('BLIP_COMMIT_HASH', "48211a1594f1321b00f14c9f7a5b4813144b2fb9")

    sys.argv += shlex.split(commandline_args)

    sys.argv, _ = extract_arg(sys.argv, '-f')
    sys.argv, update_all_extensions = extract_arg(sys.argv, '--update-all-extensions')
    sys.argv, skip_python_version_check = extract_arg(sys.argv, '--skip-python-version-check')
    sys.argv, skip_install = extract_arg(sys.argv, '--skip-install')
    ngrok = '--ngrok' in sys.argv

    if not skip_python_version_check:
        check_python_version()

    commit = commit_hash()

    if not is_installed("gfpgan"):
        run_pip(f"install {gfpgan_package}", "gfpgan")

    if not is_installed("clip"):
        run_pip(f"install {clip_package}", "clip")

    if not is_installed("open_clip"):
        run_pip(f"install {openclip_package}", "open_clip")

    if not is_installed("pyngrok") and ngrok:
        run_pip("install pyngrok", "ngrok")

    os.makedirs(os.path.join(script_path, dir_repos), exist_ok=True)

    git_clone(stable_diffusion_repo, repo_dir('stable-diffusion-stability-ai'), "Stable Diffusion", stable_diffusion_commit_hash)
    git_clone(taming_transformers_repo, repo_dir('taming-transformers'), "Taming Transformers", taming_transformers_commit_hash)
    git_clone(k_diffusion_repo, repo_dir('k-diffusion'), "K-diffusion", k_diffusion_commit_hash)
    git_clone(codeformer_repo, repo_dir('CodeFormer'), "CodeFormer", codeformer_commit_hash)
    git_clone(blip_repo, repo_dir('BLIP'), "BLIP", blip_commit_hash)

    if not is_installed("lpips"):
        run_pip(f"install -r \"{os.path.join(repo_dir('CodeFormer'), 'requirements.txt')}\"", "requirements for CodeFormer")

    if not os.path.isfile(requirements_file):
        requirements_file = os.path.join(script_path, requirements_file)
    run_pip(f"install -r \"{requirements_file}\"", "requirements for Web UI")

    if "--exit" in sys.argv:
        exit(0)

    run_extensions_installers(settings_file=args.ui_settings_file)

    if update_all_extensions:
        git_pull_recursive(os.path.join(data_path, dir_extensions))
    

def start():
    print(f"Launching server with arguments: {' '.join(sys.argv[1:])}")

    rich_installed = False
    try:
        from rich.pretty import install as pretty_install
        from rich.traceback import install as traceback_install
        from rich.console import Console
        console = Console(log_time=True, log_time_format='%H:%M:%S-%f')
        pretty_install(console=console)
        traceback_install(console=console, extra_lines=1, width=console.width, word_wrap=False, indent_guides=False, show_locals=True, max_frames=2)
        rich_installed = True
    except:
        import traceback
        pass # if rich is not installed do nothing

    import webui
    if '--nowebui' in sys.argv:
        webui.api_only()
    else:
        webui.webui()


if __name__ == "__main__":
    prepare_environment()
    start()
