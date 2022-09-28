# this scripts installs necessary requirements and launches main program in webui.py

import subprocess
import os
import sys
import importlib.util
import shlex

dir_repos = "repositories"
dir_tmp = "tmp"

python = sys.executable
git = os.environ.get('GIT', "git")
torch_command = os.environ.get('TORCH_COMMAND', "pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113")
requirements_file = os.environ.get('REQS_FILE', "requirements_versions.txt")
commandline_args = os.environ.get('COMMANDLINE_ARGS', "")

k_diffusion_package = os.environ.get('K_DIFFUSION_PACKAGE', "git+https://github.com/crowsonkb/k-diffusion.git@9e3002b7cd64df7870e08527b7664eb2f2f5f3f5")
gfpgan_package = os.environ.get('GFPGAN_PACKAGE', "git+https://github.com/TencentARC/GFPGAN.git@8d2447a2d918f8eba5a4a01463fd48e45126a379")

stable_diffusion_commit_hash = os.environ.get('STABLE_DIFFUSION_COMMIT_HASH', "69ae4b35e0a0f6ee1af8bb9a5d0016ccb27e36dc")
taming_transformers_commit_hash = os.environ.get('TAMING_TRANSFORMERS_COMMIT_HASH', "24268930bf1dce879235a7fddd0b2355b84d7ea6")
codeformer_commit_hash = os.environ.get('CODEFORMER_COMMIT_HASH', "c5b4593074ba6214284d6acd5f1719b6c5d739af")
blip_commit_hash = os.environ.get('BLIP_COMMIT_HASH', "48211a1594f1321b00f14c9f7a5b4813144b2fb9")
ldsr_commit_hash = os.environ.get('LDSR_COMMIT_HASH', "abf33e7002d59d9085081bce93ec798dcabd49af")

args = shlex.split(commandline_args)


def extract_arg(args, name):
    return [x for x in args if x != name], name in args


args, skip_torch_cuda_test = extract_arg(args, '--skip-torch-cuda-test')


def repo_dir(name):
    return os.path.join(dir_repos, name)


def run(command, desc=None, errdesc=None):
    if desc is not None:
        print(desc)

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    if result.returncode != 0:

        message = f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {result.returncode}
stdout: {result.stdout.decode(encoding="utf8", errors="ignore") if len(result.stdout)>0 else '<empty>'}
stderr: {result.stderr.decode(encoding="utf8", errors="ignore") if len(result.stderr)>0 else '<empty>'}
"""
        raise RuntimeError(message)

    return result.stdout.decode(encoding="utf8", errors="ignore")


def run_python(code, desc=None, errdesc=None):
    return run(f'"{python}" -c "{code}"', desc, errdesc)


def run_pip(args, desc=None):
    return run(f'"{python}" -m pip {args} --prefer-binary', desc=f"Installing {desc}", errdesc=f"Couldn't install {desc}")


def check_run(command):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    return result.returncode == 0


def check_run_python(code):
    return check_run(f'"{python}" -c "{code}"')


def is_installed(package):
    try:
        spec = importlib.util.find_spec(package)
    except ModuleNotFoundError:
        return False

    return spec is not None


def git_clone(url, dir, name, commithash=None):
    # TODO clone into temporary dir and move if successful

    if os.path.exists(dir):
        return

    run(f'"{git}" clone "{url}" "{dir}"', f"Cloning {name} into {dir}...", f"Couldn't clone {name}")

    if commithash is not None:
        run(f'"{git}" -C {dir} checkout {commithash}', None, "Couldn't checkout {name}'s hash: {commithash}")


try:
    commit = run(f"{git} rev-parse HEAD").strip()
except Exception:
    commit = "<none>"

print(f"Python {sys.version}")
print(f"Commit hash: {commit}")


if not is_installed("torch") or not is_installed("torchvision"):
    run(f'"{python}" -m {torch_command}', "Installing torch and torchvision", "Couldn't install torch")

if not skip_torch_cuda_test:
    run_python("import torch; assert torch.cuda.is_available(), 'Torch is not able to use GPU; add --skip-torch-cuda-test to COMMANDLINE_ARGS variable to disable this check'")

if not is_installed("k_diffusion.sampling"):
    run_pip(f"install {k_diffusion_package}", "k-diffusion")

if not check_run_python("import k_diffusion; import inspect; assert 'eta' in inspect.signature(k_diffusion.sampling.sample_euler_ancestral).parameters"):
    print(f"k-diffusion does not have 'eta' parameter; reinstalling latest version")
    try:
        run_pip(f"install --upgrade --force-reinstall {k_diffusion_package}", "k-diffusion")
    except RuntimeError as e:
        print(str(e))

if not is_installed("gfpgan"):
    run_pip(f"install {gfpgan_package}", "gfpgan")

os.makedirs(dir_repos, exist_ok=True)

git_clone("https://github.com/CompVis/stable-diffusion.git", repo_dir('stable-diffusion'), "Stable Diffusion", stable_diffusion_commit_hash)
git_clone("https://github.com/CompVis/taming-transformers.git", repo_dir('taming-transformers'), "Taming Transformers", taming_transformers_commit_hash)
git_clone("https://github.com/sczhou/CodeFormer.git", repo_dir('CodeFormer'), "CodeFormer", codeformer_commit_hash)
git_clone("https://github.com/salesforce/BLIP.git", repo_dir('BLIP'), "BLIP", blip_commit_hash)
# Using my repo until my changes are merged, as this makes interfacing with our version of SD-web a lot easier
git_clone("https://github.com/Hafiidz/latent-diffusion", repo_dir('latent-diffusion'), "LDSR", ldsr_commit_hash)

if not is_installed("lpips"):
    run_pip(f"install -r {os.path.join(repo_dir('CodeFormer'), 'requirements.txt')}", "requirements for CodeFormer")

run_pip(f"install -r {requirements_file}", "requirements for Web UI")

sys.argv += args


def start_webui():
    print(f"Launching Web UI with arguments: {' '.join(sys.argv[1:])}")
    import webui
    webui.webui()

if __name__ == "__main__":
    start_webui()
