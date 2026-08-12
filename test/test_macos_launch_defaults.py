import os
from pathlib import Path
import subprocess


SCRIPT = Path(__file__).resolve().parents[1] / "webui-macos-env.sh"


def command_line_args(cpu_brand):
    environment = os.environ.copy()
    environment.update({
        "TEST_CPU_BRAND": cpu_brand,
        "TEST_MACOS_ENV_SCRIPT": str(SCRIPT),
    })
    command = """
sysctl() { printf '%s\n' "$TEST_CPU_BRAND"; }
SCRIPT_DIR="$(dirname "$TEST_MACOS_ENV_SCRIPT")"
source "$TEST_MACOS_ENV_SCRIPT"
printf '%s' "$COMMANDLINE_ARGS"
"""
    result = subprocess.run(["bash", "-c", command], check=True, capture_output=True, text=True, env=environment)
    return result.stdout.split()


def test_m1_family_uses_fp16_vae_default():
    assert "--no-half-vae" not in command_line_args("Apple M1")
    assert "--no-half-vae" not in command_line_args("Apple M1 Max")


def test_newer_apple_silicon_retains_fp32_vae_default():
    assert "--no-half-vae" in command_line_args("Apple M3 Pro")


def test_intel_retains_fp32_vae_default():
    assert "--no-half-vae" in command_line_args("Intel(R) Core(TM) i9")
