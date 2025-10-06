import sys
import copy
import shlex
import subprocess
from functools import wraps

BAD_FLAGS = ("--prefer-binary", '-I', '--ignore-installed')


def patch():
    if hasattr(subprocess, "__original_run"):
        return

    print("using uv")
    try:
        subprocess.run(['uv', '-V'])
    except FileNotFoundError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'uv'])

    subprocess.__original_run = subprocess.run

    @wraps(subprocess.__original_run)
    def patched_run(*args, **kwargs):
        _kwargs = copy.copy(kwargs)
        if args:
            command, *_args = args
        else:
            command, _args = _kwargs.pop("args", ""), ()

        if isinstance(command, str):
            command = shlex.split(command)
        else:
            command = [arg.strip() for arg in command]

        if not isinstance(command, list) or "pip" not in command:
            return subprocess.__original_run(*args, **kwargs)

        cmd = command[command.index("pip") + 1:]

        cmd = [arg for arg in cmd if arg not in BAD_FLAGS]

        modified_command = ["uv", "pip", *cmd]

        cmd_str = shlex.join([*modified_command, *_args])
        result = subprocess.__original_run(cmd_str, **_kwargs)
        if result.returncode != 0:
            return subprocess.__original_run(*args, **kwargs)
        return result

    subprocess.run = patched_run
