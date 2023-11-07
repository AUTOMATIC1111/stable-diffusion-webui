import os
from pathlib import Path

from modules.paths_internal import script_path


def is_restartable() -> bool:
    """
    Return True if the webui is restartable (i.e. there is something watching to restart it with)
    """
    return bool(os.environ.get('SD_WEBUI_RESTART'))


def restart_program() -> None:
    """creates file tmp/restart and immediately stops the process, which webui.bat/webui.sh interpret as a command to start webui again"""

    tmpdir = Path(script_path) / "tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)
    (tmpdir / "restart").touch()

    stop_program()


def stop_program() -> None:
    os._exit(0)
