import os


def is_restartable() -> bool:
    """
    Return True if the webui is restartable (i.e. there is something watching to restart it with)
    """
    return bool(os.environ.get('SD_WEBUI_RESTART'))


def restart_program() -> None:
    """exit process with errorcode 1111, which webui.bat/webui.sh interpret as a command to start webui again"""
    os._exit(1111)


def stop_program() -> None:
    os._exit(0)
