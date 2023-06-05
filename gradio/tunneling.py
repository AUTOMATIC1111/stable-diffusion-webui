import atexit
import os
import platform
import re
import stat
import subprocess
from pathlib import Path
from typing import List

import requests

VERSION = "0.2"
CURRENT_TUNNELS: List["Tunnel"] = []

machine = platform.machine()
if machine == "x86_64":
    machine = "amd64"

# Check if the file exist
BINARY_NAME = f"frpc_{platform.system().lower()}_{machine.lower()}"
BINARY_FILENAME = f"{BINARY_NAME}_v{VERSION}"
BINARY_PATH = f"{Path(__file__).parent / BINARY_FILENAME}"
EXTENSION = ".exe" if os.name == "nt" else ""


class Tunnel:
    def __init__(self, remote_host, remote_port, local_host, local_port, share_token):
        self.proc = None
        self.url = None
        self.remote_host = remote_host
        self.remote_port = remote_port
        self.local_host = local_host
        self.local_port = local_port
        self.share_token = share_token

    @staticmethod
    def download_binary():
        if not Path(BINARY_PATH).exists():
            binary_url = f"https://cdn-media.huggingface.co/frpc-gradio-{VERSION}/{BINARY_NAME}{EXTENSION}"
            resp = requests.get(binary_url)

            if resp.status_code == 403:
                raise OSError(
                    f"Cannot set up a share link as this platform is incompatible. Please "
                    f"create a GitHub issue with information about your platform: {platform.uname()}"
                )

            resp.raise_for_status()

            # Save file data to local copy
            with open(BINARY_PATH, "wb") as file:
                file.write(resp.content)
            st = os.stat(BINARY_PATH)
            os.chmod(BINARY_PATH, st.st_mode | stat.S_IEXEC)

    def start_tunnel(self) -> str:
        self.download_binary()
        self.url = self._start_tunnel(BINARY_PATH)
        return self.url

    def kill(self):
        if self.proc is not None:
            print(f"Killing tunnel {self.local_host}:{self.local_port} <> {self.url}")
            self.proc.terminate()
            self.proc = None

    def _start_tunnel(self, binary: str) -> str:
        CURRENT_TUNNELS.append(self)
        command = [
            binary,
            "http",
            "-n",
            self.share_token,
            "-l",
            str(self.local_port),
            "-i",
            self.local_host,
            "--uc",
            "--sd",
            "random",
            "--ue",
            "--server_addr",
            f"{self.remote_host}:{self.remote_port}",
            "--disable_log_color",
        ]
        self.proc = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        atexit.register(self.kill)
        url = ""
        while url == "":
            if self.proc.stdout is None:
                continue
            line = self.proc.stdout.readline()
            line = line.decode("utf-8")
            if "start proxy success" in line:
                result = re.search("start proxy success: (.+)\n", line)
                if result is None:
                    raise ValueError("Could not create share URL")
                else:
                    url = result.group(1)
        return url
