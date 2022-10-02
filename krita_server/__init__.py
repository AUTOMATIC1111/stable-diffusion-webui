import contextlib
import threading
import time

import uvicorn
from webui import webui

from .app import app


class Server(uvicorn.Server):
    def install_signal_handlers(self):
        pass

    @contextlib.contextmanager
    def run_in_thread(self):
        thread = threading.Thread(target=self.run)
        thread.start()
        try:
            while not self.started:
                time.sleep(1e-3)
            yield
        finally:
            self.should_exit = True
            thread.join()


def start():
    config = uvicorn.Config(
        "krita_server:app", host="127.0.0.1", port=8000, log_level="info"
    )
    server = Server(config=config)

    with server.run_in_thread():
        webui()
