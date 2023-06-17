import threading
import logging
import uvicorn
import fastapi


class UvicornServer(uvicorn.Server):
    def __init__(self, app: fastapi.FastAPI, listen = None, port = None, keyfile = None, certfile = None, loop = "auto", http = "auto"):
        self.app: fastapi.FastAPI = app
        self.thread: threading.Thread = None
        self.wants_restart = False
        self.config = uvicorn.Config(
            app=self.app,
            host = "0.0.0.0" if listen else "127.0.0.1",
            port = port or 7861,
            loop = loop, # auto, asyncio, uvloop
            http = http, # auto, h11, httptools
            interface = "auto", # auto, asgi3, asgi2, wsgi
            ws = "auto", # auto, websockets, wsproto
            log_level = logging.WARNING,
            backlog = 4096, # default=2048
            timeout_keep_alive = 60, # default=5
            ssl_keyfile = keyfile,
            ssl_certfile = certfile,
            ws_max_size = 1024 * 1024 * 1024,  # default 16MB
        )
        super().__init__(config=self.config)

    def start(self):
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.wants_restart = False
        self.thread.start()

    def stop(self):
        self.should_exit = True
        self.thread.join()

    def restart(self):
        self.wants_restart = True
        self.stop()
        self.start()


class HypercornServer():
    def __init__(self, app: fastapi.FastAPI, listen = None, port = None, keyfile = None, certfile = None, loop = "auto", http = None):
        import asyncio
        import hypercorn
        self.app: fastapi.FastAPI = app
        self.server: HypercornServer = None
        self.thread = None
        self.task = None
        self.wants_restart = False
        self.loop = 'trio' if loop == 'auto' else loop # asyncio, uvloop, trio
        self.config = hypercorn.config.Config()
        self.config.bind = [f'{"0.0.0.0" if listen else "127.0.0.1"}:{port or 7861}']
        self.config.keyfile = keyfile
        self.config.certfile = certfile
        self.config.keep_alive_timeout = 60 # default=5
        self.config.backlog = 4096 # default=100
        self.config.loglevel = "WARNING"
        self.config.max_app_queue_size = 64 # default=10
        self.http = http # unused
        self.main_loop = asyncio.get_event_loop()

    def run(self):
        import trio
        from hypercorn.trio import serve
        self.server = trio.run(serve, self.app, self.config)

    def start(self):
        if self.loop == 'trio':
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
        elif self.loop == 'asyncio': # does not run in thread
            import asyncio
            from hypercorn.asyncio import serve
            self.server = serve(self.app, self.config)
            asyncio.run(self.server)
        elif self.loop == 'uvloop': # does not run in thread
            import uvloop
            from hypercorn.asyncio import serve
            uvloop.install()
            from hypercorn.asyncio import serve
            self.server = serve(self.app, self.config)
            asyncio.run(self.server)
