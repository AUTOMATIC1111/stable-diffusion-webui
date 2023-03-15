"""Implements a FastAPI server to run the gradio interface. Note that some types in this
module use the Optional/Union notation so that they work correctly with pydantic."""

from __future__ import annotations
import time
import asyncio
import inspect
import json
import mimetypes
import os
import posixpath
import secrets
import traceback
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Type
from urllib.parse import urlparse

import fastapi
import markupsafe
import orjson
import pkg_resources
from fastapi import Depends, FastAPI, HTTPException, WebSocket, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
)
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.templating import Jinja2Templates
from jinja2.exceptions import TemplateNotFound
from starlette.responses import RedirectResponse
from starlette.websockets import WebSocketState

import gradio
import gradio.ranged_response as ranged_response
from gradio import utils
from gradio.data_classes import PredictBody, ResetBody
from gradio.documentation import document, set_documentation_group
from gradio.exceptions import Error
from gradio.queueing import Estimation, Event
from gradio.utils import cancel_tasks, run_coro_in_background, set_task_name

mimetypes.init()

STATIC_TEMPLATE_LIB = pkg_resources.resource_filename("gradio", "templates/")
STATIC_PATH_LIB = pkg_resources.resource_filename("gradio", "templates/frontend/static")
BUILD_PATH_LIB = pkg_resources.resource_filename("gradio", "templates/frontend/assets")
VERSION_FILE = pkg_resources.resource_filename("gradio", "version.txt")
with open(VERSION_FILE) as version_file:
    VERSION = version_file.read()


class ORJSONResponse(JSONResponse):
    media_type = "application/json"

    @staticmethod
    def _render(content: Any) -> bytes:
        return orjson.dumps(
            content,
            option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_PASSTHROUGH_DATETIME,
            default=str,
        )

    def render(self, content: Any) -> bytes:
        return ORJSONResponse._render(content)

    @staticmethod
    def _render_str(content: Any) -> str:
        return ORJSONResponse._render(content).decode("utf-8")


def toorjson(value):
    return markupsafe.Markup(
        ORJSONResponse._render_str(value)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("'", "\\u0027")
    )


templates = Jinja2Templates(directory=STATIC_TEMPLATE_LIB)
templates.env.filters["toorjson"] = toorjson


###########
# Auth
###########

class SessionInfo:

    def __init__(self, username, exp):
        self.exp = exp + time.time()
        self.username = username

    def expired(self) -> bool:
        return self.exp > time.time()


class App(FastAPI):
    """
    FastAPI App Wrapper
    """

    def __init__(self, **kwargs):
        self.tokens = {}
        self.auth = None
        self.blocks: gradio.Blocks | None = None
        self.state_holder = {}
        self.iterators = defaultdict(dict)
        self.lock = asyncio.Lock()
        self.queue_token = secrets.token_urlsafe(32)
        self.startup_events_triggered = False
        self.expired_users = {}
        self.current_token = None
        super().__init__(**kwargs)

    def configure_app(self, blocks: gradio.Blocks) -> None:
        auth = blocks.auth
        if auth is not None:
            if not callable(auth):
                self.auth = {account[0]: account[1] for account in auth}
            else:
                self.auth = auth
        else:
            self.auth = None

        self.blocks = blocks
        if hasattr(self.blocks, "_queue"):
            self.blocks._queue.set_access_token(self.queue_token)
        self.cwd = os.getcwd()
        self.favicon_path = blocks.favicon_path
        self.tokens = {}

    def get_blocks(self) -> gradio.Blocks:
        if self.blocks is None:
            raise ValueError("No Blocks has been configured for this app.")
        return self.blocks

    @staticmethod
    def create_app(blocks: gradio.Blocks) -> App:
        app = App(default_response_class=ORJSONResponse)
        app.configure_app(blocks)

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.get("/user")
        @app.get("/user/")
        def get_current_user(request: fastapi.Request) -> Optional[str]:
            token = request.cookies.get("access-token")
            return app.tokens.get(token)

        @app.get("/login_check")
        @app.get("/login_check/")
        def login_check(user: str = Depends(get_current_user)):
            if app.auth is None or not (user is None):
                return
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated"
            )

        async def ws_login_check(websocket: WebSocket) -> Optional[str]:
            token = websocket.cookies.get("access-token")
            return token  # token is returned to allow request in queue

        @app.get("/token")
        @app.get("/token/")
        def get_token(request: fastapi.Request) -> dict:
            token = request.cookies.get("access-token")
            return {"token": token, "user": app.tokens.get(token)}

        @app.get("/app_id")
        @app.get("/app_id/")
        def app_id(request: fastapi.Request) -> dict:
            return {"app_id": app.get_blocks().app_id}

        @app.post("/login")
        @app.post("/login/")
        def login(form_data: OAuth2PasswordRequestForm = Depends()):
            username, password = form_data.username, form_data.password
            if app.auth is None:
                base_path = "/" + os.getenv("Endpoint", "")
                return RedirectResponse(url=base_path, status_code=status.HTTP_302_FOUND)
            # 清除过期
            now = time.time()
            expired_users = list(app.expired_users.keys())
            for u in expired_users:
                t = app.expired_users.get(u, now)
                if now - t > 600:
                    del app.expired_users[u]
            #
            # if username in app.expired_users:
            #     raise HTTPException(status_code=403, detail="资源被占用，请稍后登录.")

            expire_time_or_auth = app.auth(username, password)
            auth = expire_time_or_auth >= 0 if isinstance(expire_time_or_auth, int) else expire_time_or_auth
            if (
                    not callable(app.auth)
                    and username in app.auth
                    and app.auth[username] == password
            ) or (callable(app.auth) and auth):
                token = secrets.token_urlsafe(16)
                for tk, u in app.tokens.items():
                    app.expired_users[u] = now
                app.tokens = {token: username}
                base_path = "/" + os.getenv("Endpoint", "")
                response = RedirectResponse(url=base_path, status_code=status.HTTP_302_FOUND)
                exp = expire_time_or_auth - int(time.time()) if isinstance(expire_time_or_auth,
                                                                           int) and expire_time_or_auth > 0 else 3600 * 24
                if exp > 24 * 3600 * 2:
                    exp = 24 * 3600 * 2

                response.set_cookie(key="access-token", value=token, httponly=True, expires=exp)
                app.current_token = token
                return response
            else:
                raise HTTPException(status_code=400, detail="Incorrect credentials.")

        ###############
        # Main Routes
        ###############

        @app.head("/", response_class=HTMLResponse)
        @app.get("/", response_class=HTMLResponse)
        def main(request: fastapi.Request, user: str = Depends(get_current_user)):
            mimetypes.add_type("application/javascript", ".js")
            blocks = app.get_blocks()

            if app.auth is None or not (user is None):
                config = app.get_blocks().config
            else:
                config = {
                    "auth_required": True,
                    "auth_message": blocks.auth_message,
                }

            try:
                template = (
                    "frontend/share.html" if blocks.share else "frontend/index.html"
                )
                return templates.TemplateResponse(
                    template, {"request": request, "config": config}
                )
            except TemplateNotFound:
                if blocks.share:
                    raise ValueError(
                        "Did you install Gradio from source files? Share mode only "
                        "works when Gradio is installed through the pip package."
                    )
                else:
                    raise ValueError(
                        "Did you install Gradio from source files? You need to build "
                        "the frontend by running /scripts/build_frontend.sh"
                    )

        @app.get("/config/", dependencies=[Depends(login_check)])
        @app.get("/config", dependencies=[Depends(login_check)])
        def get_config():
            return app.get_blocks().config

        @app.get("/static/{path:path}")
        def static_resource(path: str):
            static_file = safe_join(STATIC_PATH_LIB, path)
            if static_file is not None:
                return FileResponse(static_file)
            raise HTTPException(status_code=404, detail="Static file not found")

        @app.get("/assets/{path:path}")
        def build_resource(path: str):
            build_file = safe_join(BUILD_PATH_LIB, path)
            if build_file is not None:
                return FileResponse(build_file)
            raise HTTPException(status_code=404, detail="Build file not found")

        @app.get("/favicon.ico")
        async def favicon():
            blocks = app.get_blocks()
            if blocks.favicon_path is None:
                return static_resource("img/logo.svg")
            else:
                return FileResponse(blocks.favicon_path)

        @app.get("/file={path:path}")
        async def file(path: str, request: fastapi.Request):
            _, ex = os.path.splitext(path)
            ex = ex.lower()

            def is_front_file():
                front_files = {'.js', '.html', '.css', '.htm'}
                return ex in front_files

            def check_token():
                token = request.cookies.get("access-token", "")
                user = app.tokens.get(token)
                if app.auth is None or not (user is None):
                    return True
                return False

            if not is_front_file() and not check_token():
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated"
                )
            abs_path = str(Path(path).resolve())
            blocks = app.get_blocks()
            if utils.validate_url(path):
                return RedirectResponse(url=path, status_code=status.HTTP_302_FOUND)

            in_app_dir = Path(app.cwd).resolve() in Path(path).resolve().parents
            created_by_app = str(Path(path).resolve()) in set().union(
                *blocks.temp_file_sets
            )
            if in_app_dir or created_by_app:
                range_val = request.headers.get("Range", "").strip()
                if range_val.startswith("bytes=") and "-" in range_val:
                    range_val = range_val[6:]
                    start, end = range_val.split("-")
                    if start.isnumeric() and end.isnumeric():
                        start = int(start)
                        end = int(end)
                        response = ranged_response.RangedFileResponse(
                            abs_path,
                            ranged_response.OpenRange(start, end),
                            dict(request.headers),
                            stat_result=os.stat(abs_path),
                        )
                        return response
                return FileResponse(abs_path, headers={"Accept-Ranges": "bytes"})

            else:
                raise ValueError(
                    f"File cannot be fetched: {path}. All files must contained within the Gradio python app working directory, or be a temp file created by the Gradio python app."
                )

        @app.get("/file/{path:path}", dependencies=[Depends(login_check)])
        def file_deprecated(path: str, request: fastapi.Request):
            return file(path, request)

        @app.post("/reset/")
        @app.post("/reset")
        async def reset_iterator(body: ResetBody):
            if body.session_hash not in app.iterators:
                return {"success": False}
            async with app.lock:
                app.iterators[body.session_hash][body.fn_index] = None
                app.iterators[body.session_hash]["should_reset"].add(body.fn_index)
            return {"success": True}

        async def run_predict(
                body: PredictBody,
                request: Request | List[Request],
                fn_index_inferred: int,
                username: str = Depends(get_current_user),
        ):
            if hasattr(body, "session_hash"):
                if body.session_hash not in app.state_holder:
                    app.state_holder[body.session_hash] = {
                        _id: deepcopy(getattr(block, "value", None))
                        for _id, block in app.get_blocks().blocks.items()
                        if getattr(block, "stateful", False)
                    }
                session_state = app.state_holder[body.session_hash]
                iterators = app.iterators[body.session_hash]
                # The should_reset set keeps track of the fn_indices
                # that have been cancelled. When a job is cancelled,
                # the /reset route will mark the jobs as having been reset.
                # That way if the cancel job finishes BEFORE the job being cancelled
                # the job being cancelled will not overwrite the state of the iterator.
                # In all cases, should_reset will be the empty set the next time
                # the fn_index is run.
                app.iterators[body.session_hash]["should_reset"] = set([])
            else:
                session_state = {}
                iterators = {}
            event_id = getattr(body, "event_id", None)
            raw_input = body.data
            fn_index = body.fn_index
            batch = app.get_blocks().dependencies[fn_index_inferred]["batch"]
            if not (body.batched) and batch:
                raw_input = [raw_input]
            try:
                output = await app.get_blocks().process_api(
                    fn_index=fn_index_inferred,
                    inputs=raw_input,
                    request=request,
                    state=session_state,
                    iterators=iterators,
                    event_id=event_id,
                )
                iterator = output.pop("iterator", None)
                if hasattr(body, "session_hash"):
                    if fn_index in app.iterators[body.session_hash]["should_reset"]:
                        app.iterators[body.session_hash][fn_index] = None
                    else:
                        app.iterators[body.session_hash][fn_index] = iterator
                if isinstance(output, Error):
                    raise output
            except BaseException as error:
                show_error = app.get_blocks().show_error or isinstance(error, Error)
                traceback.print_exc()
                return JSONResponse(
                    content={"error": str(error) if show_error else None},
                    status_code=500,
                )

            if not (body.batched) and batch:
                output["data"] = output["data"][0]
            return output

        # had to use '/run' endpoint for Colab compatibility, '/api' supported for backwards compatibility
        @app.post("/run/{api_name}", dependencies=[Depends(login_check)])
        @app.post("/run/{api_name}/", dependencies=[Depends(login_check)])
        @app.post("/api/{api_name}", dependencies=[Depends(login_check)])
        @app.post("/api/{api_name}/", dependencies=[Depends(login_check)])
        async def predict(
                api_name: str,
                body: PredictBody,
                request: fastapi.Request,
                username: str = Depends(get_current_user),
        ):
            fn_index_inferred = None
            if body.fn_index is None:
                for i, fn in enumerate(app.get_blocks().dependencies):
                    if fn["api_name"] == api_name:
                        fn_index_inferred = i
                        break
                if fn_index_inferred is None:
                    return JSONResponse(
                        content={
                            "error": f"This app has no endpoint /api/{api_name}/."
                        },
                        status_code=500,
                    )
            else:
                fn_index_inferred = body.fn_index
            if not app.get_blocks().api_open and app.get_blocks().queue_enabled_for_fn(
                    fn_index_inferred
            ):
                if f"Bearer {app.queue_token}" != request.headers.get("Authorization"):
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Not authorized to skip the queue",
                    )

            # If this fn_index cancels jobs, then the only input we need is the
            # current session hash
            if app.get_blocks().dependencies[fn_index_inferred]["cancels"]:
                body.data = [body.session_hash]
            if body.request:
                if body.batched:
                    gr_request = [Request(**req) for req in body.request]
                else:
                    assert isinstance(body.request, dict)
                    gr_request = Request(**body.request)
            else:
                gr_request = Request(request)
            result = await run_predict(
                body=body,
                fn_index_inferred=fn_index_inferred,
                username=username,
                request=gr_request,
            )
            return result

        @app.websocket("/queue/join")
        async def join_queue(
                websocket: WebSocket,
                token: Optional[str] = Depends(ws_login_check),
        ):
            blocks = app.get_blocks()
            if app.auth is not None and token is None:
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                return
            if blocks._queue.server_path is None:
                app_url = get_server_url_from_ws_url(str(websocket.url))
                blocks._queue.set_url(app_url)
            await websocket.accept()
            # In order to cancel jobs, we need the session_hash and fn_index
            # to create a unique id for each job
            await websocket.send_json({"msg": "send_hash"})
            session_info = await websocket.receive_json()
            event = Event(
                websocket, session_info["session_hash"], session_info["fn_index"]
            )
            # set the token into Event to allow using the same token for call_prediction
            event.token = token
            event.session_hash = session_info["session_hash"]

            # Continuous events are not put in the queue  so that they do not
            # occupy the queue's resource as they are expected to run forever
            if blocks.dependencies[event.fn_index].get("every", 0):
                await cancel_tasks(set([f"{event.session_hash}_{event.fn_index}"]))
                await blocks._queue.reset_iterators(event.session_hash, event.fn_index)
                task = run_coro_in_background(
                    blocks._queue.process_events, [event], False
                )
                set_task_name(task, event.session_hash, event.fn_index, batch=False)
            else:
                rank = blocks._queue.push(event)

                if rank is None:
                    await blocks._queue.send_message(event, {"msg": "queue_full"})
                    await event.disconnect()
                    return
                estimation = blocks._queue.get_estimation()
                await blocks._queue.send_estimation(event, estimation, rank)
            while True:
                await asyncio.sleep(60)
                if websocket.application_state == WebSocketState.DISCONNECTED:
                    return

        @app.get(
            "/queue/status",
            dependencies=[Depends(login_check)],
            response_model=Estimation,
        )
        async def get_queue_status():
            return app.get_blocks()._queue.get_estimation()

        @app.get("/startup-events")
        async def startup_events():
            if not app.startup_events_triggered:
                app.get_blocks().startup_events()
                app.startup_events_triggered = True
                return True
            return False

        @app.get("/robots.txt", response_class=PlainTextResponse)
        def robots_txt():
            if app.get_blocks().share:
                return "User-agent: *\nDisallow: /"
            else:
                return "User-agent: *\nDisallow: "

        return app


########
# Helper functions
########


def safe_join(directory: str, path: str) -> str | None:
    """Safely path to a base directory to avoid escaping the base directory.
    Borrowed from: werkzeug.security.safe_join"""
    _os_alt_seps: List[str] = list(
        sep for sep in [os.path.sep, os.path.altsep] if sep is not None and sep != "/"
    )

    if path != "":
        filename = posixpath.normpath(path)
    else:
        return directory

    if (
            any(sep in filename for sep in _os_alt_seps)
            or os.path.isabs(filename)
            or filename == ".."
            or filename.startswith("../")
    ):
        return None
    return posixpath.join(directory, filename)


def get_types(cls_set: List[Type]):
    docset = []
    types = []
    for cls in cls_set:
        doc = inspect.getdoc(cls) or ""
        doc_lines = doc.split("\n")
        for line in doc_lines:
            if "value (" in line:
                types.append(line.split("value (")[1].split(")")[0])
        docset.append(doc_lines[1].split(":")[-1])
    return docset, types


def get_server_url_from_ws_url(ws_url: str):
    ws_url_parsed = urlparse(ws_url)
    scheme = "http" if ws_url_parsed.scheme == "ws" else "https"
    port = f":{ws_url_parsed.port}" if ws_url_parsed.port else ""
    return f"{scheme}://{ws_url_parsed.hostname}{port}{ws_url_parsed.path.replace('queue/join', '')}"


set_documentation_group("routes")


class Obj:
    """
    Using a class to convert dictionaries into objects. Used by the `Request` class.
    Credit: https://www.geeksforgeeks.org/convert-nested-python-dictionary-to-object/
    """

    def __init__(self, dict1):
        self.__dict__.update(dict1)

    def __str__(self) -> str:
        return str(self.__dict__)

    def __repr__(self) -> str:
        return str(self.__dict__)


@document()
class Request:
    """
    A Gradio request object that can be used to access the request headers, cookies,
    query parameters and other information about the request from within the prediction
    function. The class is a thin wrapper around the fastapi.Request class. Attributes
    of this class include: `headers`, `client`, `query_params`, and `path_params`,
    Example:
        import gradio as gr
        def echo(name, request: gr.Request):
            print("Request headers dictionary:", request.headers)
            print("IP address:", request.client.host)
            return name
        io = gr.Interface(echo, "textbox", "textbox").launch()
    """

    def __init__(self, request: fastapi.Request | None = None, **kwargs):
        """
        Can be instantiated with either a fastapi.Request or by manually passing in
        attributes (needed for websocket-based queueing).
        Parameters:
            request: A fastapi.Request
        """
        self.request = request
        self.kwargs: Dict = kwargs

    def dict_to_obj(self, d):
        if isinstance(d, dict):
            return json.loads(json.dumps(d), object_hook=Obj)
        else:
            return d

    def __getattr__(self, name):
        if self.request:
            return self.dict_to_obj(getattr(self.request, name))
        else:
            try:
                obj = self.kwargs[name]
            except KeyError:
                raise AttributeError(f"'Request' object has no attribute '{name}'")
            return self.dict_to_obj(obj)


@document()
def mount_gradio_app(
        app: fastapi.FastAPI,
        blocks: gradio.Blocks,
        path: str,
        gradio_api_url: str | None = None,
) -> fastapi.FastAPI:
    """Mount a gradio.Blocks to an existing FastAPI application.

    Parameters:
        app: The parent FastAPI application.
        blocks: The blocks object we want to mount to the parent app.
        path: The path at which the gradio application will be mounted.
        gradio_api_url: The full url at which the gradio app will run. This is only needed if deploying to Huggingface spaces of if the websocket endpoints of your deployed app are on a different network location than the gradio app. If deploying to spaces, set gradio_api_url to 'http://localhost:7860/'
    Example:
        from fastapi import FastAPI
        import gradio as gr
        app = FastAPI()
        @app.get("/")
        def read_main():
            return {"message": "This is your main app"}
        io = gr.Interface(lambda x: "Hello, " + x + "!", "textbox", "textbox")
        app = gr.mount_gradio_app(app, io, path="/gradio")
        # Then run `uvicorn run:app` from the terminal and navigate to http://localhost:8000/gradio.
    """
    blocks.dev_mode = False
    blocks.config = blocks.get_config_file()
    gradio_app = App.create_app(blocks)

    @app.on_event("startup")
    async def start_queue():
        if gradio_app.get_blocks().enable_queue:
            if gradio_api_url:
                gradio_app.get_blocks()._queue.set_url(gradio_api_url)
            gradio_app.get_blocks().startup_events()

    app.mount(path, gradio_app)
    return app
