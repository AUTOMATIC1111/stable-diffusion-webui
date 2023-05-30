from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import os
import pkgutil
import shutil
import tempfile
from concurrent.futures import CancelledError
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Optional

import fsspec.asyn
import httpx
import huggingface_hub
import requests
from huggingface_hub import SpaceStage
from websockets.legacy.protocol import WebSocketCommonProtocol

API_URL = "/api/predict/"
WS_URL = "/queue/join"
UPLOAD_URL = "/upload"
CONFIG_URL = "/config"
API_INFO_URL = "/info"
RAW_API_INFO_URL = "/info?serialize=False"
SPACE_FETCHER_URL = "https://gradio-space-api-fetcher-v2.hf.space/api"
RESET_URL = "/reset"
SPACE_URL = "https://hf.space/{}"

STATE_COMPONENT = "state"
INVALID_RUNTIME = [
    SpaceStage.NO_APP_FILE,
    SpaceStage.CONFIG_ERROR,
    SpaceStage.BUILD_ERROR,
    SpaceStage.RUNTIME_ERROR,
    SpaceStage.PAUSED,
]

__version__ = (pkgutil.get_data(__name__, "version.txt") or b"").decode("ascii").strip()


class TooManyRequestsError(Exception):
    """Raised when the API returns a 429 status code."""

    pass


class QueueError(Exception):
    """Raised when the queue is full or there is an issue adding a job to the queue."""

    pass


class InvalidAPIEndpointError(Exception):
    """Raised when the API endpoint is invalid."""

    pass


class SpaceDuplicationError(Exception):
    """Raised when something goes wrong with a Space Duplication."""

    pass


class Status(Enum):
    """Status codes presented to client users."""

    STARTING = "STARTING"
    JOINING_QUEUE = "JOINING_QUEUE"
    QUEUE_FULL = "QUEUE_FULL"
    IN_QUEUE = "IN_QUEUE"
    SENDING_DATA = "SENDING_DATA"
    PROCESSING = "PROCESSING"
    ITERATING = "ITERATING"
    PROGRESS = "PROGRESS"
    FINISHED = "FINISHED"
    CANCELLED = "CANCELLED"

    @staticmethod
    def ordering(status: Status) -> int:
        """Order of messages. Helpful for testing."""
        order = [
            Status.STARTING,
            Status.JOINING_QUEUE,
            Status.QUEUE_FULL,
            Status.IN_QUEUE,
            Status.SENDING_DATA,
            Status.PROCESSING,
            Status.PROGRESS,
            Status.ITERATING,
            Status.FINISHED,
            Status.CANCELLED,
        ]
        return order.index(status)

    def __lt__(self, other: Status):
        return self.ordering(self) < self.ordering(other)

    @staticmethod
    def msg_to_status(msg: str) -> Status:
        """Map the raw message from the backend to the status code presented to users."""
        return {
            "send_hash": Status.JOINING_QUEUE,
            "queue_full": Status.QUEUE_FULL,
            "estimation": Status.IN_QUEUE,
            "send_data": Status.SENDING_DATA,
            "process_starts": Status.PROCESSING,
            "process_generating": Status.ITERATING,
            "process_completed": Status.FINISHED,
            "progress": Status.PROGRESS,
        }[msg]


@dataclass
class ProgressUnit:
    index: Optional[int]
    length: Optional[int]
    unit: Optional[str]
    progress: Optional[float]
    desc: Optional[str]

    @classmethod
    def from_ws_msg(cls, data: list[dict]) -> list[ProgressUnit]:
        return [
            cls(
                index=d.get("index"),
                length=d.get("length"),
                unit=d.get("unit"),
                progress=d.get("progress"),
                desc=d.get("desc"),
            )
            for d in data
        ]


@dataclass
class StatusUpdate:
    """Update message sent from the worker thread to the Job on the main thread."""

    code: Status
    rank: int | None
    queue_size: int | None
    eta: float | None
    success: bool | None
    time: datetime | None
    progress_data: list[ProgressUnit] | None


def create_initial_status_update():
    return StatusUpdate(
        code=Status.STARTING,
        rank=None,
        queue_size=None,
        eta=None,
        success=None,
        time=datetime.now(),
        progress_data=None,
    )


@dataclass
class JobStatus:
    """The job status.

    Keeps track of the latest status update and intermediate outputs (not yet implements).
    """

    latest_status: StatusUpdate = field(default_factory=create_initial_status_update)
    outputs: list[Any] = field(default_factory=list)


@dataclass
class Communicator:
    """Helper class to help communicate between the worker thread and main thread."""

    lock: Lock
    job: JobStatus
    prediction_processor: Callable[..., tuple]
    reset_url: str
    should_cancel: bool = False


########################
# Network utils
########################


def is_valid_url(possible_url: str) -> bool:
    headers = {"User-Agent": "gradio (https://gradio.app/; team@gradio.app)"}
    try:
        head_request = requests.head(possible_url, headers=headers)
        if head_request.status_code == 405:
            return requests.get(possible_url, headers=headers).ok
        return head_request.ok
    except Exception:
        return False


async def get_pred_from_ws(
    websocket: WebSocketCommonProtocol,
    data: str,
    hash_data: str,
    helper: Communicator | None = None,
) -> dict[str, Any]:
    completed = False
    resp = {}
    while not completed:
        # Receive message in the background so that we can
        # cancel even while running a long pred
        task = asyncio.create_task(websocket.recv())
        while not task.done():
            if helper:
                with helper.lock:
                    if helper.should_cancel:
                        # Need to reset the iterator state since the client
                        # will not reset the session
                        async with httpx.AsyncClient() as http:
                            reset = http.post(
                                helper.reset_url, json=json.loads(hash_data)
                            )
                            # Retrieve cancel exception from task
                            # otherwise will get nasty warning in console
                            task.cancel()
                            await asyncio.gather(task, reset, return_exceptions=True)
                        raise CancelledError()
            # Need to suspend this coroutine so that task actually runs
            await asyncio.sleep(0.01)
        msg = task.result()
        resp = json.loads(msg)
        if helper:
            with helper.lock:
                has_progress = "progress_data" in resp
                status_update = StatusUpdate(
                    code=Status.msg_to_status(resp["msg"]),
                    queue_size=resp.get("queue_size"),
                    rank=resp.get("rank", None),
                    success=resp.get("success"),
                    time=datetime.now(),
                    eta=resp.get("rank_eta"),
                    progress_data=ProgressUnit.from_ws_msg(resp["progress_data"])
                    if has_progress
                    else None,
                )
                output = resp.get("output", {}).get("data", [])
                if output and status_update.code != Status.FINISHED:
                    try:
                        result = helper.prediction_processor(*output)
                    except Exception as e:
                        result = [e]
                    helper.job.outputs.append(result)
                helper.job.latest_status = status_update
        if resp["msg"] == "queue_full":
            raise QueueError("Queue is full! Please try again.")
        if resp["msg"] == "send_hash":
            await websocket.send(hash_data)
        elif resp["msg"] == "send_data":
            await websocket.send(data)
        completed = resp["msg"] == "process_completed"
    return resp["output"]


########################
# Data processing utils
########################


def download_tmp_copy_of_file(
    url_path: str, hf_token: str | None = None, dir: str | None = None
) -> tempfile._TemporaryFileWrapper:
    if dir is not None:
        os.makedirs(dir, exist_ok=True)
    headers = {"Authorization": "Bearer " + hf_token} if hf_token else {}
    prefix = Path(url_path).stem
    suffix = Path(url_path).suffix
    file_obj = tempfile.NamedTemporaryFile(
        delete=False,
        prefix=prefix,
        suffix=suffix,
        dir=dir,
    )
    with requests.get(url_path, headers=headers, stream=True) as r, open(
        file_obj.name, "wb"
    ) as f:
        shutil.copyfileobj(r.raw, f)
    return file_obj


def create_tmp_copy_of_file(
    file_path: str, dir: str | None = None
) -> tempfile._TemporaryFileWrapper:
    if dir is not None:
        os.makedirs(dir, exist_ok=True)
    prefix = Path(file_path).stem
    suffix = Path(file_path).suffix
    file_obj = tempfile.NamedTemporaryFile(
        delete=False,
        prefix=prefix,
        suffix=suffix,
        dir=dir,
    )
    shutil.copy2(file_path, file_obj.name)
    return file_obj


def get_mimetype(filename: str) -> str | None:
    if filename.endswith(".vtt"):
        return "text/vtt"
    mimetype = mimetypes.guess_type(filename)[0]
    if mimetype is not None:
        mimetype = mimetype.replace("x-wav", "wav").replace("x-flac", "flac")
    return mimetype


def get_extension(encoding: str) -> str | None:
    encoding = encoding.replace("audio/wav", "audio/x-wav")
    type = mimetypes.guess_type(encoding)[0]
    if type == "audio/flac":  # flac is not supported by mimetypes
        return "flac"
    elif type is None:
        return None
    extension = mimetypes.guess_extension(type)
    if extension is not None and extension.startswith("."):
        extension = extension[1:]
    return extension


def encode_file_to_base64(f: str | Path):
    with open(f, "rb") as file:
        encoded_string = base64.b64encode(file.read())
        base64_str = str(encoded_string, "utf-8")
        mimetype = get_mimetype(str(f))
        return (
            "data:"
            + (mimetype if mimetype is not None else "")
            + ";base64,"
            + base64_str
        )


def encode_url_to_base64(url: str):
    encoded_string = base64.b64encode(requests.get(url).content)
    base64_str = str(encoded_string, "utf-8")
    mimetype = get_mimetype(url)
    return (
        "data:" + (mimetype if mimetype is not None else "") + ";base64," + base64_str
    )


def encode_url_or_file_to_base64(path: str | Path):
    path = str(path)
    if is_valid_url(path):
        return encode_url_to_base64(path)
    else:
        return encode_file_to_base64(path)


def decode_base64_to_binary(encoding: str) -> tuple[bytes, str | None]:
    extension = get_extension(encoding)
    data = encoding.rsplit(",", 1)[-1]
    return base64.b64decode(data), extension


def strip_invalid_filename_characters(filename: str, max_bytes: int = 200) -> str:
    """Strips invalid characters from a filename and ensures that the file_length is less than `max_bytes` bytes."""
    filename = "".join([char for char in filename if char.isalnum() or char in "._- "])
    filename_len = len(filename.encode())
    if filename_len > max_bytes:
        while filename_len > max_bytes:
            if len(filename) == 0:
                break
            filename = filename[:-1]
            filename_len = len(filename.encode())
    return filename


def sanitize_parameter_names(original_name: str) -> str:
    """Cleans up a Python parameter name to make the API info more readable."""
    return (
        "".join([char for char in original_name if char.isalnum() or char in " _"])
        .replace(" ", "_")
        .lower()
    )


def decode_base64_to_file(
    encoding: str,
    file_path: str | None = None,
    dir: str | Path | None = None,
    prefix: str | None = None,
):
    if dir is not None:
        os.makedirs(dir, exist_ok=True)
    data, extension = decode_base64_to_binary(encoding)
    if file_path is not None and prefix is None:
        filename = Path(file_path).name
        prefix = filename
        if "." in filename:
            prefix = filename[0 : filename.index(".")]
            extension = filename[filename.index(".") + 1 :]

    if prefix is not None:
        prefix = strip_invalid_filename_characters(prefix)

    if extension is None:
        file_obj = tempfile.NamedTemporaryFile(delete=False, prefix=prefix, dir=dir)
    else:
        file_obj = tempfile.NamedTemporaryFile(
            delete=False,
            prefix=prefix,
            suffix="." + extension,
            dir=dir,
        )
    file_obj.write(data)
    file_obj.flush()
    return file_obj


def dict_or_str_to_json_file(jsn: str | dict | list, dir: str | Path | None = None):
    if dir is not None:
        os.makedirs(dir, exist_ok=True)

    file_obj = tempfile.NamedTemporaryFile(
        delete=False, suffix=".json", dir=dir, mode="w+"
    )
    if isinstance(jsn, str):
        jsn = json.loads(jsn)
    json.dump(jsn, file_obj)
    file_obj.flush()
    return file_obj


def file_to_json(file_path: str | Path) -> dict | list:
    with open(file_path) as f:
        return json.load(f)


###########################
# HuggingFace Hub API Utils
###########################
def set_space_timeout(
    space_id: str,
    hf_token: str | None = None,
    timeout_in_seconds: int = 300,
):
    headers = huggingface_hub.utils.build_hf_headers(
        token=hf_token,
        library_name="gradio_client",
        library_version=__version__,
    )
    r = requests.post(
        f"https://huggingface.co/api/spaces/{space_id}/sleeptime",
        json={"seconds": timeout_in_seconds},
        headers=headers,
    )
    try:
        huggingface_hub.utils.hf_raise_for_status(r)
    except huggingface_hub.utils.HfHubHTTPError as err:
        raise SpaceDuplicationError(
            f"Could not set sleep timeout on duplicated Space. Please visit {SPACE_URL.format(space_id)} "
            "to set a timeout manually to reduce billing charges."
        ) from err


########################
# Misc utils
########################


def synchronize_async(func: Callable, *args, **kwargs) -> Any:
    """
    Runs async functions in sync scopes. Can be used in any scope.

    Example:
        if inspect.iscoroutinefunction(block_fn.fn):
            predictions = utils.synchronize_async(block_fn.fn, *processed_input)

    Args:
        func:
        *args:
        **kwargs:
    """
    return fsspec.asyn.sync(fsspec.asyn.get_loop(), func, *args, **kwargs)  # type: ignore


class APIInfoParseError(ValueError):
    pass


def get_type(schema: dict):
    if "type" in schema:
        return schema["type"]
    elif schema.get("oneOf"):
        return "oneOf"
    elif schema.get("anyOf"):
        return "anyOf"
    else:
        raise APIInfoParseError(f"Cannot parse type for {schema}")


def json_schema_to_python_type(schema: Any) -> str:
    """Convert the json schema into a python type hint"""
    type_ = get_type(schema)
    if type_ == {}:
        if "json" in schema["description"]:
            return "Dict[Any, Any]"
        else:
            return "Any"
    elif type_ == "null":
        return "None"
    elif type_ == "integer":
        return "int"
    elif type_ == "string":
        return "str"
    elif type_ == "boolean":
        return "bool"
    elif type_ == "number":
        return "int | float"
    elif type_ == "array":
        items = schema.get("items")
        if "prefixItems" in items:
            elements = ", ".join(
                [json_schema_to_python_type(i) for i in items["prefixItems"]]
            )
            return f"Tuple[{elements}]"
        else:
            elements = json_schema_to_python_type(items)
            return f"List[{elements}]"
    elif type_ == "object":
        des = ", ".join(
            [
                f"{n}: {json_schema_to_python_type(v)} ({v.get('description')})"
                for n, v in schema["properties"].items()
            ]
        )
        return f"Dict({des})"
    elif type_ in ["oneOf", "anyOf"]:
        desc = " | ".join([json_schema_to_python_type(i) for i in schema[type_]])
        return desc
    else:
        raise APIInfoParseError(f"Cannot parse schema {schema}")
