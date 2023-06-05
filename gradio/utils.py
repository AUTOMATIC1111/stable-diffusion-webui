""" Handy utility functions. """

from __future__ import annotations

import asyncio
import copy
import functools
import inspect
import json
import json.decoder
import os
import pkgutil
import random
import re
import sys
import time
import typing
import warnings
from contextlib import contextmanager
from enum import Enum
from io import BytesIO
from numbers import Number
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generator,
    TypeVar,
    Union,
)

import anyio
import httpx
import matplotlib
import requests
from gradio_client.serializing import Serializable
from markdown_it import MarkdownIt
from mdit_py_plugins.dollarmath.index import dollarmath_plugin
from mdit_py_plugins.footnote.index import footnote_plugin
from pydantic import BaseModel, parse_obj_as

import gradio
from gradio.context import Context
from gradio.strings import en

if TYPE_CHECKING:  # Only import for type checking (is False at runtime).
    from gradio.blocks import Block, BlockContext
    from gradio.components import Component

JSON_PATH = os.path.join(os.path.dirname(gradio.__file__), "launches.json")
GRADIO_VERSION = (
    (pkgutil.get_data(__name__, "version.txt") or b"").decode("ascii").strip()
)

T = TypeVar("T")


def colab_check() -> bool:
    """
    Check if interface is launching from Google Colab
    :return is_colab (bool): True or False
    """
    is_colab = False
    try:  # Check if running interactively using ipython.
        from IPython import get_ipython

        from_ipynb = get_ipython()
        if "google.colab" in str(from_ipynb):
            is_colab = True
    except (ImportError, NameError):
        pass
    return is_colab


def kaggle_check() -> bool:
    return bool(
        os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("GFOOTBALL_DATA_DIR")
    )


def sagemaker_check() -> bool:
    try:
        import boto3  # type: ignore

        client = boto3.client("sts")
        response = client.get_caller_identity()
        return "sagemaker" in response["Arn"].lower()
    except Exception:
        return False


def ipython_check() -> bool:
    """
    Check if interface is launching from iPython (not colab)
    :return is_ipython (bool): True or False
    """
    is_ipython = False
    try:  # Check if running interactively using ipython.
        from IPython import get_ipython

        if get_ipython() is not None:
            is_ipython = True
    except (ImportError, NameError):
        pass
    return is_ipython


def readme_to_html(article: str) -> str:
    try:
        response = requests.get(article, timeout=3)
        if response.status_code == requests.codes.ok:  # pylint: disable=no-member
            article = response.text
    except requests.exceptions.RequestException:
        pass
    return article


def show_tip(interface: gradio.Blocks) -> None:
    if interface.show_tips and random.random() < 1.5:
        tip: str = random.choice(en["TIPS"])
        print(f"Tip: {tip}")


def launch_counter() -> None:
    try:
        if not os.path.exists(JSON_PATH):
            launches = {"launches": 1}
            with open(JSON_PATH, "w+") as j:
                json.dump(launches, j)
        else:
            with open(JSON_PATH) as j:
                launches = json.load(j)
            launches["launches"] += 1
            if launches["launches"] in [25, 50, 150, 500, 1000]:
                print(en["BETA_INVITE"])
            with open(JSON_PATH, "w") as j:
                j.write(json.dumps(launches))
    except Exception:
        pass


def get_default_args(func: Callable) -> list[Any]:
    signature = inspect.signature(func)
    return [
        v.default if v.default is not inspect.Parameter.empty else None
        for v in signature.parameters.values()
    ]


def assert_configs_are_equivalent_besides_ids(
    config1: dict, config2: dict, root_keys: tuple = ("mode",)
):
    """Allows you to test if two different Blocks configs produce the same demo.

    Parameters:
    config1 (dict): nested dict with config from the first Blocks instance
    config2 (dict): nested dict with config from the second Blocks instance
    root_keys (Tuple): an interable consisting of which keys to test for equivalence at
        the root level of the config. By default, only "mode" is tested,
        so keys like "version" are ignored.
    """
    config1 = copy.deepcopy(config1)
    config2 = copy.deepcopy(config2)

    for key in root_keys:
        assert config1[key] == config2[key], f"Configs have different: {key}"

    assert len(config1["components"]) == len(
        config2["components"]
    ), "# of components are different"

    def assert_same_components(config1_id, config2_id):
        c1 = list(filter(lambda c: c["id"] == config1_id, config1["components"]))[0]
        c2 = list(filter(lambda c: c["id"] == config2_id, config2["components"]))[0]
        c1 = copy.deepcopy(c1)
        c1.pop("id")
        c2 = copy.deepcopy(c2)
        c2.pop("id")
        assert c1 == c2, f"{c1} does not match {c2}"

    def same_children_recursive(children1, chidren2):
        for child1, child2 in zip(children1, chidren2):
            assert_same_components(child1["id"], child2["id"])
            if "children" in child1 or "children" in child2:
                same_children_recursive(child1["children"], child2["children"])

    children1 = config1["layout"]["children"]
    children2 = config2["layout"]["children"]
    same_children_recursive(children1, children2)

    for d1, d2 in zip(config1["dependencies"], config2["dependencies"]):
        for t1, t2 in zip(d1.pop("targets"), d2.pop("targets")):
            assert_same_components(t1, t2)
        for i1, i2 in zip(d1.pop("inputs"), d2.pop("inputs")):
            assert_same_components(i1, i2)
        for o1, o2 in zip(d1.pop("outputs"), d2.pop("outputs")):
            assert_same_components(o1, o2)

        assert d1 == d2, f"{d1} does not match {d2}"

    return True


def format_ner_list(input_string: str, ner_groups: list[dict[str, str | int]]):
    if len(ner_groups) == 0:
        return [(input_string, None)]

    output = []
    end = 0
    prev_end = 0

    for group in ner_groups:
        entity, start, end = group["entity_group"], group["start"], group["end"]
        output.append((input_string[prev_end:start], None))
        output.append((input_string[start:end], entity))
        prev_end = end

    output.append((input_string[end:], None))
    return output


def delete_none(_dict: dict, skip_value: bool = False) -> dict:
    """
    Delete keys whose values are None from a dictionary
    """
    for key, value in list(_dict.items()):
        if skip_value and key == "value":
            continue
        elif value is None:
            del _dict[key]
    return _dict


def resolve_singleton(_list: list[Any] | Any) -> Any:
    if len(_list) == 1:
        return _list[0]
    else:
        return _list


def component_or_layout_class(cls_name: str) -> type[Component] | type[BlockContext]:
    """
    Returns the component, template, or layout class with the given class name, or
    raises a ValueError if not found.

    Parameters:
    cls_name (str): lower-case string class name of a component
    Returns:
    cls: the component class
    """
    import gradio.blocks
    import gradio.components
    import gradio.layouts
    import gradio.templates

    components = [
        (name, cls)
        for name, cls in gradio.components.__dict__.items()
        if isinstance(cls, type)
    ]
    templates = [
        (name, cls)
        for name, cls in gradio.templates.__dict__.items()
        if isinstance(cls, type)
    ]
    layouts = [
        (name, cls)
        for name, cls in gradio.layouts.__dict__.items()
        if isinstance(cls, type)
    ]
    for name, cls in components + templates + layouts:
        if name.lower() == cls_name.replace("_", "") and (
            issubclass(cls, gradio.components.Component)
            or issubclass(cls, gradio.blocks.BlockContext)
        ):
            return cls
    raise ValueError(f"No such component or layout: {cls_name}")


def run_coro_in_background(func: Callable, *args, **kwargs):
    """
    Runs coroutines in background.

    Warning, be careful to not use this function in other than FastAPI scope, because the event_loop has not started yet.
    You can use it in any scope reached by FastAPI app.

    correct scope examples: endpoints in routes, Blocks.process_api
    incorrect scope examples: Blocks.launch

    Use startup_events in routes.py if you need to run a coro in background in Blocks.launch().


    Example:
        utils.run_coro_in_background(fn, *args, **kwargs)

    Args:
        func:
        *args:
        **kwargs:

    Returns:

    """
    event_loop = asyncio.get_event_loop()
    return event_loop.create_task(func(*args, **kwargs))


def run_sync_iterator_async(iterator):
    """Helper for yielding StopAsyncIteration from sync iterators."""
    try:
        return next(iterator)
    except StopIteration:
        # raise a ValueError here because co-routines can't raise StopIteration themselves
        raise StopAsyncIteration() from None


class SyncToAsyncIterator:
    """Treat a synchronous iterator as async one."""

    def __init__(self, iterator, limiter) -> None:
        self.iterator = iterator
        self.limiter = limiter

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await anyio.to_thread.run_sync(
            run_sync_iterator_async, self.iterator, limiter=self.limiter
        )


async def async_iteration(iterator):
    # anext not introduced until 3.10 :(
    return await iterator.__anext__()


class AsyncRequest:
    """
    The AsyncRequest class is a low-level API that allow you to create asynchronous HTTP requests without a context manager.
    Compared to making calls by using httpx directly, AsyncRequest offers several advantages:
        (1) Includes response validation functionality both using validation models and functions.
        (2) Exceptions are handled silently during the request call, which provides the ability to inspect each one
        request call individually in the case where there are multiple asynchronous request calls and some of them fail.
        (3) Provides HTTP request types with AsyncRequest.Method Enum class for ease of usage

    AsyncRequest also offers some util functions such as has_exception, is_valid and status to inspect get detailed
    information about executed request call.

    The basic usage of AsyncRequest is as follows: create a AsyncRequest object with inputs(method, url etc.). Then use it
    with the "await" statement, and then you can use util functions to do some post request checks depending on your use-case.
    Finally, call the get_validated_data function to get the response data.

    You can see example usages in test_utils.py.
    """

    client = httpx.AsyncClient()

    class Method(str, Enum):
        """
        Method is an enumeration class that contains possible types of HTTP request methods.
        """

        ANY = "*"
        CONNECT = "CONNECT"
        HEAD = "HEAD"
        GET = "GET"
        DELETE = "DELETE"
        OPTIONS = "OPTIONS"
        PATCH = "PATCH"
        POST = "POST"
        PUT = "PUT"
        TRACE = "TRACE"

    def __init__(
        self,
        method: Method,
        url: str,
        *,
        validation_model: type[BaseModel] | None = None,
        validation_function: Union[Callable, None] = None,
        exception_type: type[Exception] = Exception,
        raise_for_status: bool = False,
        client: httpx.AsyncClient | None = None,
        **kwargs,
    ):
        """
        Initialize the Request instance.
        Args:
            method(Request.Method) : method of the request
            url(str): url of the request
            *
            validation_model(Type[BaseModel]): a pydantic validation class type to use in validation of the response
            validation_function(Callable): a callable instance to use in validation of the response
            exception_class(Type[Exception]): a exception type to throw with its type
            raise_for_status(bool): a flag that determines to raise httpx.Request.raise_for_status() exceptions.
        """
        self._exception: Union[Exception, None] = None
        self._status = None
        self._raise_for_status = raise_for_status
        self._validation_model = validation_model
        self._validation_function = validation_function
        self._exception_type = exception_type
        self._validated_data = None
        # Create request
        self._request = self._create_request(method, url, **kwargs)
        self.client_ = client or self.client

    def __await__(self) -> Generator[None, Any, AsyncRequest]:
        """
        Wrap Request's __await__ magic function to create request calls which are executed in one line.
        """
        return self.__run().__await__()

    async def __run(self) -> AsyncRequest:
        """
        Manage the request call lifecycle.
        Execute the request by sending it through the client, then check its status.
        Then parse the request into Json format. And then validate it using the provided validation methods.
        If a problem occurs in this sequential process,
        an exception will be raised within the corresponding method, and allowed to be examined.
        Manage the request call lifecycle.

        Returns:
            Request
        """
        try:
            # Send the request and get the response.
            self._response: httpx.Response = await self.client_.send(self._request)
            # Raise for _status
            self._status = self._response.status_code
            if self._raise_for_status:
                self._response.raise_for_status()
            # Parse client response data to JSON
            self._json_response_data = self._response.json()
            # Validate response data
            self._validated_data = self._validate_response_data(
                self._json_response_data
            )
        except Exception as exception:
            # If there is an exception, store it to do further inspections.
            self._exception = self._exception_type(exception)
        return self

    @staticmethod
    def _create_request(method: Method, url: str, **kwargs) -> httpx.Request:
        """
        Create a request. This is a httpx request wrapper function.
        Args:
            method(Request.Method): request method type
            url(str): target url of the request
            **kwargs
        Returns:
            Request
        """
        request = httpx.Request(method, url, **kwargs)
        return request

    def _validate_response_data(self, response):
        """
        Validate response using given validation methods. If there is a validation method and response is not valid,
        validation functions will raise an exception for them.
        Args:
            response(ResponseJson): response object
        Returns:
            ResponseJson: Validated Json object.
        """

        # We use raw response as a default value if there is no validation method or response is not valid.
        validated_response = response

        try:
            # If a validation model is provided, validate response using the validation model.
            if self._validation_model:
                validated_response = self._validate_response_by_model(response)
            # Then, If a validation function is provided, validate response using the validation function.
            if self._validation_function:
                validated_response = self._validate_response_by_validation_function(
                    response
                )
        except Exception as exception:
            # If one of the validation methods does not confirm, raised exception will be silently handled.
            # We assign this exception to classes instance to do further inspections via is_valid function.
            self._exception = exception

        return validated_response

    def _validate_response_by_model(self, response) -> BaseModel:
        """
        Validate response json using the validation model.
        Args:
            response(ResponseJson): response object
        Returns:
            ResponseJson: Validated Json object.
        """
        validated_data = BaseModel()
        if self._validation_model:
            validated_data = parse_obj_as(self._validation_model, response)
        return validated_data

    def _validate_response_by_validation_function(self, response):
        """
        Validate response json using the validation function.
        Args:
            response(ResponseJson): response object
        Returns:
            ResponseJson: Validated Json object.
        """
        validated_data = None

        if self._validation_function:
            validated_data = self._validation_function(response)

        return validated_data

    def is_valid(self, raise_exceptions: bool = False) -> bool:
        """
        Check response object's validity+. Raise exceptions if raise_exceptions flag is True.
        Args:
            raise_exceptions(bool) : a flag to raise exceptions in this check
        Returns:
            bool: validity of the data
        """
        if self.has_exception and self._exception:
            if raise_exceptions:
                raise self._exception
            return False
        else:
            # If there is no exception, that means there is no validation error.
            return True

    def get_validated_data(self):
        return self._validated_data

    @property
    def json(self):
        return self._json_response_data

    @property
    def exception(self):
        return self._exception

    @property
    def has_exception(self):
        return self.exception is not None

    @property
    def raise_exceptions(self):
        if self.has_exception and self._exception:
            raise self._exception

    @property
    def status(self):
        return self._status


@contextmanager
def set_directory(path: Path | str):
    """Context manager that sets the working directory to the given path."""
    origin = Path().absolute()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(origin)


def sanitize_value_for_csv(value: str | Number) -> str | Number:
    """
    Sanitizes a value that is being written to a CSV file to prevent CSV injection attacks.
    Reference: https://owasp.org/www-community/attacks/CSV_Injection
    """
    if isinstance(value, Number):
        return value
    unsafe_prefixes = ["=", "+", "-", "@", "\t", "\n"]
    unsafe_sequences = [",=", ",+", ",-", ",@", ",\t", ",\n"]
    if any(value.startswith(prefix) for prefix in unsafe_prefixes) or any(
        sequence in value for sequence in unsafe_sequences
    ):
        value = f"'{value}"
    return value


def sanitize_list_for_csv(values: list[Any]) -> list[Any]:
    """
    Sanitizes a list of values (or a list of list of values) that is being written to a
    CSV file to prevent CSV injection attacks.
    """
    sanitized_values = []
    for value in values:
        if isinstance(value, list):
            sanitized_value = [sanitize_value_for_csv(v) for v in value]
            sanitized_values.append(sanitized_value)
        else:
            sanitized_value = sanitize_value_for_csv(value)
            sanitized_values.append(sanitized_value)
    return sanitized_values


def append_unique_suffix(name: str, list_of_names: list[str]):
    """Appends a numerical suffix to `name` so that it does not appear in `list_of_names`."""
    set_of_names: set[str] = set(list_of_names)  # for O(1) lookup
    if name not in set_of_names:
        return name
    else:
        suffix_counter = 1
        new_name = f"{name}_{suffix_counter}"
        while new_name in set_of_names:
            suffix_counter += 1
            new_name = f"{name}_{suffix_counter}"
        return new_name


def validate_url(possible_url: str) -> bool:
    headers = {"User-Agent": "gradio (https://gradio.app/; team@gradio.app)"}
    try:
        head_request = requests.head(possible_url, headers=headers)
        # some URLs, such as AWS S3 presigned URLs, return a 405 or a 403 for HEAD requests
        if head_request.status_code == 405 or head_request.status_code == 403:
            return requests.get(possible_url, headers=headers).ok
        return head_request.ok
    except Exception:
        return False


def is_update(val):
    return isinstance(val, dict) and "update" in val.get("__type__", "")


def get_continuous_fn(fn: Callable, every: float) -> Callable:
    def continuous_fn(*args):
        while True:
            output = fn(*args)
            yield output
            time.sleep(every)

    return continuous_fn


async def cancel_tasks(task_ids: set[str]):
    if sys.version_info < (3, 8):
        return None

    matching_tasks = [
        task for task in asyncio.all_tasks() if task.get_name() in task_ids
    ]
    for task in matching_tasks:
        task.cancel()
    await asyncio.gather(*matching_tasks, return_exceptions=True)


def set_task_name(task, session_hash: str, fn_index: int, batch: bool):
    if sys.version_info >= (3, 8) and not (
        batch
    ):  # You shouldn't be able to cancel a task if it's part of a batch
        task.set_name(f"{session_hash}_{fn_index}")


def get_cancel_function(
    dependencies: list[dict[str, Any]]
) -> tuple[Callable, list[int]]:
    fn_to_comp = {}
    for dep in dependencies:
        if Context.root_block:
            fn_index = next(
                i for i, d in enumerate(Context.root_block.dependencies) if d == dep
            )
            fn_to_comp[fn_index] = [
                Context.root_block.blocks[o] for o in dep["outputs"]
            ]

    async def cancel(session_hash: str) -> None:
        task_ids = {f"{session_hash}_{fn}" for fn in fn_to_comp}
        await cancel_tasks(task_ids)

    return (
        cancel,
        list(fn_to_comp.keys()),
    )


def get_type_hints(fn):
    # Importing gradio with the canonical abbreviation. Used in typing._eval_type.
    import gradio as gr  # noqa: F401
    from gradio import Request  # noqa: F401

    if inspect.isfunction(fn) or inspect.ismethod(fn):
        pass
    elif callable(fn):
        fn = fn.__call__
    else:
        return {}

    try:
        return typing.get_type_hints(fn)
    except TypeError:
        # On Python 3.9 or earlier, get_type_hints throws a TypeError if the function
        # has a type annotation that include "|". We resort to parsing the signature
        # manually using inspect.signature.
        type_hints = {}
        sig = inspect.signature(fn)
        for name, param in sig.parameters.items():
            if param.annotation is inspect.Parameter.empty:
                continue
            if "|" in str(param.annotation):
                continue
            # To convert the string annotation to a class, we use the
            # internal typing._eval_type function. This is not ideal, but
            # it's the only way to do it without eval-ing the string.
            # Since the API is internal, it may change in the future.
            try:
                type_hints[name] = typing._eval_type(  # type: ignore
                    typing.ForwardRef(param.annotation), globals(), locals()
                )
            except (NameError, TypeError):
                pass
        return type_hints


def is_special_typed_parameter(name, parameter_types):
    from gradio.helpers import EventData
    from gradio.routes import Request

    """Checks if parameter has a type hint designating it as a gr.Request or gr.EventData"""
    hint = parameter_types.get(name)
    if not hint:
        return False
    is_request = hint == Request
    is_event_data = inspect.isclass(hint) and issubclass(hint, EventData)
    return is_request or is_event_data


def check_function_inputs_match(fn: Callable, inputs: list, inputs_as_dict: bool):
    """
    Checks if the input component set matches the function
    Returns: None if valid, a string error message if mismatch
    """

    signature = inspect.signature(fn)
    parameter_types = get_type_hints(fn)
    min_args = 0
    max_args = 0
    infinity = -1
    for name, param in signature.parameters.items():
        has_default = param.default != param.empty
        if param.kind in [param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD]:
            if not is_special_typed_parameter(name, parameter_types):
                if not has_default:
                    min_args += 1
                max_args += 1
        elif param.kind == param.VAR_POSITIONAL:
            max_args = infinity
        elif param.kind == param.KEYWORD_ONLY and not has_default:
            return f"Keyword-only args must have default values for function {fn}"
    arg_count = 1 if inputs_as_dict else len(inputs)
    if min_args == max_args and max_args != arg_count:
        warnings.warn(
            f"Expected {max_args} arguments for function {fn}, received {arg_count}."
        )
    if arg_count < min_args:
        warnings.warn(
            f"Expected at least {min_args} arguments for function {fn}, received {arg_count}."
        )
    if max_args != infinity and arg_count > max_args:
        warnings.warn(
            f"Expected maximum {max_args} arguments for function {fn}, received {arg_count}."
        )


class TupleNoPrint(tuple):
    # To remove printing function return in notebook
    def __repr__(self):
        return ""

    def __str__(self):
        return ""


class MatplotlibBackendMananger:
    def __enter__(self):
        self._original_backend = matplotlib.get_backend()
        matplotlib.use("agg")

    def __exit__(self, exc_type, exc_val, exc_tb):
        matplotlib.use(self._original_backend)


def tex2svg(formula, *args):
    with MatplotlibBackendMananger():
        import matplotlib.pyplot as plt

        fontsize = 20
        dpi = 300
        plt.rc("mathtext", fontset="cm")
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, rf"${formula}$", fontsize=fontsize)
        output = BytesIO()
        fig.savefig(
            output,
            dpi=dpi,
            transparent=True,
            format="svg",
            bbox_inches="tight",
            pad_inches=0.0,
        )
        plt.close(fig)
        output.seek(0)
        xml_code = output.read().decode("utf-8")
        svg_start = xml_code.index("<svg ")
        svg_code = xml_code[svg_start:]
        svg_code = re.sub(r"<metadata>.*<\/metadata>", "", svg_code, flags=re.DOTALL)
        svg_code = re.sub(r' width="[^"]+"', "", svg_code)
        height_match = re.search(r'height="([\d.]+)pt"', svg_code)
        if height_match:
            height = float(height_match.group(1))
            new_height = height / fontsize  # conversion from pt to em
            svg_code = re.sub(
                r'height="[\d.]+pt"', f'height="{new_height}em"', svg_code
            )
        copy_code = f"<span style='font-size: 0px'>{formula}</span>"
    return f"{copy_code}{svg_code}"


def abspath(path: str | Path) -> Path:
    """Returns absolute path of a str or Path path, but does not resolve symlinks."""
    path = Path(path)

    if path.is_absolute():
        return path

    # recursively check if there is a symlink within the path
    is_symlink = path.is_symlink() or any(
        parent.is_symlink() for parent in path.parents
    )

    if is_symlink or path == path.resolve():  # in case path couldn't be resolved
        return Path.cwd() / path
    else:
        return path.resolve()


def is_in_or_equal(path_1: str | Path, path_2: str | Path):
    """
    True if path_1 is a descendant (i.e. located within) path_2 or if the paths are the
    same, returns False otherwise.
    Parameters:
        path_1: str or Path (should be a file)
        path_2: str or Path (can be a file or directory)
    """
    path_1, path_2 = abspath(path_1), abspath(path_2)
    try:
        if str(path_1.relative_to(path_2)).startswith(".."):  # prevent path traversal
            return False
    except ValueError:
        return False
    return True


def get_serializer_name(block: Block) -> str | None:
    if not hasattr(block, "serialize"):
        return None

    def get_class_that_defined_method(meth: Callable):
        # Adapted from: https://stackoverflow.com/a/25959545/5209347
        if isinstance(meth, functools.partial):
            return get_class_that_defined_method(meth.func)
        if inspect.ismethod(meth) or (
            inspect.isbuiltin(meth)
            and getattr(meth, "__self__", None) is not None
            and getattr(meth.__self__, "__class__", None)
        ):
            for cls in inspect.getmro(meth.__self__.__class__):
                # Find the first serializer defined in gradio_client that
                if issubclass(cls, Serializable) and "gradio_client" in cls.__module__:
                    return cls
                if meth.__name__ in cls.__dict__:
                    return cls
            meth = getattr(meth, "__func__", meth)  # fallback to __qualname__ parsing
        if inspect.isfunction(meth):
            cls = getattr(
                inspect.getmodule(meth),
                meth.__qualname__.split(".<locals>", 1)[0].rsplit(".", 1)[0],
                None,
            )
            if isinstance(cls, type):
                return cls
        return getattr(meth, "__objclass__", None)

    cls = get_class_that_defined_method(block.serialize)  # type: ignore
    if cls:
        return cls.__name__


def get_markdown_parser() -> MarkdownIt:
    md = (
        MarkdownIt(
            "js-default",
            {
                "linkify": True,
                "typographer": True,
                "html": True,
            },
        )
        .use(dollarmath_plugin, renderer=tex2svg, allow_digits=False)
        .use(footnote_plugin)
        .enable("table")
    )

    # Add target="_blank" to all links. Taken from MarkdownIt docs: https://github.com/executablebooks/markdown-it-py/blob/master/docs/architecture.md
    def render_blank_link(self, tokens, idx, options, env):
        tokens[idx].attrSet("target", "_blank")
        return self.renderToken(tokens, idx, options, env)

    md.add_render_rule("link_open", render_blank_link)

    return md


HTML_TAG_RE = re.compile("<.*?>")


def remove_html_tags(raw_html: str | None) -> str:
    return re.sub(HTML_TAG_RE, "", raw_html or "")
