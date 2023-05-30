"""The main Client class for the Python client."""
from __future__ import annotations

import concurrent.futures
import json
import re
import threading
import time
import urllib.parse
import uuid
import warnings
from concurrent.futures import Future, TimeoutError
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Callable

import huggingface_hub
import requests
import websockets
from huggingface_hub import SpaceStage
from huggingface_hub.utils import (
    RepositoryNotFoundError,
    build_hf_headers,
    send_telemetry,
)
from packaging import version
from typing_extensions import Literal

from gradio_client import serializing, utils
from gradio_client.documentation import document, set_documentation_group
from gradio_client.serializing import Serializable
from gradio_client.utils import (
    Communicator,
    JobStatus,
    Status,
    StatusUpdate,
)

set_documentation_group("py-client")


@document("predict", "submit", "view_api", "duplicate")
class Client:
    """
    The main Client class for the Python client. This class is used to connect to a remote Gradio app and call its API endpoints.

    Example:
        from gradio_client import Client

        client = Client("abidlabs/whisper-large-v2")  # connecting to a Hugging Face Space
        client.predict("test.mp4", api_name="/predict")
        >> What a nice recording! # returns the result of the remote API call

        client = Client("https://bec81a83-5b5c-471e.gradio.live")  # connecting to a temporary Gradio share URL
        job = client.submit("hello", api_name="/predict")  # runs the prediction in a background thread
        job.result()
        >> 49 # returns the result of the remote API call (blocking call)
    """

    def __init__(
        self,
        src: str,
        hf_token: str | None = None,
        max_workers: int = 40,
        serialize: bool = True,
        verbose: bool = True,
    ):
        """
        Parameters:
            src: Either the name of the Hugging Face Space to load, (e.g. "abidlabs/whisper-large-v2") or the full URL (including "http" or "https") of the hosted Gradio app to load (e.g. "http://mydomain.com/app" or "https://bec81a83-5b5c-471e.gradio.live/").
            hf_token: The Hugging Face token to use to access private Spaces. Automatically fetched if you are logged in via the Hugging Face Hub CLI. Obtain from: https://huggingface.co/settings/token
            max_workers: The maximum number of thread workers that can be used to make requests to the remote Gradio app simultaneously.
            serialize: Whether the client should serialize the inputs and deserialize the outputs of the remote API. If set to False, the client will pass the inputs and outputs as-is, without serializing/deserializing them. E.g. you if you set this to False, you'd submit an image in base64 format instead of a filepath, and you'd get back an image in base64 format from the remote API instead of a filepath.
            verbose: Whether the client should print statements to the console.
        """
        self.verbose = verbose
        self.hf_token = hf_token
        self.serialize = serialize
        self.headers = build_hf_headers(
            token=hf_token,
            library_name="gradio_client",
            library_version=utils.__version__,
        )
        self.space_id = None

        if src.startswith("http://") or src.startswith("https://"):
            _src = src
        else:
            _src = self._space_name_to_src(src)
            if _src is None:
                raise ValueError(
                    f"Could not find Space: {src}. If it is a private Space, please provide an hf_token."
                )
            self.space_id = src
        self.src = _src
        state = self._get_space_state()
        if state == SpaceStage.BUILDING:
            if self.verbose:
                print("Space is still building. Please wait...")
            while self._get_space_state() == SpaceStage.BUILDING:
                time.sleep(2)  # so we don't get rate limited by the API
                pass
        if state in utils.INVALID_RUNTIME:
            raise ValueError(
                f"The current space is in the invalid state: {state}. "
                "Please contact the owner to fix this."
            )
        if self.verbose:
            print(f"Loaded as API: {self.src} ✔")

        self.api_url = urllib.parse.urljoin(self.src, utils.API_URL)
        self.ws_url = urllib.parse.urljoin(
            self.src.replace("http", "ws", 1), utils.WS_URL
        )
        self.upload_url = urllib.parse.urljoin(self.src, utils.UPLOAD_URL)
        self.reset_url = urllib.parse.urljoin(self.src, utils.RESET_URL)
        self.config = self._get_config()
        self.session_hash = str(uuid.uuid4())

        self.endpoints = [
            Endpoint(self, fn_index, dependency)
            for fn_index, dependency in enumerate(self.config["dependencies"])
        ]

        # Create a pool of threads to handle the requests
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

        # Disable telemetry by setting the env variable HF_HUB_DISABLE_TELEMETRY=1
        threading.Thread(target=self._telemetry_thread).start()

    @classmethod
    def duplicate(
        cls,
        from_id: str,
        to_id: str | None = None,
        hf_token: str | None = None,
        private: bool = True,
        hardware: str | None = None,
        secrets: dict[str, str] | None = None,
        sleep_timeout: int = 5,
        max_workers: int = 40,
        verbose: bool = True,
    ):
        """
        Duplicates a Hugging Face Space under your account and returns a Client object
        for the new Space. No duplication is created if the Space already exists in your
        account (to override this, provide a new name for the new Space using `to_id`).
        To use this method, you must provide an `hf_token` or be logged in via the Hugging
        Face Hub CLI.

        The new Space will be private by default and use the same hardware as the original
        Space. This can be changed by using the `private` and `hardware` parameters. For
        hardware upgrades (beyond the basic CPU tier), you may be required to provide
        billing information on Hugging Face: https://huggingface.co/settings/billing

        Parameters:
            from_id: The name of the Hugging Face Space to duplicate in the format "{username}/{space_id}", e.g. "gradio/whisper".
            to_id: The name of the new Hugging Face Space to create, e.g. "abidlabs/whisper-duplicate". If not provided, the new Space will be named "{your_HF_username}/{space_id}".
            hf_token: The Hugging Face token to use to access private Spaces. Automatically fetched if you are logged in via the Hugging Face Hub CLI. Obtain from: https://huggingface.co/settings/token
            private: Whether the new Space should be private (True) or public (False). Defaults to True.
            hardware: The hardware tier to use for the new Space. Defaults to the same hardware tier as the original Space. Options include "cpu-basic", "cpu-upgrade", "t4-small", "t4-medium", "a10g-small", "a10g-large", "a100-large", subject to availability.
            secrets: A dictionary of (secret key, secret value) to pass to the new Space. Defaults to None. Secrets are only used when the Space is duplicated for the first time, and are not updated if the duplicated Space already exists.
            sleep_timeout: The number of minutes after which the duplicate Space will be puased if no requests are made to it (to minimize billing charges). Defaults to 5 minutes.
            max_workers: The maximum number of thread workers that can be used to make requests to the remote Gradio app simultaneously.
            verbose: Whether the client should print statements to the console.
        Example:
            import os
            from gradio_client import Client
            HF_TOKEN = os.environ.get("HF_TOKEN")
            client = Client.duplicate("abidlabs/whisper", hf_token=HF_TOKEN)
            client.predict("audio_sample.wav")
            >> "This is a test of the whisper speech recognition model."
        """
        try:
            original_info = huggingface_hub.get_space_runtime(from_id, token=hf_token)
        except RepositoryNotFoundError as rnfe:
            raise ValueError(
                f"Could not find Space: {from_id}. If it is a private Space, please provide an `hf_token`."
            ) from rnfe
        if to_id:
            if "/" in to_id:
                to_id = to_id.split("/")[1]
            space_id = huggingface_hub.get_full_repo_name(to_id, token=hf_token)
        else:
            space_id = huggingface_hub.get_full_repo_name(
                from_id.split("/")[1], token=hf_token
            )
        try:
            huggingface_hub.get_space_runtime(space_id, token=hf_token)
            if verbose:
                print(
                    f"Using your existing Space: {utils.SPACE_URL.format(space_id)} 🤗"
                )
            if secrets is not None:
                warnings.warn(
                    "Secrets are only used when the Space is duplicated for the first time, and are not updated if the duplicated Space already exists."
                )
        except RepositoryNotFoundError:
            if verbose:
                print(f"Creating a duplicate of {from_id} for your own use... 🤗")
            huggingface_hub.duplicate_space(
                from_id=from_id,
                to_id=space_id,
                token=hf_token,
                exist_ok=True,
                private=private,
            )
            if secrets is not None:
                for key, value in secrets.items():
                    huggingface_hub.add_space_secret(
                        space_id, key, value, token=hf_token
                    )
            utils.set_space_timeout(
                space_id, hf_token=hf_token, timeout_in_seconds=sleep_timeout * 60
            )
            if verbose:
                print(f"Created new Space: {utils.SPACE_URL.format(space_id)}")
        current_info = huggingface_hub.get_space_runtime(space_id, token=hf_token)
        current_hardware = (
            current_info.hardware or huggingface_hub.SpaceHardware.CPU_BASIC
        )
        hardware = hardware or original_info.hardware
        if current_hardware != hardware:
            huggingface_hub.request_space_hardware(space_id, hardware)  # type: ignore
            print(
                f"-------\nNOTE: this Space uses upgraded hardware: {hardware}... see billing info at https://huggingface.co/settings/billing\n-------"
            )
        if verbose:
            print("")
        client = cls(
            space_id, hf_token=hf_token, max_workers=max_workers, verbose=verbose
        )
        return client

    def _get_space_state(self):
        if not self.space_id:
            return None
        info = huggingface_hub.get_space_runtime(self.space_id, token=self.hf_token)
        return info.stage

    def predict(
        self,
        *args,
        api_name: str | None = None,
        fn_index: int | None = None,
    ) -> Any:
        """
        Calls the Gradio API and returns the result (this is a blocking call).

        Parameters:
            args: The arguments to pass to the remote API. The order of the arguments must match the order of the inputs in the Gradio app.
            api_name: The name of the API endpoint to call starting with a leading slash, e.g. "/predict". Does not need to be provided if the Gradio app has only one named API endpoint.
            fn_index: As an alternative to api_name, this parameter takes the index of the API endpoint to call, e.g. 0. Both api_name and fn_index can be provided, but if they conflict, api_name will take precedence.
        Returns:
            The result of the API call. Will be a Tuple if the API has multiple outputs.
        Example:
            from gradio_client import Client
            client = Client(src="gradio/calculator")
            client.predict(5, "add", 4, api_name="/predict")
            >> 9.0
        """
        return self.submit(*args, api_name=api_name, fn_index=fn_index).result()

    def submit(
        self,
        *args,
        api_name: str | None = None,
        fn_index: int | None = None,
        result_callbacks: Callable | list[Callable] | None = None,
    ) -> Job:
        """
        Creates and returns a Job object which calls the Gradio API in a background thread. The job can be used to retrieve the status and result of the remote API call.

        Parameters:
            args: The arguments to pass to the remote API. The order of the arguments must match the order of the inputs in the Gradio app.
            api_name: The name of the API endpoint to call starting with a leading slash, e.g. "/predict". Does not need to be provided if the Gradio app has only one named API endpoint.
            fn_index: As an alternative to api_name, this parameter takes the index of the API endpoint to call, e.g. 0. Both api_name and fn_index can be provided, but if they conflict, api_name will take precedence.
            result_callbacks: A callback function, or list of callback functions, to be called when the result is ready. If a list of functions is provided, they will be called in order. The return values from the remote API are provided as separate parameters into the callback. If None, no callback will be called.
        Returns:
            A Job object that can be used to retrieve the status and result of the remote API call.
        Example:
            from gradio_client import Client
            client = Client(src="gradio/calculator")
            job = client.submit(5, "add", 4, api_name="/predict")
            job.status()
            >> <Status.STARTING: 'STARTING'>
            job.result()  # blocking call
            >> 9.0
        """
        inferred_fn_index = self._infer_fn_index(api_name, fn_index)

        helper = None
        if self.endpoints[inferred_fn_index].use_ws:
            helper = Communicator(
                Lock(),
                JobStatus(),
                self.endpoints[inferred_fn_index].process_predictions,
                self.reset_url,
            )
        end_to_end_fn = self.endpoints[inferred_fn_index].make_end_to_end_fn(helper)
        future = self.executor.submit(end_to_end_fn, *args)

        job = Job(
            future, communicator=helper, verbose=self.verbose, space_id=self.space_id
        )

        if result_callbacks:
            if isinstance(result_callbacks, Callable):
                result_callbacks = [result_callbacks]

            def create_fn(callback) -> Callable:
                def fn(future):
                    if isinstance(future.result(), tuple):
                        callback(*future.result())
                    else:
                        callback(future.result())

                return fn

            for callback in result_callbacks:
                job.add_done_callback(create_fn(callback))

        return job

    def view_api(
        self,
        all_endpoints: bool | None = None,
        print_info: bool = True,
        return_format: Literal["dict", "str"] | None = None,
    ) -> dict | str | None:
        """
        Prints the usage info for the API. If the Gradio app has multiple API endpoints, the usage info for each endpoint will be printed separately. If return_format="dict" the info is returned in dictionary format, as shown in the example below.

        Parameters:
            all_endpoints: If True, prints information for both named and unnamed endpoints in the Gradio app. If False, will only print info about named endpoints. If None (default), will only print info about unnamed endpoints if there are no named endpoints.
            print_info: If True, prints the usage info to the console. If False, does not print the usage info.
            return_format: If None, nothing is returned. If "str", returns the same string that would be printed to the console. If "dict", returns the usage info as a dictionary that can be programmatically parsed, and *all endpoints are returned in the dictionary* regardless of the value of `all_endpoints`. The format of the dictionary is in the docstring of this method.
        Example:
            from gradio_client import Client
            client = Client(src="gradio/calculator")
            client.view_api(return_format="dict")
            >> {
                'named_endpoints': {
                    '/predict': {
                        'parameters': [
                            {
                                'label': 'num1',
                                'type_python': 'int | float',
                                'type_description': 'numeric value',
                                'component': 'Number',
                                'example_input': '5'
                            },
                            {
                                'label': 'operation',
                                'type_python': 'str',
                                'type_description': 'string value',
                                'component': 'Radio',
                                'example_input': 'add'
                            },
                            {
                                'label': 'num2',
                                'type_python': 'int | float',
                                'type_description': 'numeric value',
                                'component': 'Number',
                                'example_input': '5'
                            },
                        ],
                        'returns': [
                            {
                                'label': 'output',
                                'type_python': 'int | float',
                                'type_description': 'numeric value',
                                'component': 'Number',
                            },
                        ]
                    },
                    '/flag': {
                        'parameters': [
                            ...
                            ],
                        'returns': [
                            ...
                            ]
                        }
                    }
                'unnamed_endpoints': {
                    2: {
                        'parameters': [
                            ...
                            ],
                        'returns': [
                            ...
                            ]
                        }
                    }
                }
            }

        """
        if self.serialize:
            api_info_url = urllib.parse.urljoin(self.src, utils.API_INFO_URL)
        else:
            api_info_url = urllib.parse.urljoin(self.src, utils.RAW_API_INFO_URL)

        # Versions of Gradio older than 3.29.0 returned format of the API info
        # from the /info endpoint
        if version.parse(self.config.get("version", "2.0")) > version.Version("3.29.0"):
            r = requests.get(api_info_url, headers=self.headers)
            if r.ok:
                info = r.json()
            else:
                raise ValueError(f"Could not fetch api info for {self.src}")
        else:
            fetch = requests.post(
                utils.SPACE_FETCHER_URL,
                json={"config": json.dumps(self.config), "serialize": self.serialize},
            )
            if fetch.ok:
                info = fetch.json()["api"]
            else:
                raise ValueError(f"Could not fetch api info for {self.src}")
        num_named_endpoints = len(info["named_endpoints"])
        num_unnamed_endpoints = len(info["unnamed_endpoints"])
        if num_named_endpoints == 0 and all_endpoints is None:
            all_endpoints = True

        human_info = "Client.predict() Usage Info\n---------------------------\n"
        human_info += f"Named API endpoints: {num_named_endpoints}\n"

        for api_name, endpoint_info in info["named_endpoints"].items():
            human_info += self._render_endpoints_info(api_name, endpoint_info)

        if all_endpoints:
            human_info += f"\nUnnamed API endpoints: {num_unnamed_endpoints}\n"
            for fn_index, endpoint_info in info["unnamed_endpoints"].items():
                # When loading from json, the fn_indices are read as strings
                # because json keys can only be strings
                human_info += self._render_endpoints_info(int(fn_index), endpoint_info)
        else:
            if num_unnamed_endpoints > 0:
                human_info += f"\nUnnamed API endpoints: {num_unnamed_endpoints}, to view, run Client.view_api(all_endpoints=True)\n"

        if print_info:
            print(human_info)
        if return_format == "str":
            return human_info
        elif return_format == "dict":
            return info

    def reset_session(self) -> None:
        self.session_hash = str(uuid.uuid4())

    def _render_endpoints_info(
        self,
        name_or_index: str | int,
        endpoints_info: dict[str, list[dict[str, Any]]],
    ) -> str:
        parameter_names = [p["label"] for p in endpoints_info["parameters"]]
        parameter_names = [utils.sanitize_parameter_names(p) for p in parameter_names]
        rendered_parameters = ", ".join(parameter_names)
        if rendered_parameters:
            rendered_parameters = rendered_parameters + ", "
        return_values = [p["label"] for p in endpoints_info["returns"]]
        return_values = [utils.sanitize_parameter_names(r) for r in return_values]
        rendered_return_values = ", ".join(return_values)
        if len(return_values) > 1:
            rendered_return_values = f"({rendered_return_values})"

        if isinstance(name_or_index, str):
            final_param = f'api_name="{name_or_index}"'
        elif isinstance(name_or_index, int):
            final_param = f"fn_index={name_or_index}"
        else:
            raise ValueError("name_or_index must be a string or integer")

        human_info = f"\n - predict({rendered_parameters}{final_param}) -> {rendered_return_values}\n"
        human_info += "    Parameters:\n"
        if endpoints_info["parameters"]:
            for info in endpoints_info["parameters"]:
                desc = (
                    f" ({info['python_type']['description']})"
                    if info["python_type"].get("description")
                    else ""
                )
                type_ = info["python_type"]["type"]
                human_info += f"     - [{info['component']}] {utils.sanitize_parameter_names(info['label'])}: {type_}{desc} \n"
        else:
            human_info += "     - None\n"
        human_info += "    Returns:\n"
        if endpoints_info["returns"]:
            for info in endpoints_info["returns"]:
                desc = (
                    f" ({info['python_type']['description']})"
                    if info["python_type"].get("description")
                    else ""
                )
                type_ = info["python_type"]["type"]
                human_info += f"     - [{info['component']}] {utils.sanitize_parameter_names(info['label'])}: {type_}{desc} \n"
        else:
            human_info += "     - None\n"

        return human_info

    def __repr__(self):
        return self.view_api(print_info=False, return_format="str")

    def __str__(self):
        return self.view_api(print_info=False, return_format="str")

    def _telemetry_thread(self) -> None:
        # Disable telemetry by setting the env variable HF_HUB_DISABLE_TELEMETRY=1
        data = {
            "src": self.src,
        }
        try:
            send_telemetry(
                topic="py_client/initiated",
                library_name="gradio_client",
                library_version=utils.__version__,
                user_agent=data,
            )
        except Exception:
            pass

    def _infer_fn_index(self, api_name: str | None, fn_index: int | None) -> int:
        inferred_fn_index = None
        if api_name is not None:
            for i, d in enumerate(self.config["dependencies"]):
                config_api_name = d.get("api_name")
                if config_api_name is None:
                    continue
                if "/" + config_api_name == api_name:
                    inferred_fn_index = i
                    break
            else:
                error_message = f"Cannot find a function with `api_name`: {api_name}."
                if not api_name.startswith("/"):
                    error_message += " Did you mean to use a leading slash?"
                raise ValueError(error_message)
        elif fn_index is not None:
            inferred_fn_index = fn_index
        else:
            valid_endpoints = [
                e for e in self.endpoints if e.is_valid and e.api_name is not None
            ]
            if len(valid_endpoints) == 1:
                inferred_fn_index = valid_endpoints[0].fn_index
            else:
                raise ValueError(
                    "This Gradio app might have multiple endpoints. Please specify an `api_name` or `fn_index`"
                )
        return inferred_fn_index

    def __del__(self):
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=True)

    def _space_name_to_src(self, space) -> str | None:
        return huggingface_hub.space_info(space, token=self.hf_token).host  # type: ignore

    def _get_config(self) -> dict:
        r = requests.get(
            urllib.parse.urljoin(self.src, utils.CONFIG_URL), headers=self.headers
        )
        if r.ok:
            return r.json()
        else:  # to support older versions of Gradio
            r = requests.get(self.src, headers=self.headers)
            # some basic regex to extract the config
            result = re.search(r"window.gradio_config = (.*?);[\s]*</script>", r.text)
            try:
                config = json.loads(result.group(1))  # type: ignore
            except AttributeError as ae:
                raise ValueError(
                    f"Could not get Gradio config from: {self.src}"
                ) from ae
            if "allow_flagging" in config:
                raise ValueError(
                    "Gradio 2.x is not supported by this client. Please upgrade your Gradio app to Gradio 3.x or higher."
                )
            return config


class Endpoint:
    """Helper class for storing all the information about a single API endpoint."""

    def __init__(self, client: Client, fn_index: int, dependency: dict):
        self.client: Client = client
        self.fn_index = fn_index
        self.dependency = dependency
        api_name = dependency.get("api_name")
        self.api_name: str | None = None if api_name is None else "/" + api_name
        self.use_ws = self._use_websocket(self.dependency)
        self.input_component_types = []
        self.output_component_types = []
        self.root_url = client.src + "/" if not client.src.endswith("/") else client.src
        try:
            self.serializers, self.deserializers = self._setup_serializers()
            self.is_valid = self.dependency[
                "backend_fn"
            ]  # Only a real API endpoint if backend_fn is True and serializers are valid
        except AssertionError:
            self.is_valid = False

    def __repr__(self):
        return f"Endpoint src: {self.client.src}, api_name: {self.api_name}, fn_index: {self.fn_index}"

    def __str__(self):
        return self.__repr__()

    def make_end_to_end_fn(self, helper: Communicator | None = None):

        _predict = self.make_predict(helper)

        def _inner(*data):
            if not self.is_valid:
                raise utils.InvalidAPIEndpointError()
            data = self.insert_state(*data)
            if self.client.serialize:
                data = self.serialize(*data)
            predictions = _predict(*data)
            predictions = self.process_predictions(*predictions)
            # Append final output only if not already present
            # for consistency between generators and not generators
            if helper:
                with helper.lock:
                    if not helper.job.outputs:
                        helper.job.outputs.append(predictions)
            return predictions

        return _inner

    def make_predict(self, helper: Communicator | None = None):
        def _predict(*data) -> tuple:
            data = json.dumps(
                {
                    "data": data,
                    "fn_index": self.fn_index,
                    "session_hash": self.client.session_hash,
                }
            )
            hash_data = json.dumps(
                {
                    "fn_index": self.fn_index,
                    "session_hash": self.client.session_hash,
                }
            )

            if self.use_ws:
                result = utils.synchronize_async(self._ws_fn, data, hash_data, helper)
                if "error" in result:
                    raise ValueError(result["error"])
            else:
                response = requests.post(
                    self.client.api_url, headers=self.client.headers, data=data
                )
                result = json.loads(response.content.decode("utf-8"))
            try:
                output = result["data"]
            except KeyError as ke:
                is_public_space = (
                    self.client.space_id
                    and not huggingface_hub.space_info(self.client.space_id).private
                )
                if "error" in result and "429" in result["error"] and is_public_space:
                    raise utils.TooManyRequestsError(
                        f"Too many requests to the API, please try again later. To avoid being rate-limited, "
                        f"please duplicate the Space using Client.duplicate({self.client.space_id}) "
                        f"and pass in your Hugging Face token."
                    ) from None
                elif "error" in result:
                    raise ValueError(result["error"]) from None
                raise KeyError(
                    f"Could not find 'data' key in response. Response received: {result}"
                ) from ke
            return tuple(output)

        return _predict

    def _predict_resolve(self, *data) -> Any:
        """Needed for gradio.load(), which has a slightly different signature for serializing/deserializing"""
        outputs = self.make_predict()(*data)
        if len(self.dependency["outputs"]) == 1:
            return outputs[0]
        return outputs

    def _upload(
        self, file_paths: list[str | list[str]]
    ) -> list[str | list[str]] | list[dict[str, Any] | list[dict[str, Any]]]:
        if not file_paths:
            return []
        # Put all the filepaths in one file
        # but then keep track of which index in the
        # original list they came from so we can recreate
        # the original structure
        files = []
        indices = []
        for i, fs in enumerate(file_paths):
            if not isinstance(fs, list):
                fs = [fs]
            for f in fs:
                files.append(("files", (Path(f).name, open(f, "rb"))))  # noqa: SIM115
                indices.append(i)
        r = requests.post(
            self.client.upload_url, headers=self.client.headers, files=files
        )
        if r.status_code != 200:
            uploaded = file_paths
        else:
            uploaded = []
            result = r.json()
            for i, fs in enumerate(file_paths):
                if isinstance(fs, list):
                    output = [o for ix, o in enumerate(result) if indices[ix] == i]
                    res = [
                        {
                            "is_file": True,
                            "name": o,
                            "orig_name": Path(f).name,
                            "data": None,
                        }
                        for f, o in zip(fs, output)
                    ]
                else:
                    o = next(o for ix, o in enumerate(result) if indices[ix] == i)
                    res = {
                        "is_file": True,
                        "name": o,
                        "orig_name": Path(fs).name,
                        "data": None,
                    }
                uploaded.append(res)
        return uploaded

    def _add_uploaded_files_to_data(
        self,
        files: list[str | list[str]] | list[dict[str, Any] | list[dict[str, Any]]],
        data: list[Any],
    ) -> None:
        """Helper function to modify the input data with the uploaded files."""
        file_counter = 0
        for i, t in enumerate(self.input_component_types):
            if t in ["file", "uploadbutton"]:
                data[i] = files[file_counter]
                file_counter += 1

    def insert_state(self, *data) -> tuple:
        data = list(data)
        for i, input_component_type in enumerate(self.input_component_types):
            if input_component_type == utils.STATE_COMPONENT:
                data.insert(i, None)
        return tuple(data)

    def remove_state(self, *data) -> tuple:
        data = [
            d
            for d, oct in zip(data, self.output_component_types)
            if oct != utils.STATE_COMPONENT
        ]
        return tuple(data)

    def reduce_singleton_output(self, *data) -> Any:
        if (
            len(
                [
                    oct
                    for oct in self.output_component_types
                    if oct != utils.STATE_COMPONENT
                ]
            )
            == 1
        ):
            return data[0]
        else:
            return data

    def serialize(self, *data) -> tuple:
        assert len(data) == len(
            self.serializers
        ), f"Expected {len(self.serializers)} arguments, got {len(data)}"

        files = [
            f
            for f, t in zip(data, self.input_component_types)
            if t in ["file", "uploadbutton"]
        ]
        uploaded_files = self._upload(files)
        self._add_uploaded_files_to_data(uploaded_files, list(data))

        o = tuple([s.serialize(d) for s, d in zip(self.serializers, data)])
        return o

    def deserialize(self, *data) -> tuple:
        assert len(data) == len(
            self.deserializers
        ), f"Expected {len(self.deserializers)} outputs, got {len(data)}"
        outputs = tuple(
            [
                s.deserialize(d, hf_token=self.client.hf_token, root_url=self.root_url)
                for s, d in zip(self.deserializers, data)
            ]
        )
        return outputs

    def process_predictions(self, *predictions):
        if self.client.serialize:
            predictions = self.deserialize(*predictions)
        predictions = self.remove_state(*predictions)
        predictions = self.reduce_singleton_output(*predictions)
        return predictions

    def _setup_serializers(self) -> tuple[list[Serializable], list[Serializable]]:
        inputs = self.dependency["inputs"]
        serializers = []

        for i in inputs:
            for component in self.client.config["components"]:
                if component["id"] == i:
                    component_name = component["type"]
                    self.input_component_types.append(component_name)
                    if component.get("serializer"):
                        serializer_name = component["serializer"]
                        assert (
                            serializer_name in serializing.SERIALIZER_MAPPING
                        ), f"Unknown serializer: {serializer_name}, you may need to update your gradio_client version."
                        serializer = serializing.SERIALIZER_MAPPING[serializer_name]
                    else:
                        assert (
                            component_name in serializing.COMPONENT_MAPPING
                        ), f"Unknown component: {component_name}, you may need to update your gradio_client version."
                        serializer = serializing.COMPONENT_MAPPING[component_name]
                    serializers.append(serializer())  # type: ignore

        outputs = self.dependency["outputs"]
        deserializers = []
        for i in outputs:
            for component in self.client.config["components"]:
                if component["id"] == i:
                    component_name = component["type"]
                    self.output_component_types.append(component_name)
                    if component.get("serializer"):
                        serializer_name = component["serializer"]
                        assert (
                            serializer_name in serializing.SERIALIZER_MAPPING
                        ), f"Unknown serializer: {serializer_name}, you may need to update your gradio_client version."
                        deserializer = serializing.SERIALIZER_MAPPING[serializer_name]
                    else:
                        assert (
                            component_name in serializing.COMPONENT_MAPPING
                        ), f"Unknown component: {component_name}, you may need to update your gradio_client version."
                        deserializer = serializing.COMPONENT_MAPPING[component_name]
                    deserializers.append(deserializer())  # type: ignore

        return serializers, deserializers

    def _use_websocket(self, dependency: dict) -> bool:
        queue_enabled = self.client.config.get("enable_queue", False)
        queue_uses_websocket = version.parse(
            self.client.config.get("version", "2.0")
        ) >= version.Version("3.2")
        dependency_uses_queue = dependency.get("queue", False) is not False
        return queue_enabled and queue_uses_websocket and dependency_uses_queue

    async def _ws_fn(self, data, hash_data, helper: Communicator):
        async with websockets.connect(  # type: ignore
            self.client.ws_url,
            open_timeout=10,
            extra_headers=self.client.headers,
            max_size=1024 * 1024 * 1024,
        ) as websocket:
            return await utils.get_pred_from_ws(websocket, data, hash_data, helper)


@document("result", "outputs", "status")
class Job(Future):
    """
    A Job is a wrapper over the Future class that represents a prediction call that has been
    submitted by the Gradio client. This class is not meant to be instantiated directly, but rather
    is created by the Client.submit() method.

    A Job object includes methods to get the status of the prediction call, as well to get the outputs of
    the prediction call. Job objects are also iterable, and can be used in a loop to get the outputs
    of prediction calls as they become available for generator endpoints.
    """

    def __init__(
        self,
        future: Future,
        communicator: Communicator | None = None,
        verbose: bool = True,
        space_id: str | None = None,
    ):
        """
        Parameters:
            future: The future object that represents the prediction call, created by the Client.submit() method
            communicator: The communicator object that is used to communicate between the client and the background thread running the job
            verbose: Whether to print any status-related messages to the console
            space_id: The space ID corresponding to the Client object that created this Job object
        """
        self.future = future
        self.communicator = communicator
        self._counter = 0
        self.verbose = verbose
        self.space_id = space_id

    def __iter__(self) -> Job:
        return self

    def __next__(self) -> tuple | Any:
        if not self.communicator:
            raise StopIteration()

        with self.communicator.lock:
            if self.communicator.job.latest_status.code == Status.FINISHED:
                raise StopIteration()

        while True:
            with self.communicator.lock:
                if len(self.communicator.job.outputs) == self._counter + 1:
                    o = self.communicator.job.outputs[self._counter]
                    self._counter += 1
                    return o
                if self.communicator.job.latest_status.code == Status.FINISHED:
                    raise StopIteration()

    def result(self, timeout=None) -> Any:
        """
        Return the result of the call that the future represents. Raises CancelledError: If the future was cancelled, TimeoutError: If the future didn't finish executing before the given timeout, and Exception: If the call raised then that exception will be raised.

        Parameters:
            timeout: The number of seconds to wait for the result if the future isn't done. If None, then there is no limit on the wait time.
        Returns:
            The result of the call that the future represents.
        Example:
            from gradio_client import Client
            calculator = Client(src="gradio/calculator")
            job = calculator.submit("foo", "add", 4, fn_index=0)
            job.result(timeout=5)
            >> 9
        """
        if self.communicator:
            timeout = timeout or float("inf")
            if self.future._exception:  # type: ignore
                raise self.future._exception  # type: ignore
            with self.communicator.lock:
                if self.communicator.job.outputs:
                    return self.communicator.job.outputs[0]
            start = datetime.now()
            while True:
                if (datetime.now() - start).seconds > timeout:
                    raise TimeoutError()
                if self.future._exception:  # type: ignore
                    raise self.future._exception  # type: ignore
                with self.communicator.lock:
                    if self.communicator.job.outputs:
                        return self.communicator.job.outputs[0]
                time.sleep(0.01)
        else:
            return super().result(timeout=timeout)

    def outputs(self) -> list[tuple | Any]:
        """
        Returns a list containing the latest outputs from the Job.

        If the endpoint has multiple output components, the list will contain
        a tuple of results. Otherwise, it will contain the results without storing them
        in tuples.

        For endpoints that are queued, this list will contain the final job output even
        if that endpoint does not use a generator function.

        Example:
            from gradio_client import Client
            client = Client(src="gradio/count_generator")
            job = client.submit(3, api_name="/count")
            while not job.done():
                time.sleep(0.1)
            job.outputs()
            >> ['0', '1', '2']
        """
        if not self.communicator:
            return []
        else:
            with self.communicator.lock:
                return self.communicator.job.outputs

    def status(self) -> StatusUpdate:
        """
        Returns the latest status update from the Job in the form of a StatusUpdate
        object, which contains the following fields: code, rank, queue_size, success, time, eta, and progress_data.

        progress_data is a list of updates emitted by the gr.Progress() tracker of the event handler. Each element
        of the list has the following fields: index, length, unit, progress, desc. If the event handler does not have
        a gr.Progress() tracker, the progress_data field will be None.

        Example:
            from gradio_client import Client
            client = Client(src="gradio/calculator")
            job = client.submit(5, "add", 4, api_name="/predict")
            job.status()
            >> <Status.STARTING: 'STARTING'>
            job.status().eta
            >> 43.241  # seconds
        """
        time = datetime.now()
        cancelled = False
        if self.communicator:
            with self.communicator.lock:
                cancelled = self.communicator.should_cancel
        if cancelled:
            return StatusUpdate(
                code=Status.CANCELLED,
                rank=0,
                queue_size=None,
                success=False,
                time=time,
                eta=None,
                progress_data=None,
            )
        if self.done():
            if not self.future._exception:  # type: ignore
                return StatusUpdate(
                    code=Status.FINISHED,
                    rank=0,
                    queue_size=None,
                    success=True,
                    time=time,
                    eta=None,
                    progress_data=None,
                )
            else:
                return StatusUpdate(
                    code=Status.FINISHED,
                    rank=0,
                    queue_size=None,
                    success=False,
                    time=time,
                    eta=None,
                    progress_data=None,
                )
        else:
            if not self.communicator:
                return StatusUpdate(
                    code=Status.PROCESSING,
                    rank=0,
                    queue_size=None,
                    success=None,
                    time=time,
                    eta=None,
                    progress_data=None,
                )
            else:
                with self.communicator.lock:
                    eta = self.communicator.job.latest_status.eta
                    if self.verbose and self.space_id and eta and eta > 30:
                        print(
                            f"Due to heavy traffic on this app, the prediction will take approximately {int(eta)} seconds."
                            f"For faster predictions without waiting in queue, you may duplicate the space using: Client.duplicate({self.space_id})"
                        )
                    return self.communicator.job.latest_status

    def __getattr__(self, name):
        """Forwards any properties to the Future class."""
        return getattr(self.future, name)

    def cancel(self) -> bool:
        """Cancels the job as best as possible.

        If the app you are connecting to has the gradio queue enabled, the job
        will be cancelled locally as soon as possible. For apps that do not use the
        queue, the job cannot be cancelled if it's been sent to the local executor
        (for the time being).

        Note: In general, this DOES not stop the process from running in the upstream server
        except for the following situations:

        1. If the job is queued upstream, it will be removed from the queue and the server will not run the job
        2. If the job has iterative outputs, the job will finish as soon as the current iteration finishes running
        3. If the job has not been picked up by the queue yet, the queue will not pick up the job
        """
        if self.communicator:
            with self.communicator.lock:
                self.communicator.should_cancel = True
                return True
        return self.future.cancel()
