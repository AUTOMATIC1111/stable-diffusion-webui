"""Contains all of the components that can be used with Gradio Interface / Blocks.
Along with the docs for each component, you can find the names of example demos that use
each component. These demos are located in the `demo` directory."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import operator
import os
import random
import secrets
import shutil
import tempfile
import urllib.request
import warnings
from copy import deepcopy
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable, Dict

import aiofiles
import altair as alt
import numpy as np
import pandas as pd
import PIL
import PIL.ImageOps
import requests
from fastapi import UploadFile
from ffmpy import FFmpeg
from gradio_client import media_data
from gradio_client import utils as client_utils
from gradio_client.data_classes import FileData
from gradio_client.documentation import document, set_documentation_group
from gradio_client.serializing import (
    BooleanSerializable,
    FileSerializable,
    GallerySerializable,
    ImgSerializable,
    JSONSerializable,
    ListStringSerializable,
    NumberSerializable,
    Serializable,
    SimpleSerializable,
    StringSerializable,
    VideoSerializable,
)
from pandas.api.types import is_numeric_dtype
from PIL import Image as _Image  # using _ to minimize namespace pollution
from typing_extensions import Literal

from gradio import processing_utils, utils
from gradio.blocks import Block, BlockContext
from gradio.events import (
    Blurrable,
    Changeable,
    Clearable,
    Clickable,
    Editable,
    EventListener,
    EventListenerMethod,
    Inputable,
    Playable,
    Releaseable,
    Selectable,
    Streamable,
    Submittable,
    Uploadable,
)
from gradio.interpretation import NeighborInterpretable, TokenInterpretable
from gradio.layouts import Column, Form, Row

if TYPE_CHECKING:
    from typing import TypedDict

    class DataframeData(TypedDict):
        headers: list[str]
        data: list[list[str | int | bool]]


set_documentation_group("component")
_Image.init()  # fixes https://github.com/gradio-app/gradio/issues/2843


class _Keywords(Enum):
    NO_VALUE = "NO_VALUE"  # Used as a sentinel to determine if nothing is provided as a argument for `value` in `Component.update()`
    FINISHED_ITERATING = "FINISHED_ITERATING"  # Used to skip processing of a component's value (needed for generators + state)


class Component(Block, Serializable):
    """
    A base class for defining the methods that all gradio components should have.
    """

    def __init__(self, *args, **kwargs):
        Block.__init__(self, *args, **kwargs)
        EventListener.__init__(self)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"{self.get_block_name()}"

    def get_config(self):
        """
        :return: a dictionary with context variables for the javascript file associated with the context
        """
        return {
            "name": self.get_block_name(),
            **super().get_config(),
        }

    def preprocess(self, x: Any) -> Any:
        """
        Any preprocessing needed to be performed on function input.
        """
        return x

    def postprocess(self, y):
        """
        Any postprocessing needed to be performed on function output.
        """
        return y

    def style(
        self,
        *,
        container: bool | None = None,
        **kwargs,
    ):
        """
        This method can be used to change the appearance of the component.
        Parameters:
            container: If True, will place the component in a container - providing some extra padding around the border.
        """
        put_deprecated_params_in_box = False
        if "rounded" in kwargs:
            warnings.warn(
                "'rounded' styling is no longer supported. To round adjacent components together, place them in a Column(variant='box')."
            )
            if isinstance(kwargs["rounded"], (list, tuple)):
                put_deprecated_params_in_box = True
            kwargs.pop("rounded")
        if "margin" in kwargs:
            warnings.warn(
                "'margin' styling is no longer supported. To place adjacent components together without margin, place them in a Column(variant='box')."
            )
            if isinstance(kwargs["margin"], (list, tuple)):
                put_deprecated_params_in_box = True
            kwargs.pop("margin")
        if "border" in kwargs:
            warnings.warn(
                "'border' styling is no longer supported. To place adjacent components in a shared border, place them in a Column(variant='box')."
            )
            kwargs.pop("border")
        if container is not None:
            self._style["container"] = container
        if len(kwargs):
            for key in kwargs:
                warnings.warn(f"Unknown style parameter: {key}")
        if (
            put_deprecated_params_in_box
            and isinstance(self.parent, (Row, Column))
            and self.parent.variant == "default"
        ):
            self.parent.variant = "compact"
        return self


class IOComponent(Component):
    """
    A base class for defining methods that all input/output components should have.
    """

    def __init__(
        self,
        *,
        value: Any = None,
        label: str | None = None,
        info: str | None = None,
        show_label: bool = True,
        interactive: bool | None = None,
        visible: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        load_fn: Callable | None = None,
        every: float | None = None,
        **kwargs,
    ):
        self.temp_files: set[str] = set()
        self.DEFAULT_TEMP_DIR = os.environ.get("GRADIO_TEMP_DIR") or str(
            Path(tempfile.gettempdir()) / "gradio"
        )

        Component.__init__(
            self, elem_id=elem_id, elem_classes=elem_classes, visible=visible, **kwargs
        )

        self.label = label
        self.info = info
        self.show_label = show_label
        self.interactive = interactive

        # load_event is set in the Blocks.attach_load_events method
        self.load_event: None | dict[str, Any] = None
        self.load_event_to_attach = None
        load_fn, initial_value = self.get_load_fn_and_initial_value(value)
        self.value = (
            initial_value
            if self._skip_init_processing
            else self.postprocess(initial_value)
        )
        if callable(load_fn):
            self.attach_load_event(load_fn, every)

    @staticmethod
    def hash_file(file_path: str, chunk_num_blocks: int = 128) -> str:
        sha1 = hashlib.sha1()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_num_blocks * sha1.block_size), b""):
                sha1.update(chunk)
        return sha1.hexdigest()

    @staticmethod
    def hash_url(url: str, chunk_num_blocks: int = 128) -> str:
        sha1 = hashlib.sha1()
        remote = urllib.request.urlopen(url)
        max_file_size = 100 * 1024 * 1024  # 100MB
        total_read = 0
        while True:
            data = remote.read(chunk_num_blocks * sha1.block_size)
            total_read += chunk_num_blocks * sha1.block_size
            if not data or total_read > max_file_size:
                break
            sha1.update(data)
        return sha1.hexdigest()

    @staticmethod
    def hash_bytes(bytes: bytes):
        sha1 = hashlib.sha1()
        sha1.update(bytes)
        return sha1.hexdigest()

    @staticmethod
    def hash_base64(base64_encoding: str, chunk_num_blocks: int = 128) -> str:
        sha1 = hashlib.sha1()
        for i in range(0, len(base64_encoding), chunk_num_blocks * sha1.block_size):
            data = base64_encoding[i : i + chunk_num_blocks * sha1.block_size]
            sha1.update(data.encode("utf-8"))
        return sha1.hexdigest()

    def make_temp_copy_if_needed(self, file_path: str) -> str:
        """Returns a temporary file path for a copy of the given file path if it does
        not already exist. Otherwise returns the path to the existing temp file."""
        temp_dir = self.hash_file(file_path)
        temp_dir = Path(self.DEFAULT_TEMP_DIR) / temp_dir
        temp_dir.mkdir(exist_ok=True, parents=True)

        name = client_utils.strip_invalid_filename_characters(Path(file_path).name)
        full_temp_file_path = str(utils.abspath(temp_dir / name))

        if not Path(full_temp_file_path).exists():
            shutil.copy2(file_path, full_temp_file_path)

        self.temp_files.add(full_temp_file_path)
        return full_temp_file_path

    async def save_uploaded_file(self, file: UploadFile, upload_dir: str) -> str:
        temp_dir = secrets.token_hex(
            20
        )  # Since the full file is being uploaded anyways, there is no benefit to hashing the file.
        temp_dir = Path(upload_dir) / temp_dir
        temp_dir.mkdir(exist_ok=True, parents=True)

        if file.filename:
            file_name = Path(file.filename).name
            name = client_utils.strip_invalid_filename_characters(file_name)
        else:
            name = f"tmp{secrets.token_hex(5)}"

        full_temp_file_path = str(utils.abspath(temp_dir / name))

        async with aiofiles.open(full_temp_file_path, "wb") as output_file:
            while True:
                content = await file.read(100 * 1024 * 1024)
                if not content:
                    break
                await output_file.write(content)

        return full_temp_file_path

    def download_temp_copy_if_needed(self, url: str) -> str:
        """Downloads a file and makes a temporary file path for a copy if does not already
        exist. Otherwise returns the path to the existing temp file."""
        temp_dir = self.hash_url(url)
        temp_dir = Path(self.DEFAULT_TEMP_DIR) / temp_dir
        temp_dir.mkdir(exist_ok=True, parents=True)

        name = client_utils.strip_invalid_filename_characters(Path(url).name)
        full_temp_file_path = str(utils.abspath(temp_dir / name))

        if not Path(full_temp_file_path).exists():
            with requests.get(url, stream=True) as r, open(
                full_temp_file_path, "wb"
            ) as f:
                shutil.copyfileobj(r.raw, f)

        self.temp_files.add(full_temp_file_path)
        return full_temp_file_path

    def base64_to_temp_file_if_needed(
        self, base64_encoding: str, file_name: str | None = None
    ) -> str:
        """Converts a base64 encoding to a file and returns the path to the file if
        the file doesn't already exist. Otherwise returns the path to the existing file.
        """
        temp_dir = self.hash_base64(base64_encoding)
        temp_dir = Path(self.DEFAULT_TEMP_DIR) / temp_dir
        temp_dir.mkdir(exist_ok=True, parents=True)

        guess_extension = client_utils.get_extension(base64_encoding)
        if file_name:
            file_name = client_utils.strip_invalid_filename_characters(file_name)
        elif guess_extension:
            file_name = f"file.{guess_extension}"
        else:
            file_name = "file"

        full_temp_file_path = str(utils.abspath(temp_dir / file_name))  # type: ignore

        if not Path(full_temp_file_path).exists():
            data, _ = client_utils.decode_base64_to_binary(base64_encoding)
            with open(full_temp_file_path, "wb") as fb:
                fb.write(data)

        self.temp_files.add(full_temp_file_path)
        return full_temp_file_path

    def pil_to_temp_file(self, img: _Image.Image, dir: str, format="png") -> str:
        bytes_data = processing_utils.encode_pil_to_bytes(img, format)
        temp_dir = Path(dir) / self.hash_bytes(bytes_data)
        temp_dir.mkdir(exist_ok=True, parents=True)
        filename = str(temp_dir / f"image.{format}")
        img.save(filename, pnginfo=processing_utils.get_pil_metadata(img))
        return filename

    def img_array_to_temp_file(self, arr: np.ndarray, dir: str) -> str:
        pil_image = _Image.fromarray(
            processing_utils._convert(arr, np.uint8, force_copy=False)
        )
        return self.pil_to_temp_file(pil_image, dir, format="png")

    def audio_to_temp_file(
        self, data: np.ndarray, sample_rate: int, dir: str, format: str
    ):
        temp_dir = Path(dir) / self.hash_bytes(data.tobytes())
        temp_dir.mkdir(exist_ok=True, parents=True)
        filename = str(temp_dir / f"audio.{format}")
        processing_utils.audio_to_file(sample_rate, data, filename, format=format)
        return filename

    def file_bytes_to_file(self, data: bytes, dir: str, file_name: str):
        path = Path(dir) / self.hash_bytes(data)
        path.mkdir(exist_ok=True, parents=True)
        path = path / Path(file_name).name
        path.write_bytes(data)
        return path

    def get_config(self):
        config = {
            "label": self.label,
            "show_label": self.show_label,
            "interactive": self.interactive,
            **super().get_config(),
        }
        if self.info:
            config["info"] = self.info
        return config

    @staticmethod
    def get_load_fn_and_initial_value(value):
        if callable(value):
            initial_value = value()
            load_fn = value
        else:
            initial_value = value
            load_fn = None
        return load_fn, initial_value

    def attach_load_event(self, callable: Callable, every: float | None):
        """Add a load event that runs `callable`, optionally every `every` seconds."""
        self.load_event_to_attach = (callable, every)

    def as_example(self, input_data):
        """Return the input data in a way that can be displayed by the examples dataset component in the front-end."""
        return input_data


class FormComponent:
    def get_expected_parent(self) -> type[Form]:
        return Form


@document("style")
class Textbox(
    FormComponent,
    Changeable,
    Inputable,
    Selectable,
    Submittable,
    Blurrable,
    IOComponent,
    StringSerializable,
    TokenInterpretable,
):
    """
    Creates a textarea for user to enter string input or display string output.
    Preprocessing: passes textarea value as a {str} into the function.
    Postprocessing: expects a {str} returned from function and sets textarea value to it.
    Examples-format: a {str} representing the textbox input.

    Demos: hello_world, diff_texts, sentence_builder
    Guides: creating-a-chatbot, real-time-speech-recognition
    """

    def __init__(
        self,
        value: str | Callable | None = "",
        *,
        lines: int = 1,
        max_lines: int = 20,
        placeholder: str | None = None,
        label: str | None = None,
        info: str | None = None,
        every: float | None = None,
        show_label: bool = True,
        interactive: bool | None = None,
        visible: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        type: str = "text",
        **kwargs,
    ):
        """
        Parameters:
            value: default text to provide in textarea. If callable, the function will be called whenever the app loads to set the initial value of the component.
            lines: minimum number of line rows to provide in textarea.
            max_lines: maximum number of line rows to provide in textarea.
            placeholder: placeholder hint to provide behind textarea.
            label: component name in interface.
            info: additional component description.
            every: If `value` is a callable, run the function 'every' number of seconds while the client connection is open. Has no effect otherwise. Queue must be enabled. The event can be accessed (e.g. to cancel it) via this component's .load_event attribute.
            show_label: if True, will display label.
            interactive: if True, will be rendered as an editable textbox; if False, editing will be disabled. If not provided, this is inferred based on whether the component is used as an input or output.
            visible: If False, component will be hidden.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.
            type: The type of textbox. One of: 'text', 'password', 'email', Default is 'text'.
        """
        if type not in ["text", "password", "email"]:
            raise ValueError('`type` must be one of "text", "password", or "email".')

        #
        self.lines = lines
        if type == "text":
            self.max_lines = max(lines, max_lines)
        else:
            self.max_lines = 1
        self.placeholder = placeholder
        self.select: EventListenerMethod
        """
        Event listener for when the user selects text in the Textbox.
        Uses event data gradio.SelectData to carry `value` referring to selected substring, and `index` tuple referring to selected range endpoints.
        See EventData documentation on how to use this event data.
        """
        IOComponent.__init__(
            self,
            label=label,
            info=info,
            every=every,
            show_label=show_label,
            interactive=interactive,
            visible=visible,
            elem_id=elem_id,
            elem_classes=elem_classes,
            value=value,
            **kwargs,
        )
        TokenInterpretable.__init__(self)
        self.cleared_value = ""
        self.type = type

    def get_config(self):
        return {
            "lines": self.lines,
            "max_lines": self.max_lines,
            "placeholder": self.placeholder,
            "value": self.value,
            "type": self.type,
            **IOComponent.get_config(self),
        }

    @staticmethod
    def update(
        value: str | Literal[_Keywords.NO_VALUE] | None = _Keywords.NO_VALUE,
        lines: int | None = None,
        max_lines: int | None = None,
        placeholder: str | None = None,
        label: str | None = None,
        show_label: bool | None = None,
        visible: bool | None = None,
        interactive: bool | None = None,
        type: str | None = None,
    ):
        return {
            "lines": lines,
            "max_lines": max_lines,
            "placeholder": placeholder,
            "label": label,
            "show_label": show_label,
            "visible": visible,
            "value": value,
            "type": type,
            "interactive": interactive,
            "__type__": "update",
        }

    def preprocess(self, x: str | None) -> str | None:
        """
        Preprocesses input (converts it to a string) before passing it to the function.
        Parameters:
            x: text
        Returns:
            text
        """
        return None if x is None else str(x)

    def postprocess(self, y: str | None) -> str | None:
        """
        Postproccess the function output y by converting it to a str before passing it to the frontend.
        Parameters:
            y: function output to postprocess.
        Returns:
            text
        """
        return None if y is None else str(y)

    def set_interpret_parameters(
        self, separator: str = " ", replacement: str | None = None
    ):
        """
        Calculates interpretation score of characters in input by splitting input into tokens, then using a "leave one out" method to calculate the score of each token by removing each token and measuring the delta of the output value.
        Parameters:
            separator: Separator to use to split input into tokens.
            replacement: In the "leave one out" step, the text that the token should be replaced with. If None, the token is removed altogether.
        """
        self.interpretation_separator = separator
        self.interpretation_replacement = replacement
        return self

    def tokenize(self, x: str) -> tuple[list[str], list[str], None]:
        """
        Tokenizes an input string by dividing into "words" delimited by self.interpretation_separator
        """
        tokens = x.split(self.interpretation_separator)
        leave_one_out_strings = []
        for index in range(len(tokens)):
            leave_one_out_set = list(tokens)
            if self.interpretation_replacement is None:
                leave_one_out_set.pop(index)
            else:
                leave_one_out_set[index] = self.interpretation_replacement
            leave_one_out_strings.append(
                self.interpretation_separator.join(leave_one_out_set)
            )
        return tokens, leave_one_out_strings, None

    def get_masked_inputs(
        self, tokens: list[str], binary_mask_matrix: list[list[int]]
    ) -> list[str]:
        """
        Constructs partially-masked sentences for SHAP interpretation
        """
        masked_inputs = []
        for binary_mask_vector in binary_mask_matrix:
            masked_input = np.array(tokens)[np.array(binary_mask_vector, dtype=bool)]
            masked_inputs.append(self.interpretation_separator.join(masked_input))
        return masked_inputs

    def get_interpretation_scores(
        self, x, neighbors, scores: list[float], tokens: list[str], masks=None, **kwargs
    ) -> list[tuple[str, float]]:
        """
        Returns:
            Each tuple set represents a set of characters and their corresponding interpretation score.
        """
        result = []
        for token, score in zip(tokens, scores):
            result.append((token, score))
            result.append((self.interpretation_separator, 0))
        return result

    def style(
        self,
        *,
        show_copy_button: bool | None = None,
        container: bool | None = None,
        **kwargs,
    ):
        """
        This method can be used to change the appearance of the Textbox component.
        Parameters:
            show_copy_button: If True, includes a copy button to copy the text in the textbox. Only applies if show_label is True.
            container: If True, will place the component in a container - providing some extra padding around the border.
        """
        if show_copy_button is not None:
            self._style["show_copy_button"] = show_copy_button

        return Component.style(self, container=container, **kwargs)


@document("style")
class Number(
    FormComponent,
    Changeable,
    Inputable,
    Submittable,
    Blurrable,
    IOComponent,
    NumberSerializable,
    NeighborInterpretable,
):
    """
    Creates a numeric field for user to enter numbers as input or display numeric output.
    Preprocessing: passes field value as a {float} or {int} into the function, depending on `precision`.
    Postprocessing: expects an {int} or {float} returned from the function and sets field value to it.
    Examples-format: a {float} or {int} representing the number's value.

    Demos: tax_calculator, titanic_survival, blocks_simple_squares
    """

    def __init__(
        self,
        value: float | Callable | None = None,
        *,
        label: str | None = None,
        info: str | None = None,
        every: float | None = None,
        show_label: bool = True,
        interactive: bool | None = None,
        visible: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        precision: int | None = None,
        **kwargs,
    ):
        """
        Parameters:
            value: default value. If callable, the function will be called whenever the app loads to set the initial value of the component.
            label: component name in interface.
            info: additional component description.
            every: If `value` is a callable, run the function 'every' number of seconds while the client connection is open. Has no effect otherwise. Queue must be enabled. The event can be accessed (e.g. to cancel it) via this component's .load_event attribute.
            show_label: if True, will display label.
            interactive: if True, will be editable; if False, editing will be disabled. If not provided, this is inferred based on whether the component is used as an input or output.
            visible: If False, component will be hidden.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.
            precision: Precision to round input/output to. If set to 0, will round to nearest integer and convert type to int. If None, no rounding happens.
        """
        self.precision = precision
        IOComponent.__init__(
            self,
            label=label,
            info=info,
            every=every,
            show_label=show_label,
            interactive=interactive,
            visible=visible,
            elem_id=elem_id,
            elem_classes=elem_classes,
            value=value,
            **kwargs,
        )
        NeighborInterpretable.__init__(self)

    @staticmethod
    def _round_to_precision(num: float | int, precision: int | None) -> float | int:
        """
        Round to a given precision.

        If precision is None, no rounding happens. If 0, num is converted to int.

        Parameters:
            num: Number to round.
            precision: Precision to round to.
        Returns:
            rounded number
        """
        if precision is None:
            return float(num)
        elif precision == 0:
            return int(round(num, precision))
        else:
            return round(num, precision)

    def get_config(self):
        return {
            "value": self.value,
            **IOComponent.get_config(self),
        }

    @staticmethod
    def update(
        value: float | Literal[_Keywords.NO_VALUE] | None = _Keywords.NO_VALUE,
        label: str | None = None,
        show_label: bool | None = None,
        interactive: bool | None = None,
        visible: bool | None = None,
    ):
        return {
            "label": label,
            "show_label": show_label,
            "visible": visible,
            "value": value,
            "interactive": interactive,
            "__type__": "update",
        }

    def preprocess(self, x: float | None) -> float | None:
        """
        Parameters:
            x: numeric input
        Returns:
            number representing function input
        """
        if x is None:
            return None
        return self._round_to_precision(x, self.precision)

    def postprocess(self, y: float | None) -> float | None:
        """
        Any postprocessing needed to be performed on function output.

        Parameters:
            y: numeric output
        Returns:
            number representing function output
        """
        if y is None:
            return None
        return self._round_to_precision(y, self.precision)

    def set_interpret_parameters(
        self, steps: int = 3, delta: float = 1, delta_type: str = "percent"
    ):
        """
        Calculates interpretation scores of numeric values close to the input number.
        Parameters:
            steps: Number of nearby values to measure in each direction (above and below the input number).
            delta: Size of step in each direction between nearby values.
            delta_type: "percent" if delta step between nearby values should be a calculated as a percent, or "absolute" if delta should be a constant step change.
        """
        self.interpretation_steps = steps
        self.interpretation_delta = delta
        self.interpretation_delta_type = delta_type
        return self

    def get_interpretation_neighbors(self, x: float | int) -> tuple[list[float], dict]:
        x = self._round_to_precision(x, self.precision)
        if self.interpretation_delta_type == "percent":
            delta = 1.0 * self.interpretation_delta * x / 100
        elif self.interpretation_delta_type == "absolute":
            delta = self.interpretation_delta
        else:
            delta = self.interpretation_delta
        if self.precision == 0 and math.floor(delta) != delta:
            raise ValueError(
                f"Delta value {delta} is not an integer and precision=0. Cannot generate valid set of neighbors. "
                "If delta_type='percent', pick a value of delta such that x * delta is an integer. "
                "If delta_type='absolute', pick a value of delta that is an integer."
            )
        # run_interpretation will preprocess the neighbors so no need to convert to int here
        negatives = (
            np.array(x) + np.arange(-self.interpretation_steps, 0) * delta
        ).tolist()
        positives = (
            np.array(x) + np.arange(1, self.interpretation_steps + 1) * delta
        ).tolist()
        return negatives + positives, {}

    def get_interpretation_scores(
        self, x: float, neighbors: list[float], scores: list[float | None], **kwargs
    ) -> list[tuple[float, float | None]]:
        """
        Returns:
            Each tuple set represents a numeric value near the input and its corresponding interpretation score.
        """
        interpretation = list(zip(neighbors, scores))
        interpretation.insert(int(len(interpretation) / 2), (x, None))
        return interpretation


@document("style")
class Slider(
    FormComponent,
    Changeable,
    Inputable,
    Releaseable,
    IOComponent,
    NumberSerializable,
    NeighborInterpretable,
):
    """
    Creates a slider that ranges from `minimum` to `maximum` with a step size of `step`.
    Preprocessing: passes slider value as a {float} into the function.
    Postprocessing: expects an {int} or {float} returned from function and sets slider value to it as long as it is within range.
    Examples-format: A {float} or {int} representing the slider's value.

    Demos: sentence_builder, slider_release, generate_tone, titanic_survival, interface_random_slider, blocks_random_slider
    Guides: create-your-own-friends-with-a-gan
    """

    def __init__(
        self,
        minimum: float = 0,
        maximum: float = 100,
        value: float | Callable | None = None,
        *,
        step: float | None = None,
        label: str | None = None,
        info: str | None = None,
        every: float | None = None,
        show_label: bool = True,
        interactive: bool | None = None,
        visible: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        randomize: bool = False,
        **kwargs,
    ):
        """
        Parameters:
            minimum: minimum value for slider.
            maximum: maximum value for slider.
            value: default value. If callable, the function will be called whenever the app loads to set the initial value of the component. Ignored if randomized=True.
            step: increment between slider values.
            label: component name in interface.
            info: additional component description.
            every: If `value` is a callable, run the function 'every' number of seconds while the client connection is open. Has no effect otherwise. Queue must be enabled. The event can be accessed (e.g. to cancel it) via this component's .load_event attribute.
            show_label: if True, will display label.
            interactive: if True, slider will be adjustable; if False, adjusting will be disabled. If not provided, this is inferred based on whether the component is used as an input or output.
            visible: If False, component will be hidden.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.
            randomize: If True, the value of the slider when the app loads is taken uniformly at random from the range given by the minimum and maximum.
        """
        self.minimum = minimum
        self.maximum = maximum
        if step is None:
            difference = maximum - minimum
            power = math.floor(math.log10(difference) - 2)
            self.step = 10**power
        else:
            self.step = step
        if randomize:
            value = self.get_random_value
        IOComponent.__init__(
            self,
            label=label,
            info=info,
            every=every,
            show_label=show_label,
            interactive=interactive,
            visible=visible,
            elem_id=elem_id,
            elem_classes=elem_classes,
            value=value,
            **kwargs,
        )
        NeighborInterpretable.__init__(self)
        self.cleared_value = self.value

    def api_info(self) -> dict[str, dict | bool]:
        return {
            "info": {
                "type": "number",
                "description": f"numeric value between {self.minimum} and {self.maximum}",
            },
            "serialized_info": False,
        }

    def example_inputs(self) -> dict[str, Any]:
        return {
            "raw": self.minimum,
            "serialized": self.minimum,
        }

    def get_config(self):
        return {
            "minimum": self.minimum,
            "maximum": self.maximum,
            "step": self.step,
            "value": self.value,
            **IOComponent.get_config(self),
        }

    def get_random_value(self):
        n_steps = int((self.maximum - self.minimum) / self.step)
        step = random.randint(0, n_steps)
        value = self.minimum + step * self.step
        # Round to number of decimals in step so that UI doesn't display long decimals
        n_decimals = max(str(self.step)[::-1].find("."), 0)
        if n_decimals:
            value = round(value, n_decimals)
        return value

    @staticmethod
    def update(
        value: float | Literal[_Keywords.NO_VALUE] | None = _Keywords.NO_VALUE,
        minimum: float | None = None,
        maximum: float | None = None,
        step: float | None = None,
        label: str | None = None,
        show_label: bool | None = None,
        interactive: bool | None = None,
        visible: bool | None = None,
    ):
        return {
            "minimum": minimum,
            "maximum": maximum,
            "step": step,
            "label": label,
            "show_label": show_label,
            "interactive": interactive,
            "visible": visible,
            "value": value,
            "__type__": "update",
        }

    def postprocess(self, y: float | None) -> float | None:
        """
        Any postprocessing needed to be performed on function output.
        Parameters:
            y: numeric output
        Returns:
            numeric output or minimum number if None
        """
        return self.minimum if y is None else y

    def set_interpret_parameters(self, steps: int = 8) -> Slider:
        """
        Calculates interpretation scores of numeric values ranging between the minimum and maximum values of the slider.
        Parameters:
            steps: Number of neighboring values to measure between the minimum and maximum values of the slider range.
        """
        self.interpretation_steps = steps
        return self

    def get_interpretation_neighbors(self, x) -> tuple[object, dict]:
        return (
            np.linspace(self.minimum, self.maximum, self.interpretation_steps).tolist(),
            {},
        )

    def style(
        self,
        *,
        container: bool | None = None,
    ):
        """
        This method can be used to change the appearance of the slider.
        Parameters:
            container: If True, will place the component in a container - providing some extra padding around the border.
        """
        Component.style(
            self,
            container=container,
        )
        return self


@document("style")
class Checkbox(
    FormComponent,
    Changeable,
    Inputable,
    Selectable,
    IOComponent,
    BooleanSerializable,
    NeighborInterpretable,
):
    """
    Creates a checkbox that can be set to `True` or `False`.

    Preprocessing: passes the status of the checkbox as a {bool} into the function.
    Postprocessing: expects a {bool} returned from the function and, if it is True, checks the checkbox.
    Examples-format: a {bool} representing whether the box is checked.
    Demos: sentence_builder, titanic_survival
    """

    def __init__(
        self,
        value: bool | Callable = False,
        *,
        label: str | None = None,
        info: str | None = None,
        every: float | None = None,
        show_label: bool = True,
        interactive: bool | None = None,
        visible: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        **kwargs,
    ):
        """
        Parameters:
            value: if True, checked by default. If callable, the function will be called whenever the app loads to set the initial value of the component.
            label: component name in interface.
            info: additional component description.
            every: If `value` is a callable, run the function 'every' number of seconds while the client connection is open. Has no effect otherwise. Queue must be enabled. The event can be accessed (e.g. to cancel it) via this component's .load_event attribute.
            show_label: if True, will display label.
            interactive: if True, this checkbox can be checked; if False, checking will be disabled. If not provided, this is inferred based on whether the component is used as an input or output.
            visible: If False, component will be hidden.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.
        """
        self.select: EventListenerMethod
        """
        Event listener for when the user selects or deselects Checkbox.
        Uses event data gradio.SelectData to carry `value` referring to label of checkbox, and `selected` to refer to state of checkbox.
        See EventData documentation on how to use this event data.
        """
        IOComponent.__init__(
            self,
            label=label,
            info=info,
            every=every,
            show_label=show_label,
            interactive=interactive,
            visible=visible,
            elem_id=elem_id,
            elem_classes=elem_classes,
            value=value,
            **kwargs,
        )
        NeighborInterpretable.__init__(self)

    def get_config(self):
        return {
            "value": self.value,
            **IOComponent.get_config(self),
        }

    @staticmethod
    def update(
        value: bool | Literal[_Keywords.NO_VALUE] | None = _Keywords.NO_VALUE,
        label: str | None = None,
        show_label: bool | None = None,
        interactive: bool | None = None,
        visible: bool | None = None,
    ):
        return {
            "label": label,
            "show_label": show_label,
            "interactive": interactive,
            "visible": visible,
            "value": value,
            "__type__": "update",
        }

    def get_interpretation_neighbors(self, x):
        return [not x], {}

    def get_interpretation_scores(self, x, neighbors, scores, **kwargs):
        """
        Returns:
            The first value represents the interpretation score if the input is False, and the second if the input is True.
        """
        if x:
            return scores[0], None
        else:
            return None, scores[0]


@document("style")
class CheckboxGroup(
    FormComponent,
    Changeable,
    Inputable,
    Selectable,
    IOComponent,
    ListStringSerializable,
    NeighborInterpretable,
):
    """
    Creates a set of checkboxes of which a subset can be checked.
    Preprocessing: passes the list of checked checkboxes as a {List[str]} or their indices as a {List[int]} into the function, depending on `type`.
    Postprocessing: expects a {List[str]}, each element of which becomes a checked checkbox.
    Examples-format: a {List[str]} representing the values to be checked.
    Demos: sentence_builder, titanic_survival
    """

    def __init__(
        self,
        choices: list[str] | None = None,
        *,
        value: list[str] | str | Callable | None = None,
        type: str = "value",
        label: str | None = None,
        info: str | None = None,
        every: float | None = None,
        show_label: bool = True,
        interactive: bool | None = None,
        visible: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        **kwargs,
    ):
        """
        Parameters:
            choices: list of options to select from.
            value: default selected list of options. If callable, the function will be called whenever the app loads to set the initial value of the component.
            type: Type of value to be returned by component. "value" returns the list of strings of the choices selected, "index" returns the list of indices of the choices selected.
            label: component name in interface.
            info: additional component description.
            every: If `value` is a callable, run the function 'every' number of seconds while the client connection is open. Has no effect otherwise. Queue must be enabled. The event can be accessed (e.g. to cancel it) via this component's .load_event attribute.
            show_label: if True, will display label.
            interactive: if True, choices in this checkbox group will be checkable; if False, checking will be disabled. If not provided, this is inferred based on whether the component is used as an input or output.
            visible: If False, component will be hidden.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.
        """
        self.choices = choices or []
        self.cleared_value = []
        valid_types = ["value", "index"]
        if type not in valid_types:
            raise ValueError(
                f"Invalid value for parameter `type`: {type}. Please choose from one of: {valid_types}"
            )
        self.type = type
        self.select: EventListenerMethod
        """
        Event listener for when the user selects or deselects within CheckboxGroup.
        Uses event data gradio.SelectData to carry `value` referring to label of selected checkbox, `index` to refer to index, and `selected` to refer to state of checkbox.
        See EventData documentation on how to use this event data.
        """
        IOComponent.__init__(
            self,
            label=label,
            info=info,
            every=every,
            show_label=show_label,
            interactive=interactive,
            visible=visible,
            elem_id=elem_id,
            elem_classes=elem_classes,
            value=value,
            **kwargs,
        )
        NeighborInterpretable.__init__(self)

    def get_config(self):
        return {
            "choices": self.choices,
            "value": self.value,
            **IOComponent.get_config(self),
        }

    def example_inputs(self) -> dict[str, Any]:
        return {
            "raw": self.choices[0] if self.choices else None,
            "serialized": self.choices[0] if self.choices else None,
        }

    @staticmethod
    def update(
        value: list[str]
        | str
        | Literal[_Keywords.NO_VALUE]
        | None = _Keywords.NO_VALUE,
        choices: list[str] | None = None,
        label: str | None = None,
        show_label: bool | None = None,
        interactive: bool | None = None,
        visible: bool | None = None,
    ):
        return {
            "choices": choices,
            "label": label,
            "show_label": show_label,
            "interactive": interactive,
            "visible": visible,
            "value": value,
            "__type__": "update",
        }

    def preprocess(self, x: list[str]) -> list[str] | list[int]:
        """
        Parameters:
            x: list of selected choices
        Returns:
            list of selected choices as strings or indices within choice list
        """
        if self.type == "value":
            return x
        elif self.type == "index":
            return [self.choices.index(choice) for choice in x]
        else:
            raise ValueError(
                f"Unknown type: {self.type}. Please choose from: 'value', 'index'."
            )

    def postprocess(self, y: list[str] | str | None) -> list[str]:
        """
        Any postprocessing needed to be performed on function output.
        Parameters:
            y: List of selected choices. If a single choice is selected, it can be passed in as a string
        Returns:
            List of selected choices
        """
        if y is None:
            return []
        if not isinstance(y, list):
            y = [y]
        return y

    def get_interpretation_neighbors(self, x):
        leave_one_out_sets = []
        for choice in self.choices:
            leave_one_out_set = list(x)
            if choice in leave_one_out_set:
                leave_one_out_set.remove(choice)
            else:
                leave_one_out_set.append(choice)
            leave_one_out_sets.append(leave_one_out_set)
        return leave_one_out_sets, {}

    def get_interpretation_scores(self, x, neighbors, scores, **kwargs):
        """
        Returns:
            For each tuple in the list, the first value represents the interpretation score if the input is False, and the second if the input is True.
        """
        final_scores = []
        for choice, score in zip(self.choices, scores):
            score_set = [score, None] if choice in x else [None, score]
            final_scores.append(score_set)
        return final_scores

    def style(
        self,
        *,
        item_container: bool | None = None,
        container: bool | None = None,
        **kwargs,
    ):
        """
        This method can be used to change the appearance of the CheckboxGroup.
        Parameters:
            item_container: If True, will place the items in a container.
            container: If True, will place the component in a container - providing some extra padding around the border.
        """
        if item_container is not None:
            self._style["item_container"] = item_container

        Component.style(self, container=container, **kwargs)
        return self


@document("style")
class Radio(
    FormComponent,
    Selectable,
    Changeable,
    Inputable,
    IOComponent,
    StringSerializable,
    NeighborInterpretable,
):
    """
    Creates a set of radio buttons of which only one can be selected.
    Preprocessing: passes the value of the selected radio button as a {str} or its index as an {int} into the function, depending on `type`.
    Postprocessing: expects a {str} corresponding to the value of the radio button to be selected.
    Examples-format: a {str} representing the radio option to select.

    Demos: sentence_builder, titanic_survival, blocks_essay
    """

    def __init__(
        self,
        choices: list[str] | None = None,
        *,
        value: str | Callable | None = None,
        type: str = "value",
        label: str | None = None,
        info: str | None = None,
        every: float | None = None,
        show_label: bool = True,
        interactive: bool | None = None,
        visible: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        **kwargs,
    ):
        """
        Parameters:
            choices: list of options to select from.
            value: the button selected by default. If None, no button is selected by default. If callable, the function will be called whenever the app loads to set the initial value of the component.
            type: Type of value to be returned by component. "value" returns the string of the choice selected, "index" returns the index of the choice selected.
            label: component name in interface.
            info: additional component description.
            every: If `value` is a callable, run the function 'every' number of seconds while the client connection is open. Has no effect otherwise. Queue must be enabled. The event can be accessed (e.g. to cancel it) via this component's .load_event attribute.
            show_label: if True, will display label.
            interactive: if True, choices in this radio group will be selectable; if False, selection will be disabled. If not provided, this is inferred based on whether the component is used as an input or output.
            visible: If False, component will be hidden.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.
        """
        self.choices = choices or []
        valid_types = ["value", "index"]
        if type not in valid_types:
            raise ValueError(
                f"Invalid value for parameter `type`: {type}. Please choose from one of: {valid_types}"
            )
        self.type = type
        self.select: EventListenerMethod
        """
        Event listener for when the user selects Radio option.
        Uses event data gradio.SelectData to carry `value` referring to label of selected option, and `index` to refer to index.
        See EventData documentation on how to use this event data.
        """
        IOComponent.__init__(
            self,
            label=label,
            info=info,
            every=every,
            show_label=show_label,
            interactive=interactive,
            visible=visible,
            elem_id=elem_id,
            elem_classes=elem_classes,
            value=value,
            **kwargs,
        )
        NeighborInterpretable.__init__(self)
        self.cleared_value = self.value

    def get_config(self):
        return {
            "choices": self.choices,
            "value": self.value,
            **IOComponent.get_config(self),
        }

    def example_inputs(self) -> dict[str, Any]:
        return {
            "raw": self.choices[0] if self.choices else None,
            "serialized": self.choices[0] if self.choices else None,
        }

    @staticmethod
    def update(
        value: Any | Literal[_Keywords.NO_VALUE] | None = _Keywords.NO_VALUE,
        choices: list[str] | None = None,
        label: str | None = None,
        show_label: bool | None = None,
        interactive: bool | None = None,
        visible: bool | None = None,
    ):
        return {
            "choices": choices,
            "label": label,
            "show_label": show_label,
            "interactive": interactive,
            "visible": visible,
            "value": value,
            "__type__": "update",
        }

    def preprocess(self, x: str | None) -> str | int | None:
        """
        Parameters:
            x: selected choice
        Returns:
            selected choice as string or index within choice list
        """
        if self.type == "value":
            return x
        elif self.type == "index":
            if x is None:
                return None
            else:
                return self.choices.index(x)
        else:
            raise ValueError(
                f"Unknown type: {self.type}. Please choose from: 'value', 'index'."
            )

    def get_interpretation_neighbors(self, x):
        choices = list(self.choices)
        choices.remove(x)
        return choices, {}

    def get_interpretation_scores(
        self, x, neighbors, scores: list[float | None], **kwargs
    ) -> list:
        """
        Returns:
            Each value represents the interpretation score corresponding to each choice.
        """
        scores.insert(self.choices.index(x), None)
        return scores

    def style(
        self,
        *,
        item_container: bool | None = None,
        container: bool | None = None,
        **kwargs,
    ):
        """
        This method can be used to change the appearance of the radio component.
        Parameters:
            item_container: If True, will place items in a container.
            container: If True, will place the component in a container - providing some extra padding around the border.
        """
        if item_container is not None:
            self._style["item_container"] = item_container

        Component.style(self, container=container, **kwargs)
        return self


@document("style")
class Dropdown(
    Changeable,
    Inputable,
    Selectable,
    Blurrable,
    IOComponent,
    SimpleSerializable,
    FormComponent,
):
    """
    Creates a dropdown of choices from which entries can be selected.
    Preprocessing: passes the value of the selected dropdown entry as a {str} or its index as an {int} into the function, depending on `type`.
    Postprocessing: expects a {str} corresponding to the value of the dropdown entry to be selected.
    Examples-format: a {str} representing the drop down value to select.
    Demos: sentence_builder, titanic_survival
    """

    def __init__(
        self,
        choices: list[str] | None = None,
        *,
        value: str | list[str] | Callable | None = None,
        type: str = "value",
        multiselect: bool | None = None,
        max_choices: int | None = None,
        label: str | None = None,
        info: str | None = None,
        every: float | None = None,
        show_label: bool = True,
        interactive: bool | None = None,
        visible: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        allow_custom_value: bool = False,
        **kwargs,
    ):
        """
        Parameters:
            choices: list of options to select from.
            value: default value(s) selected in dropdown. If None, no value is selected by default. If callable, the function will be called whenever the app loads to set the initial value of the component.
            type: Type of value to be returned by component. "value" returns the string of the choice selected, "index" returns the index of the choice selected.
            multiselect: if True, multiple choices can be selected.
            max_choices: maximum number of choices that can be selected. If None, no limit is enforced.
            label: component name in interface.
            info: additional component description.
            every: If `value` is a callable, run the function 'every' number of seconds while the client connection is open. Has no effect otherwise. Queue must be enabled. The event can be accessed (e.g. to cancel it) via this component's .load_event attribute.
            show_label: if True, will display label.
            interactive: if True, choices in this dropdown will be selectable; if False, selection will be disabled. If not provided, this is inferred based on whether the component is used as an input or output.
            visible: If False, component will be hidden.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.
            allow_custom_value: If True, allows user to enter a custom value that is not in the list of choices.
        """
        self.choices = [str(choice) for choice in choices] if choices else []
        valid_types = ["value", "index"]
        if type not in valid_types:
            raise ValueError(
                f"Invalid value for parameter `type`: {type}. Please choose from one of: {valid_types}"
            )
        self.type = type
        self.multiselect = multiselect
        if multiselect and isinstance(value, str):
            value = [value]
        if not multiselect and max_choices is not None:
            warnings.warn(
                "The `max_choices` parameter is ignored when `multiselect` is False."
            )
        self.max_choices = max_choices
        self.allow_custom_value = allow_custom_value
        if multiselect and allow_custom_value:
            raise ValueError(
                "Custom values are not supported when `multiselect` is True."
            )
        self.interpret_by_tokens = False
        self.select: EventListenerMethod
        """
        Event listener for when the user selects Dropdown option.
        Uses event data gradio.SelectData to carry `value` referring to label of selected option, and `index` to refer to index.
        See EventData documentation on how to use this event data.
        """
        IOComponent.__init__(
            self,
            label=label,
            info=info,
            every=every,
            show_label=show_label,
            interactive=interactive,
            visible=visible,
            elem_id=elem_id,
            elem_classes=elem_classes,
            value=value,
            **kwargs,
        )

        self.cleared_value = self.value or ([] if multiselect else "")

    def api_info(self) -> dict[str, dict | bool]:
        if self.multiselect:
            type = {
                "type": "array",
                "items": {"type": "string"},
                "description": f"List of options from: {self.choices}",
            }
        else:
            type = {"type": "string", "description": f"Option from: {self.choices}"}
        return {"info": type, "serialized_info": False}

    def example_inputs(self) -> dict[str, Any]:
        if self.multiselect:
            return {
                "raw": [self.choices[0]] if self.choices else [],
                "serialized": [self.choices[0]] if self.choices else [],
            }
        else:
            return {
                "raw": self.choices[0] if self.choices else None,
                "serialized": self.choices[0] if self.choices else None,
            }

    def get_config(self):
        return {
            "choices": self.choices,
            "value": self.value,
            "multiselect": self.multiselect,
            "max_choices": self.max_choices,
            "allow_custom_value": self.allow_custom_value,
            **IOComponent.get_config(self),
        }

    @staticmethod
    def update(
        value: Any | Literal[_Keywords.NO_VALUE] | None = _Keywords.NO_VALUE,
        choices: str | list[str] | None = None,
        label: str | None = None,
        show_label: bool | None = None,
        interactive: bool | None = None,
        placeholder: str | None = None,
        visible: bool | None = None,
    ):
        return {
            "choices": choices,
            "label": label,
            "show_label": show_label,
            "visible": visible,
            "value": value,
            "interactive": interactive,
            "placeholder": placeholder,
            "__type__": "update",
        }

    def preprocess(
        self, x: str | list[str]
    ) -> str | int | list[str] | list[int] | None:
        """
        Parameters:
            x: selected choice(s)
        Returns:
            selected choice(s) as string or index within choice list or list of string or indices
        """
        if self.type == "value":
            return x
        elif self.type == "index":
            if x is None:
                return None
            elif self.multiselect:
                return [self.choices.index(c) for c in x]
            else:
                if isinstance(x, str):
                    return self.choices.index(x) if x in self.choices else None
        else:
            raise ValueError(
                f"Unknown type: {self.type}. Please choose from: 'value', 'index'."
            )

    def set_interpret_parameters(self):
        """
        Calculates interpretation score of each choice by comparing the output against each of the outputs when alternative choices are selected.
        """
        return self

    def get_interpretation_neighbors(self, x):
        choices = list(self.choices)
        choices.remove(x)
        return choices, {}

    def get_interpretation_scores(
        self, x, neighbors, scores: list[float | None], **kwargs
    ) -> list:
        """
        Returns:
            Each value represents the interpretation score corresponding to each choice.
        """
        scores.insert(self.choices.index(x), None)
        return scores

    def style(self, *, container: bool | None = None, **kwargs):
        """
        This method can be used to change the appearance of the Dropdown.
        Parameters:
            container: If True, will place the component in a container - providing some extra padding around the border.
        """
        Component.style(self, container=container, **kwargs)
        return self


@document("style")
class Image(
    Editable,
    Clearable,
    Changeable,
    Streamable,
    Selectable,
    Uploadable,
    IOComponent,
    ImgSerializable,
    TokenInterpretable,
):
    """
    Creates an image component that can be used to upload/draw images (as an input) or display images (as an output).
    Preprocessing: passes the uploaded image as a {numpy.array}, {PIL.Image} or {str} filepath depending on `type` -- unless `tool` is `sketch` AND source is one of `upload` or `webcam`. In these cases, a {dict} with keys `image` and `mask` is passed, and the format of the corresponding values depends on `type`.
    Postprocessing: expects a {numpy.array}, {PIL.Image} or {str} or {pathlib.Path} filepath to an image and displays the image.
    Examples-format: a {str} filepath to a local file that contains the image.
    Demos: image_mod, image_mod_default_image
    Guides: image-classification-in-pytorch, image-classification-in-tensorflow, image-classification-with-vision-transformers, building-a-pictionary_app, create-your-own-friends-with-a-gan
    """

    def __init__(
        self,
        value: str | _Image.Image | np.ndarray | None = None,
        *,
        shape: tuple[int, int] | None = None,
        image_mode: str = "RGB",
        invert_colors: bool = False,
        source: str = "upload",
        tool: str | None = None,
        type: str = "numpy",
        label: str | None = None,
        every: float | None = None,
        show_label: bool = True,
        interactive: bool | None = None,
        visible: bool = True,
        streaming: bool = False,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        mirror_webcam: bool = True,
        brush_radius: float | None = None,
        **kwargs,
    ):
        """
        Parameters:
            value: A PIL Image, numpy array, path or URL for the default value that Image component is going to take. If callable, the function will be called whenever the app loads to set the initial value of the component.
            shape: (width, height) shape to crop and resize image to; if None, matches input image size. Pass None for either width or height to only crop and resize the other.
            image_mode: "RGB" if color, or "L" if black and white.
            invert_colors: whether to invert the image as a preprocessing step.
            source: Source of image. "upload" creates a box where user can drop an image file, "webcam" allows user to take snapshot from their webcam, "canvas" defaults to a white image that can be edited and drawn upon with tools.
            tool: Tools used for editing. "editor" allows a full screen editor (and is the default if source is "upload" or "webcam"), "select" provides a cropping and zoom tool, "sketch" allows you to create a binary sketch (and is the default if source="canvas"), and "color-sketch" allows you to created a sketch in different colors. "color-sketch" can be used with source="upload" or "webcam" to allow sketching on an image. "sketch" can also be used with "upload" or "webcam" to create a mask over an image and in that case both the image and mask are passed into the function as a dictionary with keys "image" and "mask" respectively.
            type: The format the image is converted to before being passed into the prediction function. "numpy" converts the image to a numpy array with shape (height, width, 3) and values from 0 to 255, "pil" converts the image to a PIL image object, "filepath" passes a str path to a temporary file containing the image.
            label: component name in interface.
            every: If `value` is a callable, run the function 'every' number of seconds while the client connection is open. Has no effect otherwise. Queue must be enabled. The event can be accessed (e.g. to cancel it) via this component's .load_event attribute.
            show_label: if True, will display label.
            interactive: if True, will allow users to upload and edit an image; if False, can only be used to display images. If not provided, this is inferred based on whether the component is used as an input or output.
            visible: If False, component will be hidden.
            streaming: If True when used in a `live` interface, will automatically stream webcam feed. Only valid is source is 'webcam'.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.
            mirror_webcam: If True webcam will be mirrored. Default is True.
            brush_radius: Size of the brush for Sketch. Default is None which chooses a sensible default
        """
        self.brush_radius = brush_radius
        self.mirror_webcam = mirror_webcam
        valid_types = ["numpy", "pil", "filepath"]
        if type not in valid_types:
            raise ValueError(
                f"Invalid value for parameter `type`: {type}. Please choose from one of: {valid_types}"
            )
        self.type = type
        self.shape = shape
        self.image_mode = image_mode
        valid_sources = ["upload", "webcam", "canvas"]
        if source not in valid_sources:
            raise ValueError(
                f"Invalid value for parameter `source`: {source}. Please choose from one of: {valid_sources}"
            )
        self.source = source
        if tool is None:
            self.tool = "sketch" if source == "canvas" else "editor"
        else:
            self.tool = tool
        self.invert_colors = invert_colors
        self.streaming = streaming
        if streaming and source != "webcam":
            raise ValueError("Image streaming only available if source is 'webcam'.")
        self.select: EventListenerMethod
        """
        Event listener for when the user clicks on a pixel within the image.
        Uses event data gradio.SelectData to carry `index` to refer to the [x, y] coordinates of the clicked pixel.
        See EventData documentation on how to use this event data.
        """

        IOComponent.__init__(
            self,
            label=label,
            every=every,
            show_label=show_label,
            interactive=interactive,
            visible=visible,
            elem_id=elem_id,
            elem_classes=elem_classes,
            value=value,
            **kwargs,
        )
        TokenInterpretable.__init__(self)

    def get_config(self):
        return {
            "image_mode": self.image_mode,
            "shape": self.shape,
            "source": self.source,
            "tool": self.tool,
            "value": self.value,
            "streaming": self.streaming,
            "mirror_webcam": self.mirror_webcam,
            "brush_radius": self.brush_radius,
            "selectable": self.selectable,
            **IOComponent.get_config(self),
        }

    @staticmethod
    def update(
        value: Any | Literal[_Keywords.NO_VALUE] | None = _Keywords.NO_VALUE,
        label: str | None = None,
        show_label: bool | None = None,
        interactive: bool | None = None,
        visible: bool | None = None,
        brush_radius: float | None = None,
    ):
        return {
            "label": label,
            "show_label": show_label,
            "interactive": interactive,
            "visible": visible,
            "value": value,
            "brush_radius": brush_radius,
            "__type__": "update",
        }

    def _format_image(
        self, im: _Image.Image | None
    ) -> np.ndarray | _Image.Image | str | None:
        """Helper method to format an image based on self.type"""
        if im is None:
            return im
        fmt = im.format
        if self.type == "pil":
            return im
        elif self.type == "numpy":
            return np.array(im)
        elif self.type == "filepath":
            path = self.pil_to_temp_file(
                im, dir=self.DEFAULT_TEMP_DIR, format=fmt or "png"
            )
            self.temp_files.add(path)
            return path
        else:
            raise ValueError(
                "Unknown type: "
                + str(self.type)
                + ". Please choose from: 'numpy', 'pil', 'filepath'."
            )

    def preprocess(
        self, x: str | dict[str, str]
    ) -> np.ndarray | _Image.Image | str | dict | None:
        """
        Parameters:
            x: base64 url data, or (if tool == "sketch") a dict of image and mask base64 url data
        Returns:
            image in requested format, or (if tool == "sketch") a dict of image and mask in requested format
        """
        if x is None:
            return x

        mask = ""
        if self.tool == "sketch" and self.source in ["upload", "webcam"]:
            assert isinstance(x, dict)
            x, mask = x["image"], x["mask"]

        assert isinstance(x, str)
        im = processing_utils.decode_base64_to_image(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            im = im.convert(self.image_mode)
        if self.shape is not None:
            im = processing_utils.resize_and_crop(im, self.shape)
        if self.invert_colors:
            im = PIL.ImageOps.invert(im)
        if (
            self.source == "webcam"
            and self.mirror_webcam is True
            and self.tool != "color-sketch"
        ):
            im = PIL.ImageOps.mirror(im)

        if self.tool == "sketch" and self.source in ["upload", "webcam"]:
            mask_im = processing_utils.decode_base64_to_image(mask)
            return {
                "image": self._format_image(im),
                "mask": self._format_image(mask_im),
            }

        return self._format_image(im)

    def postprocess(
        self, y: np.ndarray | _Image.Image | str | Path | None
    ) -> str | None:
        """
        Parameters:
            y: image as a numpy array, PIL Image, string/Path filepath, or string URL
        Returns:
            base64 url data
        """
        if y is None:
            return None
        if isinstance(y, np.ndarray):
            return processing_utils.encode_array_to_base64(y)
        elif isinstance(y, _Image.Image):
            return processing_utils.encode_pil_to_base64(y)
        elif isinstance(y, (str, Path)):
            return client_utils.encode_url_or_file_to_base64(y)
        else:
            raise ValueError("Cannot process this value as an Image")

    def set_interpret_parameters(self, segments: int = 16):
        """
        Calculates interpretation score of image subsections by splitting the image into subsections, then using a "leave one out" method to calculate the score of each subsection by whiting out the subsection and measuring the delta of the output value.
        Parameters:
            segments: Number of interpretation segments to split image into.
        """
        self.interpretation_segments = segments
        return self

    def _segment_by_slic(self, x):
        """
        Helper method that segments an image into superpixels using slic.
        Parameters:
            x: base64 representation of an image
        """
        x = processing_utils.decode_base64_to_image(x)
        if self.shape is not None:
            x = processing_utils.resize_and_crop(x, self.shape)
        resized_and_cropped_image = np.array(x)
        try:
            from skimage.segmentation import slic
        except (ImportError, ModuleNotFoundError) as err:
            raise ValueError(
                "Error: running this interpretation for images requires scikit-image, please install it first."
            ) from err
        try:
            segments_slic = slic(
                resized_and_cropped_image,
                self.interpretation_segments,
                compactness=10,
                sigma=1,
                start_label=1,
            )
        except TypeError:  # For skimage 0.16 and older
            segments_slic = slic(
                resized_and_cropped_image,
                self.interpretation_segments,
                compactness=10,
                sigma=1,
            )
        return segments_slic, resized_and_cropped_image

    def tokenize(self, x):
        """
        Segments image into tokens, masks, and leave-one-out-tokens
        Parameters:
            x: base64 representation of an image
        Returns:
            tokens: list of tokens, used by the get_masked_input() method
            leave_one_out_tokens: list of left-out tokens, used by the get_interpretation_neighbors() method
            masks: list of masks, used by the get_interpretation_neighbors() method
        """
        segments_slic, resized_and_cropped_image = self._segment_by_slic(x)
        tokens, masks, leave_one_out_tokens = [], [], []
        replace_color = np.mean(resized_and_cropped_image, axis=(0, 1))
        for segment_value in np.unique(segments_slic):
            mask = segments_slic == segment_value
            image_screen = np.copy(resized_and_cropped_image)
            image_screen[segments_slic == segment_value] = replace_color
            leave_one_out_tokens.append(
                processing_utils.encode_array_to_base64(image_screen)
            )
            token = np.copy(resized_and_cropped_image)
            token[segments_slic != segment_value] = 0
            tokens.append(token)
            masks.append(mask)
        return tokens, leave_one_out_tokens, masks

    def get_masked_inputs(self, tokens, binary_mask_matrix):
        masked_inputs = []
        for binary_mask_vector in binary_mask_matrix:
            masked_input = np.zeros_like(tokens[0], dtype=int)
            for token, b in zip(tokens, binary_mask_vector):
                masked_input = masked_input + token * int(b)
            masked_inputs.append(processing_utils.encode_array_to_base64(masked_input))
        return masked_inputs

    def get_interpretation_scores(
        self, x, neighbors, scores, masks, tokens=None, **kwargs
    ) -> list[list[float]]:
        """
        Returns:
            A 2D array representing the interpretation score of each pixel of the image.
        """
        x = processing_utils.decode_base64_to_image(x)
        if self.shape is not None:
            x = processing_utils.resize_and_crop(x, self.shape)
        x = np.array(x)
        output_scores = np.zeros((x.shape[0], x.shape[1]))

        for score, mask in zip(scores, masks):
            output_scores += score * mask

        max_val, min_val = np.max(output_scores), np.min(output_scores)
        if max_val > 0:
            output_scores = (output_scores - min_val) / (max_val - min_val)
        return output_scores.tolist()

    def style(self, *, height: int | None = None, width: int | None = None, **kwargs):
        """
        This method can be used to change the appearance of the Image component.
        Parameters:
            height: Height of the image.
            width: Width of the image.
        """
        self._style["height"] = height
        self._style["width"] = width
        Component.style(
            self,
            **kwargs,
        )
        return self

    def check_streamable(self):
        if self.source != "webcam":
            raise ValueError("Image streaming only available if source is 'webcam'.")

    def as_example(self, input_data: str | None) -> str:
        if input_data is None:
            return ""
        elif (
            self.root_url
        ):  # If an externally hosted image, don't convert to absolute path
            return input_data
        return str(utils.abspath(input_data))


@document("style")
class Video(
    Changeable,
    Clearable,
    Playable,
    Uploadable,
    IOComponent,
    VideoSerializable,
):
    """
    Creates a video component that can be used to upload/record videos (as an input) or display videos (as an output).
    For the video to be playable in the browser it must have a compatible container and codec combination. Allowed
    combinations are .mp4 with h264 codec, .ogg with theora codec, and .webm with vp9 codec. If the component detects
    that the output video would not be playable in the browser it will attempt to convert it to a playable mp4 video.
    If the conversion fails, the original video is returned.
    Preprocessing: passes the uploaded video as a {str} filepath or URL whose extension can be modified by `format`.
    Postprocessing: expects a {str} filepath to a video which is displayed, or a {Tuple[str, str]} where the first element is a filepath to a video and the second element is a filepath to a subtitle file.
    Examples-format: a {str} filepath to a local file that contains the video, or a {Tuple[str, str]} where the first element is a filepath to a video file and the second element is a filepath to a subtitle file.
    Demos: video_identity, video_subtitle
    """

    def __init__(
        self,
        value: str | tuple[str, str | None] | Callable | None = None,
        *,
        format: str | None = None,
        source: str = "upload",
        label: str | None = None,
        every: float | None = None,
        show_label: bool = True,
        interactive: bool | None = None,
        visible: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        mirror_webcam: bool = True,
        include_audio: bool | None = None,
        **kwargs,
    ):
        """
        Parameters:
            value: A path or URL for the default value that Video component is going to take. Can also be a tuple consisting of (video filepath, subtitle filepath). If a subtitle file is provided, it should be of type .srt or .vtt. Or can be callable, in which case the function will be called whenever the app loads to set the initial value of the component.
            format: Format of video format to be returned by component, such as 'avi' or 'mp4'. Use 'mp4' to ensure browser playability. If set to None, video will keep uploaded format.
            source: Source of video. "upload" creates a box where user can drop an video file, "webcam" allows user to record a video from their webcam.
            label: component name in interface.
            every: If `value` is a callable, run the function 'every' number of seconds while the client connection is open. Has no effect otherwise. Queue must be enabled. The event can be accessed (e.g. to cancel it) via this component's .load_event attribute.
            show_label: if True, will display label.
            interactive: if True, will allow users to upload a video; if False, can only be used to display videos. If not provided, this is inferred based on whether the component is used as an input or output.
            visible: If False, component will be hidden.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.
            mirror_webcam: If True webcam will be mirrored. Default is True.
            include_audio: Whether the component should record/retain the audio track for a video. By default, audio is excluded for webcam videos and included for uploaded videos.
        """
        self.format = format
        valid_sources = ["upload", "webcam"]
        if source not in valid_sources:
            raise ValueError(
                f"Invalid value for parameter `source`: {source}. Please choose from one of: {valid_sources}"
            )
        self.source = source
        self.mirror_webcam = mirror_webcam
        self.include_audio = (
            include_audio if include_audio is not None else source == "upload"
        )
        IOComponent.__init__(
            self,
            label=label,
            every=every,
            show_label=show_label,
            interactive=interactive,
            visible=visible,
            elem_id=elem_id,
            elem_classes=elem_classes,
            value=value,
            **kwargs,
        )

    def get_config(self):
        return {
            "source": self.source,
            "value": self.value,
            "mirror_webcam": self.mirror_webcam,
            "include_audio": self.include_audio,
            **IOComponent.get_config(self),
        }

    @staticmethod
    def update(
        value: str
        | tuple[str, str | None]
        | Literal[_Keywords.NO_VALUE]
        | None = _Keywords.NO_VALUE,
        source: str | None = None,
        label: str | None = None,
        show_label: bool | None = None,
        interactive: bool | None = None,
        visible: bool | None = None,
    ):
        return {
            "source": source,
            "label": label,
            "show_label": show_label,
            "interactive": interactive,
            "visible": visible,
            "value": value,
            "__type__": "update",
        }

    def preprocess(
        self, x: tuple[FileData, FileData | None] | FileData | None
    ) -> str | None:
        """
        Parameters:
            x: A tuple of (video file data, subtitle file data) or just video file data.
        Returns:
            A string file path or URL to the preprocessed video. Subtitle file data is ignored.
        """
        if x is None:
            return None
        elif isinstance(x, dict):
            video = x
        else:
            video = x[0]

        file_name, file_data, is_file = (
            video.get("name"),
            video["data"],
            video.get("is_file", False),
        )

        if is_file:
            assert file_name is not None, "Received file data without a file name."
            file_name = Path(self.make_temp_copy_if_needed(file_name))
        else:
            assert file_data is not None, "Received empty file data."
            file_name = Path(self.base64_to_temp_file_if_needed(file_data, file_name))

        uploaded_format = file_name.suffix.replace(".", "")
        needs_formatting = self.format is not None and uploaded_format != self.format
        flip = self.source == "webcam" and self.mirror_webcam

        if needs_formatting or flip:
            format = f".{self.format if needs_formatting else uploaded_format}"
            output_options = ["-vf", "hflip", "-c:a", "copy"] if flip else []
            output_options += ["-an"] if not self.include_audio else []
            flip_suffix = "_flip" if flip else ""
            output_file_name = str(
                file_name.with_name(f"{file_name.stem}{flip_suffix}{format}")
            )
            if Path(output_file_name).exists():
                return output_file_name
            ff = FFmpeg(
                inputs={str(file_name): None},
                outputs={output_file_name: output_options},
            )
            ff.run()
            return output_file_name
        elif not self.include_audio:
            output_file_name = str(file_name.with_name(f"muted_{file_name.name}"))
            ff = FFmpeg(
                inputs={str(file_name): None},
                outputs={output_file_name: ["-an"]},
            )
            ff.run()
            return output_file_name
        else:
            return str(file_name)

    def postprocess(
        self, y: str | tuple[str, str | None] | None
    ) -> tuple[FileData | None, FileData | None] | None:
        """
        Processes a video to ensure that it is in the correct format before
        returning it to the front end.
        Parameters:
            y: video data in either of the following formats: a tuple of (str video filepath, str subtitle filepath), or a string filepath or URL to an video file, or None.
        Returns:
            a tuple with the two dictionary, reresent to video and (optional) subtitle, which following formats:
            - The first dictionary represents the video file and contains the following keys:
                - 'name': a file path to a temporary copy of the processed video.
                - 'data': None
                - 'is_file': True
            - The second dictionary represents the subtitle file and contains the following keys:
                - 'name': None
                - 'data': Base64 encode the processed subtitle data.
                - 'is_file': False
            - If subtitle is None, returns (video, None).
            - If both video and subtitle are None, returns None.
        """

        if y is None or y == [None, None] or y == (None, None):
            return None
        if isinstance(y, str):
            processed_files = (self._format_video(y), None)
        elif isinstance(y, (tuple, list)):
            assert (
                len(y) == 2
            ), f"Expected lists of length 2 or tuples of length 2. Received: {y}"
            video = y[0]
            subtitle = y[1]
            processed_files = (
                self._format_video(video),
                self._format_subtitle(subtitle),
            )
        else:
            raise Exception(f"Cannot process type as video: {type(y)}")

        return processed_files

    def _format_video(self, video: str | None) -> FileData | None:
        """
        Processes a video to ensure that it is in the correct format.
        Parameters:
            video: video data in either of the following formats: a string filepath or URL to an video file, or None.
        Returns:
            a dictionary with the following keys:

            - 'name': a file path to a temporary copy of the processed video.
            - 'data': None
            - 'is_file': True
        """
        if video is None:
            return None

        returned_format = video.split(".")[-1].lower()

        if self.format is None or returned_format == self.format:
            conversion_needed = False
        else:
            conversion_needed = True

        # For cases where the video is a URL and does not need to be converted to another format, we can just return the URL
        if utils.validate_url(video) and not (conversion_needed):
            return {"name": video, "data": None, "is_file": True}

        # For cases where the video needs to be converted to another format
        if utils.validate_url(video):
            video = self.download_temp_copy_if_needed(video)
        if (
            processing_utils.ffmpeg_installed()
            and not processing_utils.video_is_playable(video)
        ):
            warnings.warn(
                "Video does not have browser-compatible container or codec. Converting to mp4"
            )
            video = processing_utils.convert_video_to_playable_mp4(video)
        if self.format is not None and returned_format != self.format:
            output_file_name = video[0 : video.rindex(".") + 1] + self.format
            ff = FFmpeg(inputs={video: None}, outputs={output_file_name: None})
            ff.run()
            video = output_file_name

        video = self.make_temp_copy_if_needed(video)

        return {
            "name": video,
            "data": None,
            "is_file": True,
            "orig_name": Path(video).name,
        }

    def _format_subtitle(self, subtitle: str | None) -> FileData | None:
        """
        Convert subtitle format to VTT and process the video to ensure it meets the HTML5 requirements.
        Parameters:
            subtitle: subtitle path in either of the VTT and SRT format.
        Returns:
            a dictionary with the following keys:
            - 'name': None
            - 'data': base64-encoded subtitle data.
            - 'is_file': False
        """

        def srt_to_vtt(srt_file_path, vtt_file_path):
            """Convert an SRT subtitle file to a VTT subtitle file"""
            with open(srt_file_path, encoding="utf-8") as srt_file, open(
                vtt_file_path, "w", encoding="utf-8"
            ) as vtt_file:
                vtt_file.write("WEBVTT\n\n")
                for subtitle_block in srt_file.read().strip().split("\n\n"):
                    subtitle_lines = subtitle_block.split("\n")
                    subtitle_timing = subtitle_lines[1].replace(",", ".")
                    subtitle_text = "\n".join(subtitle_lines[2:])
                    vtt_file.write(f"{subtitle_timing} --> {subtitle_timing}\n")
                    vtt_file.write(f"{subtitle_text}\n\n")

        if subtitle is None:
            return None

        valid_extensions = (".srt", ".vtt")

        if Path(subtitle).suffix not in valid_extensions:
            raise ValueError(
                f"Invalid value for parameter `subtitle`: {subtitle}. Please choose a file with one of these extensions: {valid_extensions}"
            )

        # HTML5 only support vtt format
        if Path(subtitle).suffix == ".srt":
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, suffix=".vtt", dir=self.DEFAULT_TEMP_DIR
            )

            srt_to_vtt(subtitle, temp_file.name)
            subtitle = temp_file.name

        subtitle_data = client_utils.encode_url_or_file_to_base64(subtitle)
        return {"name": None, "data": subtitle_data, "is_file": False}

    def style(self, *, height: int | None = None, width: int | None = None, **kwargs):
        """
        This method can be used to change the appearance of the video component.
        Parameters:
            height: Height of the video.
            width: Width of the video.
        """
        self._style["height"] = height
        self._style["width"] = width
        Component.style(
            self,
            **kwargs,
        )
        return self


@document("style")
class Audio(
    Changeable,
    Clearable,
    Playable,
    Streamable,
    Uploadable,
    IOComponent,
    FileSerializable,
    TokenInterpretable,
):
    """
    Creates an audio component that can be used to upload/record audio (as an input) or display audio (as an output).
    Preprocessing: passes the uploaded audio as a {Tuple(int, numpy.array)} corresponding to (sample rate in Hz, audio data as a 16-bit int array whose values range from -32768 to 32767), or as a {str} filepath, depending on `type`.
    Postprocessing: expects a {Tuple(int, numpy.array)} corresponding to (sample rate in Hz, audio data as a float or int numpy array) or as a {str} filepath or URL to an audio file, which gets displayed
    Examples-format: a {str} filepath to a local file that contains audio.
    Demos: main_note, generate_tone, reverse_audio
    Guides: real-time-speech-recognition
    """

    def __init__(
        self,
        value: str | tuple[int, np.ndarray] | Callable | None = None,
        *,
        source: str = "upload",
        type: str = "numpy",
        label: str | None = None,
        every: float | None = None,
        show_label: bool = True,
        interactive: bool | None = None,
        visible: bool = True,
        streaming: bool = False,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        format: Literal["wav", "mp3"] = "wav",
        **kwargs,
    ):
        """
        Parameters:
            value: A path, URL, or [sample_rate, numpy array] tuple (sample rate in Hz, audio data as a float or int numpy array) for the default value that Audio component is going to take. If callable, the function will be called whenever the app loads to set the initial value of the component.
            source: Source of audio. "upload" creates a box where user can drop an audio file, "microphone" creates a microphone input.
            type: The format the audio file is converted to before being passed into the prediction function. "numpy" converts the audio to a tuple consisting of: (int sample rate, numpy.array for the data), "filepath" passes a str path to a temporary file containing the audio.
            label: component name in interface.
            every: If `value` is a callable, run the function 'every' number of seconds while the client connection is open. Has no effect otherwise. Queue must be enabled. The event can be accessed (e.g. to cancel it) via this component's .load_event attribute.
            show_label: if True, will display label.
            interactive: if True, will allow users to upload and edit a audio file; if False, can only be used to play audio. If not provided, this is inferred based on whether the component is used as an input or output.
            visible: If False, component will be hidden.
            streaming: If set to True when used in a `live` interface, will automatically stream webcam feed. Only valid is source is 'microphone'.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.
            format: The file format to save audio files. Either 'wav' or 'mp3'. wav files are lossless but will tend to be larger files. mp3 files tend to be smaller. Default is wav. Applies both when this component is used as an input (when `type` is "format") and when this component is used as an output.
        """
        valid_sources = ["upload", "microphone"]
        if source not in valid_sources:
            raise ValueError(
                f"Invalid value for parameter `source`: {source}. Please choose from one of: {valid_sources}"
            )
        self.source = source
        valid_types = ["numpy", "filepath"]
        if type not in valid_types:
            raise ValueError(
                f"Invalid value for parameter `type`: {type}. Please choose from one of: {valid_types}"
            )
        self.type = type
        self.streaming = streaming
        if streaming and source != "microphone":
            raise ValueError(
                "Audio streaming only available if source is 'microphone'."
            )
        IOComponent.__init__(
            self,
            label=label,
            every=every,
            show_label=show_label,
            interactive=interactive,
            visible=visible,
            elem_id=elem_id,
            elem_classes=elem_classes,
            value=value,
            **kwargs,
        )
        TokenInterpretable.__init__(self)
        self.format = format

    def get_config(self):
        return {
            "source": self.source,
            "value": self.value,
            "streaming": self.streaming,
            **IOComponent.get_config(self),
        }

    def example_inputs(self) -> dict[str, Any]:
        return {
            "raw": {"is_file": False, "data": media_data.BASE64_AUDIO},
            "serialized": "https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav",
        }

    @staticmethod
    def update(
        value: Any | Literal[_Keywords.NO_VALUE] | None = _Keywords.NO_VALUE,
        source: str | None = None,
        label: str | None = None,
        show_label: bool | None = None,
        interactive: bool | None = None,
        visible: bool | None = None,
    ):
        return {
            "source": source,
            "label": label,
            "show_label": show_label,
            "interactive": interactive,
            "visible": visible,
            "value": value,
            "__type__": "update",
        }

    def preprocess(
        self, x: dict[str, Any] | None
    ) -> tuple[int, np.ndarray] | str | None:
        """
        Parameters:
            x: dictionary with keys "name", "data", "is_file", "crop_min", "crop_max".
        Returns:
            audio in requested format
        """
        if x is None:
            return x
        file_name, file_data, is_file = (
            x["name"],
            x["data"],
            x.get("is_file", False),
        )
        crop_min, crop_max = x.get("crop_min", 0), x.get("crop_max", 100)
        if is_file:
            if utils.validate_url(file_name):
                temp_file_path = self.download_temp_copy_if_needed(file_name)
            else:
                temp_file_path = self.make_temp_copy_if_needed(file_name)
        else:
            temp_file_path = self.base64_to_temp_file_if_needed(file_data, file_name)

        sample_rate, data = processing_utils.audio_from_file(
            temp_file_path, crop_min=crop_min, crop_max=crop_max
        )

        # Need a unique name for the file to avoid re-using the same audio file if
        # a user submits the same audio file twice, but with different crop min/max.
        temp_file_path = Path(temp_file_path)
        output_file_name = str(
            temp_file_path.with_name(
                f"{temp_file_path.stem}-{crop_min}-{crop_max}{temp_file_path.suffix}"
            )
        )

        if self.type == "numpy":
            return sample_rate, data
        elif self.type == "filepath":
            output_file = str(Path(output_file_name).with_suffix(f".{self.format}"))
            processing_utils.audio_to_file(
                sample_rate, data, output_file, format=self.format
            )
            return output_file
        else:
            raise ValueError(
                "Unknown type: "
                + str(self.type)
                + ". Please choose from: 'numpy', 'filepath'."
            )

    def set_interpret_parameters(self, segments: int = 8):
        """
        Calculates interpretation score of audio subsections by splitting the audio into subsections, then using a "leave one out" method to calculate the score of each subsection by removing the subsection and measuring the delta of the output value.
        Parameters:
            segments: Number of interpretation segments to split audio into.
        """
        self.interpretation_segments = segments
        return self

    def tokenize(self, x):
        if x.get("is_file"):
            sample_rate, data = processing_utils.audio_from_file(x["name"])
        else:
            file_name = self.base64_to_temp_file_if_needed(x["data"])
            sample_rate, data = processing_utils.audio_from_file(file_name)
        leave_one_out_sets = []
        tokens = []
        masks = []
        duration = data.shape[0]
        boundaries = np.linspace(0, duration, self.interpretation_segments + 1).tolist()
        boundaries = [round(boundary) for boundary in boundaries]
        for index in range(len(boundaries) - 1):
            start, stop = boundaries[index], boundaries[index + 1]
            masks.append((start, stop))

            # Handle the leave one outs
            leave_one_out_data = np.copy(data)
            leave_one_out_data[start:stop] = 0
            file = tempfile.NamedTemporaryFile(
                delete=False, suffix=".wav", dir=self.DEFAULT_TEMP_DIR
            )
            processing_utils.audio_to_file(sample_rate, leave_one_out_data, file.name)
            out_data = client_utils.encode_file_to_base64(file.name)
            leave_one_out_sets.append(out_data)
            file.close()
            Path(file.name).unlink()

            # Handle the tokens
            token = np.copy(data)
            token[0:start] = 0
            token[stop:] = 0
            file = tempfile.NamedTemporaryFile(
                delete=False, suffix=".wav", dir=self.DEFAULT_TEMP_DIR
            )
            processing_utils.audio_to_file(sample_rate, token, file.name)
            token_data = client_utils.encode_file_to_base64(file.name)
            file.close()
            Path(file.name).unlink()

            tokens.append(token_data)
        tokens = [{"name": "token.wav", "data": token} for token in tokens]
        leave_one_out_sets = [
            {"name": "loo.wav", "data": loo_set} for loo_set in leave_one_out_sets
        ]
        return tokens, leave_one_out_sets, masks

    def get_masked_inputs(self, tokens, binary_mask_matrix):
        # create a "zero input" vector and get sample rate
        x = tokens[0]["data"]
        file_name = self.base64_to_temp_file_if_needed(x)
        sample_rate, data = processing_utils.audio_from_file(file_name)
        zero_input = np.zeros_like(data, dtype="int16")
        # decode all of the tokens
        token_data = []
        for token in tokens:
            file_name = self.base64_to_temp_file_if_needed(token["data"])
            _, data = processing_utils.audio_from_file(file_name)
            token_data.append(data)
        # construct the masked version
        masked_inputs = []
        for binary_mask_vector in binary_mask_matrix:
            masked_input = np.copy(zero_input)
            for t, b in zip(token_data, binary_mask_vector):
                masked_input = masked_input + t * int(b)
            file = tempfile.NamedTemporaryFile(delete=False, dir=self.DEFAULT_TEMP_DIR)
            processing_utils.audio_to_file(sample_rate, masked_input, file.name)
            masked_data = client_utils.encode_file_to_base64(file.name)
            file.close()
            Path(file.name).unlink()
            masked_inputs.append(masked_data)
        return masked_inputs

    def postprocess(self, y: tuple[int, np.ndarray] | str | None) -> str | dict | None:
        """
        Parameters:
            y: audio data in either of the following formats: a tuple of (sample_rate, data), or a string filepath or URL to an audio file, or None.
        Returns:
            base64 url data
        """
        if y is None:
            return None
        if isinstance(y, str) and utils.validate_url(y):
            return {"name": y, "data": None, "is_file": True}
        if isinstance(y, tuple):
            sample_rate, data = y
            file_path = self.audio_to_temp_file(
                data, sample_rate, dir=self.DEFAULT_TEMP_DIR, format=self.format
            )
            self.temp_files.add(file_path)
        else:
            file_path = self.make_temp_copy_if_needed(y)
        return {"name": file_path, "data": None, "is_file": True}

    def check_streamable(self):
        if self.source != "microphone":
            raise ValueError(
                "Audio streaming only available if source is 'microphone'."
            )

    def style(
        self,
        **kwargs,
    ):
        """
        This method can be used to change the appearance of the audio component.
        """
        Component.style(
            self,
            **kwargs,
        )
        return self

    def as_example(self, input_data: str | None) -> str:
        return Path(input_data).name if input_data else ""


@document("style")
class File(
    Changeable,
    Selectable,
    Clearable,
    Uploadable,
    IOComponent,
    FileSerializable,
):
    """
    Creates a file component that allows uploading generic file (when used as an input) and or displaying generic files (output).
    Preprocessing: passes the uploaded file as a {tempfile._TemporaryFileWrapper} or {List[tempfile._TemporaryFileWrapper]} depending on `file_count` (or a {bytes}/{List{bytes}} depending on `type`)
    Postprocessing: expects function to return a {str} path to a file, or {List[str]} consisting of paths to files.
    Examples-format: a {str} path to a local file that populates the component.
    Demos: zip_to_json, zip_files
    """

    def __init__(
        self,
        value: str | list[str] | Callable | None = None,
        *,
        file_count: str = "single",
        file_types: list[str] | None = None,
        type: str = "file",
        label: str | None = None,
        every: float | None = None,
        show_label: bool = True,
        interactive: bool | None = None,
        visible: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        **kwargs,
    ):
        """
        Parameters:
            value: Default file to display, given as str file path. If callable, the function will be called whenever the app loads to set the initial value of the component.
            file_count: if single, allows user to upload one file. If "multiple", user uploads multiple files. If "directory", user uploads all files in selected directory. Return type will be list for each file in case of "multiple" or "directory".
            file_types: List of file extensions or types of files to be uploaded (e.g. ['image', '.json', '.mp4']). "file" allows any file to be uploaded, "image" allows only image files to be uploaded, "audio" allows only audio files to be uploaded, "video" allows only video files to be uploaded, "text" allows only text files to be uploaded.
            type: Type of value to be returned by component. "file" returns a temporary file object with the same base name as the uploaded file, whose full path can be retrieved by file_obj.name, "binary" returns an bytes object.
            label: component name in interface.
            every: If `value` is a callable, run the function 'every' number of seconds while the client connection is open. Has no effect otherwise. Queue must be enabled. The event can be accessed (e.g. to cancel it) via this component's .load_event attribute.
            show_label: if True, will display label.
            interactive: if True, will allow users to upload a file; if False, can only be used to display files. If not provided, this is inferred based on whether the component is used as an input or output.
            visible: If False, component will be hidden.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.
        """
        self.file_count = file_count
        self.file_types = file_types
        if file_types is not None and not isinstance(file_types, list):
            raise ValueError(
                f"Parameter file_types must be a list. Received {file_types.__class__.__name__}"
            )
        valid_types = [
            "file",
            "binary",
            "bytes",
        ]  # "bytes" is included for backwards compatibility
        if type not in valid_types:
            raise ValueError(
                f"Invalid value for parameter `type`: {type}. Please choose from one of: {valid_types}"
            )
        if type == "bytes":
            warnings.warn(
                "The `bytes` type is deprecated and may not work as expected. Please use `binary` instead."
            )
        if file_count == "directory" and file_types is not None:
            warnings.warn(
                "The `file_types` parameter is ignored when `file_count` is 'directory'."
            )
        self.type = type
        self.select: EventListenerMethod
        """
        Event listener for when the user selects file from list.
        Uses event data gradio.SelectData to carry `value` referring to name of selected file, and `index` to refer to index.
        See EventData documentation on how to use this event data.
        """
        IOComponent.__init__(
            self,
            label=label,
            every=every,
            show_label=show_label,
            interactive=interactive,
            visible=visible,
            elem_id=elem_id,
            elem_classes=elem_classes,
            value=value,
            **kwargs,
        )

    def get_config(self):
        return {
            "file_count": self.file_count,
            "file_types": self.file_types,
            "value": self.value,
            "selectable": self.selectable,
            **IOComponent.get_config(self),
        }

    @staticmethod
    def update(
        value: Any | Literal[_Keywords.NO_VALUE] | None = _Keywords.NO_VALUE,
        label: str | None = None,
        show_label: bool | None = None,
        interactive: bool | None = None,
        visible: bool | None = None,
    ):
        return {
            "label": label,
            "show_label": show_label,
            "interactive": interactive,
            "visible": visible,
            "value": value,
            "__type__": "update",
        }

    def preprocess(
        self, x: list[dict[str, Any]] | None
    ) -> (
        bytes
        | tempfile._TemporaryFileWrapper
        | list[bytes | tempfile._TemporaryFileWrapper]
        | None
    ):
        """
        Parameters:
            x: List of JSON objects with filename as 'name' property and base64 data as 'data' property
        Returns:
            File objects in requested format
        """
        if x is None:
            return None

        def process_single_file(f) -> bytes | tempfile._TemporaryFileWrapper:
            file_name, data, is_file = (
                f["name"],
                f["data"],
                f.get("is_file", False),
            )
            if self.type == "file":
                if is_file:
                    path = self.make_temp_copy_if_needed(file_name)
                else:
                    data, _ = client_utils.decode_base64_to_binary(data)
                    path = self.file_bytes_to_file(
                        data, dir=self.DEFAULT_TEMP_DIR, file_name=file_name
                    )
                    path = str(utils.abspath(path))
                    self.temp_files.add(path)

                # Creation of tempfiles here
                file = tempfile.NamedTemporaryFile(
                    delete=False, dir=self.DEFAULT_TEMP_DIR
                )
                file.name = path
                file.orig_name = file_name  # type: ignore
                return file
            elif (
                self.type == "binary" or self.type == "bytes"
            ):  # "bytes" is included for backwards compatibility
                if is_file:
                    with open(file_name, "rb") as file_data:
                        return file_data.read()
                return client_utils.decode_base64_to_binary(data)[0]
            else:
                raise ValueError(
                    "Unknown type: "
                    + str(self.type)
                    + ". Please choose from: 'file', 'bytes'."
                )

        if self.file_count == "single":
            if isinstance(x, list):
                return process_single_file(x[0])
            else:
                return process_single_file(x)
        else:
            if isinstance(x, list):
                return [process_single_file(f) for f in x]
            else:
                return process_single_file(x)

    def postprocess(
        self, y: str | list[str] | None
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        """
        Parameters:
            y: file path
        Returns:
            JSON object with key 'name' for filename, 'data' for base64 url, and 'size' for filesize in bytes
        """
        if y is None:
            return None
        if isinstance(y, list):
            return [
                {
                    "orig_name": Path(file).name,
                    "name": self.make_temp_copy_if_needed(file),
                    "size": Path(file).stat().st_size,
                    "data": None,
                    "is_file": True,
                }
                for file in y
            ]
        else:
            d = {
                "orig_name": Path(y).name,
                "name": self.make_temp_copy_if_needed(y),
                "size": Path(y).stat().st_size,
                "data": None,
                "is_file": True,
            }
            return d

    def style(
        self,
        **kwargs,
    ):
        """
        This method can be used to change the appearance of the file component.
        """
        Component.style(
            self,
            **kwargs,
        )
        return self

    def as_example(self, input_data: str | list | None) -> str:
        if input_data is None:
            return ""
        elif isinstance(input_data, list):
            return ", ".join([Path(file).name for file in input_data])
        else:
            return Path(input_data).name

    def api_info(self) -> dict[str, dict | bool]:
        if self.file_count == "single":
            return self._single_file_api_info()
        else:
            return self._multiple_file_api_info()

    def serialized_info(self):
        if self.file_count == "single":
            return self._single_file_serialized_info()
        else:
            return self._multiple_file_serialized_info()

    def example_inputs(self) -> dict[str, Any]:
        if self.file_count == "single":
            return self._single_file_example_inputs()
        else:
            return self._multiple_file_example_inputs()


@document("style")
class Dataframe(Changeable, Inputable, Selectable, IOComponent, JSONSerializable):
    """
    Accepts or displays 2D input through a spreadsheet-like component for dataframes.
    Preprocessing: passes the uploaded spreadsheet data as a {pandas.DataFrame}, {numpy.array}, {List[List]}, or {List} depending on `type`
    Postprocessing: expects a {pandas.DataFrame}, {numpy.array}, {List[List]}, {List}, a {Dict} with keys `data` (and optionally `headers`), or {str} path to a csv, which is rendered in the spreadsheet.
    Examples-format: a {str} filepath to a csv with data, a pandas dataframe, or a list of lists (excluding headers) where each sublist is a row of data.
    Demos: filter_records, matrix_transpose, tax_calculator
    """

    markdown_parser = None

    def __init__(
        self,
        value: list[list[Any]] | Callable | None = None,
        *,
        headers: list[str] | None = None,
        row_count: int | tuple[int, str] = (1, "dynamic"),
        col_count: int | tuple[int, str] | None = None,
        datatype: str | list[str] = "str",
        type: str = "pandas",
        max_rows: int | None = 20,
        max_cols: int | None = None,
        overflow_row_behaviour: str = "paginate",
        label: str | None = None,
        every: float | None = None,
        show_label: bool = True,
        interactive: bool | None = None,
        visible: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        wrap: bool = False,
        **kwargs,
    ):
        """
        Parameters:
            value: Default value as a 2-dimensional list of values. If callable, the function will be called whenever the app loads to set the initial value of the component.
            headers: List of str header names. If None, no headers are shown.
            row_count: Limit number of rows for input and decide whether user can create new rows. The first element of the tuple is an `int`, the row count; the second should be 'fixed' or 'dynamic', the new row behaviour. If an `int` is passed the rows default to 'dynamic'
            col_count: Limit number of columns for input and decide whether user can create new columns. The first element of the tuple is an `int`, the number of columns; the second should be 'fixed' or 'dynamic', the new column behaviour. If an `int` is passed the columns default to 'dynamic'
            datatype: Datatype of values in sheet. Can be provided per column as a list of strings, or for the entire sheet as a single string. Valid datatypes are "str", "number", "bool", "date", and "markdown".
            type: Type of value to be returned by component. "pandas" for pandas dataframe, "numpy" for numpy array, or "array" for a Python array.
            label: component name in interface.
            max_rows: Maximum number of rows to display at once. Set to None for infinite.
            max_cols: Maximum number of columns to display at once. Set to None for infinite.
            overflow_row_behaviour: If set to "paginate", will create pages for overflow rows. If set to "show_ends", will show initial and final rows and truncate middle rows.
            label: component name in interface.
            every: If `value` is a callable, run the function 'every' number of seconds while the client connection is open. Has no effect otherwise. Queue must be enabled. The event can be accessed (e.g. to cancel it) via this component's .load_event attribute.
            show_label: if True, will display label.
            interactive: if True, will allow users to edit the dataframe; if False, can only be used to display data. If not provided, this is inferred based on whether the component is used as an input or output.
            visible: If False, component will be hidden.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.
            wrap: if True text in table cells will wrap when appropriate, if False the table will scroll horizontally. Defaults to False.
        """

        self.wrap = wrap
        self.row_count = self.__process_counts(row_count)
        self.col_count = self.__process_counts(
            col_count, len(headers) if headers else 3
        )

        self.__validate_headers(headers, self.col_count[0])

        self.headers = (
            headers if headers is not None else list(range(1, self.col_count[0] + 1))
        )
        self.datatype = (
            datatype if isinstance(datatype, list) else [datatype] * self.col_count[0]
        )
        valid_types = ["pandas", "numpy", "array"]
        if type not in valid_types:
            raise ValueError(
                f"Invalid value for parameter `type`: {type}. Please choose from one of: {valid_types}"
            )
        self.type = type
        values = {
            "str": "",
            "number": 0,
            "bool": False,
            "date": "01/01/1970",
            "markdown": "",
            "html": "",
        }
        column_dtypes = (
            [datatype] * self.col_count[0] if isinstance(datatype, str) else datatype
        )
        self.empty_input = [
            [values[c] for c in column_dtypes] for _ in range(self.row_count[0])
        ]

        self.max_rows = max_rows
        self.max_cols = max_cols
        self.overflow_row_behaviour = overflow_row_behaviour
        self.select: EventListenerMethod
        """
        Event listener for when the user selects cell within Dataframe.
        Uses event data gradio.SelectData to carry `value` referring to value of selected cell, and `index` tuple to refer to index row and column.
        See EventData documentation on how to use this event data.
        """
        IOComponent.__init__(
            self,
            label=label,
            every=every,
            show_label=show_label,
            interactive=interactive,
            visible=visible,
            elem_id=elem_id,
            elem_classes=elem_classes,
            value=value,
            **kwargs,
        )

    def get_config(self):
        return {
            "headers": self.headers,
            "datatype": self.datatype,
            "row_count": self.row_count,
            "col_count": self.col_count,
            "value": self.value,
            "max_rows": self.max_rows,
            "max_cols": self.max_cols,
            "overflow_row_behaviour": self.overflow_row_behaviour,
            "wrap": self.wrap,
            **IOComponent.get_config(self),
        }

    @staticmethod
    def update(
        value: Any | Literal[_Keywords.NO_VALUE] | None = _Keywords.NO_VALUE,
        max_rows: int | None = None,
        max_cols: str | None = None,
        label: str | None = None,
        show_label: bool | None = None,
        interactive: bool | None = None,
        visible: bool | None = None,
    ):
        return {
            "max_rows": max_rows,
            "max_cols": max_cols,
            "label": label,
            "show_label": show_label,
            "interactive": interactive,
            "visible": visible,
            "value": value,
            "__type__": "update",
        }

    def preprocess(self, x: DataframeData):
        """
        Parameters:
            x: 2D array of str, numeric, or bool data
        Returns:
            Dataframe in requested format
        """
        if self.type == "pandas":
            if x.get("headers") is not None:
                return pd.DataFrame(x["data"], columns=x.get("headers"))
            else:
                return pd.DataFrame(x["data"])
        if self.type == "numpy":
            return np.array(x["data"])
        elif self.type == "array":
            return x["data"]
        else:
            raise ValueError(
                "Unknown type: "
                + str(self.type)
                + ". Please choose from: 'pandas', 'numpy', 'array'."
            )

    def postprocess(
        self, y: str | pd.DataFrame | np.ndarray | list[list[str | float]] | dict
    ) -> dict:
        """
        Parameters:
            y: dataframe in given format
        Returns:
            JSON object with key 'headers' for list of header names, 'data' for 2D array of string or numeric data
        """
        if y is None:
            return self.postprocess(self.empty_input)
        if isinstance(y, dict):
            return y
        if isinstance(y, str):
            dataframe = pd.read_csv(y)
            return {
                "headers": list(dataframe.columns),
                "data": Dataframe.__process_markdown(
                    dataframe.to_dict(orient="split")["data"], self.datatype
                ),
            }
        if isinstance(y, pd.DataFrame):
            return {
                "headers": list(y.columns),  # type: ignore
                "data": Dataframe.__process_markdown(
                    y.to_dict(orient="split")["data"], self.datatype  # type: ignore
                ),
            }
        if isinstance(y, (np.ndarray, list)):
            if len(y) == 0:
                return self.postprocess([[]])
            if isinstance(y, np.ndarray):
                y = y.tolist()
            assert isinstance(y, list), "output cannot be converted to list"

            _headers = self.headers

            if len(self.headers) < len(y[0]):
                _headers = [
                    *self.headers,
                    *list(range(len(self.headers) + 1, len(y[0]) + 1)),
                ]
            elif len(self.headers) > len(y[0]):
                _headers = self.headers[: len(y[0])]

            return {
                "headers": _headers,
                "data": Dataframe.__process_markdown(y, self.datatype),
            }
        raise ValueError("Cannot process value as a Dataframe")

    @staticmethod
    def __process_counts(count, default=3) -> tuple[int, str]:
        if count is None:
            return (default, "dynamic")
        if type(count) == int or type(count) == float:
            return (int(count), "dynamic")
        else:
            return count

    @staticmethod
    def __validate_headers(headers: list[str] | None, col_count: int):
        if headers is not None and len(headers) != col_count:
            raise ValueError(
                f"The length of the headers list must be equal to the col_count int.\n"
                f"The column count is set to {col_count} but `headers` has {len(headers)} items. "
                f"Check the values passed to `col_count` and `headers`."
            )

    @classmethod
    def __process_markdown(cls, data: list[list[Any]], datatype: list[str]):
        if "markdown" not in datatype:
            return data

        if cls.markdown_parser is None:
            cls.markdown_parser = utils.get_markdown_parser()

        for i in range(len(data)):
            for j in range(len(data[i])):
                if datatype[j] == "markdown":
                    data[i][j] = cls.markdown_parser.render(data[i][j])

        return data

    def style(
        self,
        **kwargs,
    ):
        """
        This method can be used to change the appearance of the DataFrame component.
        """
        Component.style(
            self,
            **kwargs,
        )
        return self

    def as_example(self, input_data: pd.DataFrame | np.ndarray | str | None):
        if input_data is None:
            return ""
        elif isinstance(input_data, pd.DataFrame):
            return input_data.head(n=5).to_dict(orient="split")["data"]  # type: ignore
        elif isinstance(input_data, np.ndarray):
            return input_data.tolist()
        return input_data


@document("style")
class Timeseries(Changeable, IOComponent, JSONSerializable):
    """
    Creates a component that can be used to upload/preview timeseries csv files or display a dataframe consisting of a time series graphically.
    Preprocessing: passes the uploaded timeseries data as a {pandas.DataFrame} into the function
    Postprocessing: expects a {pandas.DataFrame} or {str} path to a csv to be returned, which is then displayed as a timeseries graph
    Examples-format: a {str} filepath of csv data with time series data.
    Demos: fraud_detector
    """

    def __init__(
        self,
        value: str | Callable | None = None,
        *,
        x: str | None = None,
        y: str | list[str] | None = None,
        colors: list[str] | None = None,
        label: str | None = None,
        every: float | None = None,
        show_label: bool = True,
        interactive: bool | None = None,
        visible: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        **kwargs,
    ):
        """
        Parameters:
            value: File path for the timeseries csv file. If callable, the function will be called whenever the app loads to set the initial value of the component.
            x: Column name of x (time) series. None if csv has no headers, in which case first column is x series.
            y: Column name of y series, or list of column names if multiple series. None if csv has no headers, in which case every column after first is a y series.
            label: component name in interface.
            every: If `value` is a callable, run the function 'every' number of seconds while the client connection is open. Has no effect otherwise. Queue must be enabled. The event can be accessed (e.g. to cancel it) via this component's .load_event attribute.
            colors: an ordered list of colors to use for each line plot
            show_label: if True, will display label.
            interactive: if True, will allow users to upload a timeseries csv; if False, can only be used to display timeseries data. If not provided, this is inferred based on whether the component is used as an input or output.
            visible: If False, component will be hidden.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.
        """
        self.x = x
        if isinstance(y, str):
            y = [y]
        self.y = y
        self.colors = colors
        IOComponent.__init__(
            self,
            label=label,
            every=every,
            show_label=show_label,
            interactive=interactive,
            visible=visible,
            elem_id=elem_id,
            elem_classes=elem_classes,
            value=value,
            **kwargs,
        )

    def get_config(self):
        return {
            "x": self.x,
            "y": self.y,
            "value": self.value,
            "colors": self.colors,
            **IOComponent.get_config(self),
        }

    @staticmethod
    def update(
        value: Any | Literal[_Keywords.NO_VALUE] | None = _Keywords.NO_VALUE,
        colors: list[str] | None = None,
        label: str | None = None,
        show_label: bool | None = None,
        interactive: bool | None = None,
        visible: bool | None = None,
    ):
        return {
            "colors": colors,
            "label": label,
            "show_label": show_label,
            "interactive": interactive,
            "visible": visible,
            "value": value,
            "__type__": "update",
        }

    def preprocess(self, x: dict | None) -> pd.DataFrame | None:
        """
        Parameters:
            x: Dict with keys 'data': 2D array of str, numeric, or bool data, 'headers': list of strings for header names, 'range': optional two element list designating start of end of subrange.
        Returns:
            Dataframe of timeseries data
        """
        if x is None:
            return x
        elif x.get("is_file"):
            dataframe = pd.read_csv(x["name"])
        else:
            dataframe = pd.DataFrame(data=x["data"], columns=x["headers"])
        if x.get("range") is not None:
            dataframe = dataframe.loc[dataframe[self.x or 0] >= x["range"][0]]
            dataframe = dataframe.loc[dataframe[self.x or 0] <= x["range"][1]]
        return dataframe

    def postprocess(self, y: str | pd.DataFrame | None) -> dict | None:
        """
        Parameters:
            y: csv or dataframe with timeseries data
        Returns:
            JSON object with key 'headers' for list of header names, 'data' for 2D array of string or numeric data
        """
        if y is None:
            return None
        if isinstance(y, str):
            dataframe = pd.read_csv(y)
            return {
                "headers": dataframe.columns.values.tolist(),
                "data": dataframe.values.tolist(),
            }
        if isinstance(y, pd.DataFrame):
            return {"headers": y.columns.values.tolist(), "data": y.values.tolist()}
        raise ValueError("Cannot process value as Timeseries data")

    def style(
        self,
        **kwargs,
    ):
        """
        This method can be used to change the appearance of the TimeSeries component.
        """
        Component.style(
            self,
            **kwargs,
        )
        return self

    def as_example(self, input_data: str | None) -> str:
        return Path(input_data).name if input_data else ""


@document()
class State(IOComponent, SimpleSerializable):
    """
    Special hidden component that stores session state across runs of the demo by the
    same user. The value of the State variable is cleared when the user refreshes the page.

    Preprocessing: No preprocessing is performed
    Postprocessing: No postprocessing is performed
    Demos: blocks_simple_squares
    Guides: real-time-speech-recognition
    """

    allow_string_shortcut = False

    def __init__(
        self,
        value: Any = None,
        **kwargs,
    ):
        """
        Parameters:
            value: the initial value (of arbitrary type) of the state. The provided argument is deepcopied. If a callable is provided, the function will be called whenever the app loads to set the initial value of the state.
        """
        self.stateful = True
        IOComponent.__init__(self, value=deepcopy(value), **kwargs)


class Variable(State):
    """Variable was renamed to State. This class is kept for backwards compatibility."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_block_name(self):
        return "state"


@document("style")
class Button(Clickable, IOComponent, StringSerializable):
    """
    Used to create a button, that can be assigned arbitrary click() events. The label (value) of the button can be used as an input or set via the output of a function.

    Preprocessing: passes the button value as a {str} into the function
    Postprocessing: expects a {str} to be returned from a function, which is set as the label of the button
    Demos: blocks_inputs, blocks_kinematics
    """

    def __init__(
        self,
        value: str | Callable = "Run",
        *,
        variant: str = "secondary",
        visible: bool = True,
        interactive: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        **kwargs,
    ):
        """
        Parameters:
            value: Default text for the button to display. If callable, the function will be called whenever the app loads to set the initial value of the component.
            variant: 'primary' for main call-to-action, 'secondary' for a more subdued style, 'stop' for a stop button.
            visible: If False, component will be hidden.
            interactive: If False, the Button will be in a disabled state.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.
        """
        IOComponent.__init__(
            self,
            visible=visible,
            elem_id=elem_id,
            elem_classes=elem_classes,
            value=value,
            interactive=interactive,
            **kwargs,
        )
        if variant == "plain":
            warnings.warn("'plain' variant deprecated, using 'secondary' instead.")
            variant = "secondary"
        self.variant = variant

    def get_config(self):
        return {
            "value": self.value,
            "variant": self.variant,
            "interactive": self.interactive,
            **Component.get_config(self),
        }

    @staticmethod
    def update(
        value: str | Literal[_Keywords.NO_VALUE] | None = _Keywords.NO_VALUE,
        variant: str | None = None,
        visible: bool | None = None,
        interactive: bool | None = None,
    ):
        return {
            "variant": variant,
            "visible": visible,
            "value": value,
            "interactive": interactive,
            "__type__": "update",
        }

    def style(
        self,
        *,
        full_width: bool | None = None,
        size: Literal["sm"] | Literal["lg"] | None = None,
        **kwargs,
    ):
        """
        This method can be used to change the appearance of the button component.
        Parameters:
            full_width: If True, will expand to fill parent container.
            size: Size of the button. Can be "sm" or "lg".
        """
        if full_width is not None:
            self._style["full_width"] = full_width
        if size is not None:
            self._style["size"] = size

        Component.style(self, **kwargs)
        return self


@document("style")
class UploadButton(Clickable, Uploadable, IOComponent, FileSerializable):
    """
    Used to create an upload button, when cicked allows a user to upload files that satisfy the specified file type or generic files (if file_type not set).
    Preprocessing: passes the uploaded file as a {file-object} or {List[file-object]} depending on `file_count` (or a {bytes}/{List{bytes}} depending on `type`)
    Postprocessing: expects function to return a {str} path to a file, or {List[str]} consisting of paths to files.
    Examples-format: a {str} path to a local file that populates the component.
    Demos: upload_button
    """

    def __init__(
        self,
        label: str = "Upload a File",
        value: str | list[str] | Callable | None = None,
        *,
        visible: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        type: str = "file",
        file_count: str = "single",
        file_types: list[str] | None = None,
        **kwargs,
    ):
        """
        Parameters:
            value: Default text for the button to display.
            type: Type of value to be returned by component. "file" returns a temporary file object with the same base name as the uploaded file, whose full path can be retrieved by file_obj.name, "binary" returns an bytes object.
            file_count: if single, allows user to upload one file. If "multiple", user uploads multiple files. If "directory", user uploads all files in selected directory. Return type will be list for each file in case of "multiple" or "directory".
            file_types: List of type of files to be uploaded. "file" allows any file to be uploaded, "image" allows only image files to be uploaded, "audio" allows only audio files to be uploaded, "video" allows only video files to be uploaded, "text" allows only text files to be uploaded.
            label: Text to display on the button. Defaults to "Upload a File".
            visible: If False, component will be hidden.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.
        """
        self.type = type
        self.file_count = file_count
        if file_count == "directory" and file_types is not None:
            warnings.warn(
                "The `file_types` parameter is ignored when `file_count` is 'directory'."
            )
        if file_types is not None and not isinstance(file_types, list):
            raise ValueError(
                f"Parameter file_types must be a list. Received {file_types.__class__.__name__}"
            )
        self.file_types = file_types
        self.label = label
        IOComponent.__init__(
            self,
            label=label,
            visible=visible,
            elem_id=elem_id,
            elem_classes=elem_classes,
            value=value,
            **kwargs,
        )

    def get_config(self):
        return {
            "label": self.label,
            "value": self.value,
            "file_count": self.file_count,
            "file_types": self.file_types,
            **Component.get_config(self),
        }

    @staticmethod
    def update(
        value: str | Literal[_Keywords.NO_VALUE] | None = _Keywords.NO_VALUE,
        interactive: bool | None = None,
        visible: bool | None = None,
    ):
        return {
            "interactive": interactive,
            "visible": visible,
            "value": value,
            "__type__": "update",
        }

    def preprocess(
        self, x: list[dict[str, Any]] | None
    ) -> (
        bytes
        | tempfile._TemporaryFileWrapper
        | list[bytes | tempfile._TemporaryFileWrapper]
        | None
    ):
        """
        Parameters:
            x: List of JSON objects with filename as 'name' property and base64 data as 'data' property
        Returns:
            File objects in requested format
        """
        if x is None:
            return None

        def process_single_file(f) -> bytes | tempfile._TemporaryFileWrapper:
            file_name, data, is_file = (
                f["name"],
                f["data"],
                f.get("is_file", False),
            )
            if self.type == "file":
                if is_file:
                    path = self.make_temp_copy_if_needed(file_name)
                else:
                    data, _ = client_utils.decode_base64_to_binary(data)
                    path = self.file_bytes_to_file(
                        data, dir=self.DEFAULT_TEMP_DIR, file_name=file_name
                    )
                    path = str(utils.abspath(path))
                    self.temp_files.add(path)
                file = tempfile.NamedTemporaryFile(
                    delete=False, dir=self.DEFAULT_TEMP_DIR
                )
                file.name = path
                file.orig_name = file_name  # type: ignore
                return file
            elif self.type == "bytes":
                if is_file:
                    with open(file_name, "rb") as file_data:
                        return file_data.read()
                return client_utils.decode_base64_to_binary(data)[0]
            else:
                raise ValueError(
                    "Unknown type: "
                    + str(self.type)
                    + ". Please choose from: 'file', 'bytes'."
                )

        if self.file_count == "single":
            if isinstance(x, list):
                return process_single_file(x[0])
            else:
                return process_single_file(x)
        else:
            if isinstance(x, list):
                return [process_single_file(f) for f in x]
            else:
                return process_single_file(x)

    def style(
        self,
        *,
        full_width: bool | None = None,
        size: Literal["sm"] | Literal["lg"] | None = None,
        **kwargs,
    ):
        """
        This method can be used to change the appearance of the button component.
        Parameters:
            full_width: If True, will expand to fill parent container.
            size: Size of the button. Can be "sm" or "lg".
        """
        if full_width is not None:
            self._style["full_width"] = full_width
        if size is not None:
            self._style["size"] = size

        Component.style(self, **kwargs)
        return self


@document("style")
class ColorPicker(
    Changeable, Inputable, Submittable, Blurrable, IOComponent, StringSerializable
):
    """
    Creates a color picker for user to select a color as string input.
    Preprocessing: passes selected color value as a {str} into the function.
    Postprocessing: expects a {str} returned from function and sets color picker value to it.
    Examples-format: a {str} with a hexadecimal representation of a color, e.g. "#ff0000" for red.
    Demos: color_picker, color_generator
    """

    def __init__(
        self,
        value: str | Callable | None = None,
        *,
        label: str | None = None,
        info: str | None = None,
        every: float | None = None,
        show_label: bool = True,
        interactive: bool | None = None,
        visible: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        **kwargs,
    ):
        """
        Parameters:
            value: default text to provide in color picker. If callable, the function will be called whenever the app loads to set the initial value of the component.
            label: component name in interface.
            info: additional component description.
            every: If `value` is a callable, run the function 'every' number of seconds while the client connection is open. Has no effect otherwise. Queue must be enabled. The event can be accessed (e.g. to cancel it) via this component's .load_event attribute.
            show_label: if True, will display label.
            interactive: if True, will be rendered as an editable color picker; if False, editing will be disabled. If not provided, this is inferred based on whether the component is used as an input or output.
            visible: If False, component will be hidden.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.
        """
        self.cleared_value = "#000000"
        IOComponent.__init__(
            self,
            label=label,
            info=info,
            every=every,
            show_label=show_label,
            interactive=interactive,
            visible=visible,
            elem_id=elem_id,
            elem_classes=elem_classes,
            value=value,
            **kwargs,
        )

    def example_inputs(self) -> dict[str, Any]:
        return {
            "raw": "#000000",
            "serialized": "#000000",
        }

    def get_config(self):
        return {
            "value": self.value,
            **IOComponent.get_config(self),
        }

    @staticmethod
    def update(
        value: str | Literal[_Keywords.NO_VALUE] | None = _Keywords.NO_VALUE,
        label: str | None = None,
        show_label: bool | None = None,
        visible: bool | None = None,
        interactive: bool | None = None,
    ):
        return {
            "value": value,
            "label": label,
            "show_label": show_label,
            "visible": visible,
            "interactive": interactive,
            "__type__": "update",
        }

    def preprocess(self, x: str | None) -> str | None:
        """
        Any preprocessing needed to be performed on function input.
        Parameters:
            x: text
        Returns:
            text
        """
        if x is None:
            return None
        else:
            return str(x)

    def postprocess(self, y: str | None) -> str | None:
        """
        Any postprocessing needed to be performed on function output.
        Parameters:
            y: text
        Returns:
            text
        """
        if y is None:
            return None
        else:
            return str(y)


############################
# Only Output Components
############################


@document("style")
class Label(Changeable, Selectable, IOComponent, JSONSerializable):
    """
    Displays a classification label, along with confidence scores of top categories, if provided.
    Preprocessing: this component does *not* accept input.
    Postprocessing: expects a {Dict[str, float]} of classes and confidences, or {str} with just the class or an {int}/{float} for regression outputs, or a {str} path to a .json file containing a json dictionary in the structure produced by Label.postprocess().

    Demos: main_note, titanic_survival
    Guides: image-classification-in-pytorch, image-classification-in-tensorflow, image-classification-with-vision-transformers, building-a-pictionary-app
    """

    CONFIDENCES_KEY = "confidences"

    def __init__(
        self,
        value: dict[str, float] | str | float | Callable | None = None,
        *,
        num_top_classes: int | None = None,
        label: str | None = None,
        every: float | None = None,
        show_label: bool = True,
        visible: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        color: str | None = None,
        **kwargs,
    ):
        """
        Parameters:
            value: Default value to show in the component. If a str or number is provided, simply displays the string or number. If a {Dict[str, float]} of classes and confidences is provided, displays the top class on top and the `num_top_classes` below, along with their confidence bars. If callable, the function will be called whenever the app loads to set the initial value of the component.
            num_top_classes: number of most confident classes to show.
            label: component name in interface.
            every: If `value` is a callable, run the function 'every' number of seconds while the client connection is open. Has no effect otherwise. Queue must be enabled. The event can be accessed (e.g. to cancel it) via this component's .load_event attribute.
            show_label: if True, will display label.
            visible: If False, component will be hidden.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.
            color: The background color of the label (either a valid css color name or hexadecimal string).
        """
        self.num_top_classes = num_top_classes
        self.color = color
        self.select: EventListenerMethod
        """
        Event listener for when the user selects a category from Label.
        Uses event data gradio.SelectData to carry `value` referring to name of selected category, and `index` to refer to index.
        See EventData documentation on how to use this event data.
        """
        IOComponent.__init__(
            self,
            label=label,
            every=every,
            show_label=show_label,
            visible=visible,
            elem_id=elem_id,
            elem_classes=elem_classes,
            value=value,
            **kwargs,
        )

    def get_config(self):
        return {
            "num_top_classes": self.num_top_classes,
            "value": self.value,
            "color": self.color,
            "selectable": self.selectable,
            **IOComponent.get_config(self),
        }

    def postprocess(self, y: dict[str, float] | str | float | None) -> dict | None:
        """
        Parameters:
            y: a dictionary mapping labels to confidence value, or just a string/numerical label by itself
        Returns:
            Object with key 'label' representing primary label, and key 'confidences' representing a list of label-confidence pairs
        """
        if y is None or y == {}:
            return None
        if isinstance(y, str) and y.endswith(".json") and Path(y).exists():
            return self.serialize(y)
        if isinstance(y, (str, float, int)):
            return {"label": str(y)}
        if isinstance(y, dict):
            if "confidences" in y and isinstance(y["confidences"], dict):
                y = y["confidences"]
                y = {c["label"]: c["confidence"] for c in y}
            sorted_pred = sorted(y.items(), key=operator.itemgetter(1), reverse=True)
            if self.num_top_classes is not None:
                sorted_pred = sorted_pred[: self.num_top_classes]
            return {
                "label": sorted_pred[0][0],
                "confidences": [
                    {"label": pred[0], "confidence": pred[1]} for pred in sorted_pred
                ],
            }
        raise ValueError(
            "The `Label` output interface expects one of: a string label, or an int label, a "
            "float label, or a dictionary whose keys are labels and values are confidences. "
            f"Instead, got a {type(y)}"
        )

    @staticmethod
    def update(
        value: dict[str, float]
        | str
        | float
        | Literal[_Keywords.NO_VALUE]
        | None = _Keywords.NO_VALUE,
        label: str | None = None,
        show_label: bool | None = None,
        visible: bool | None = None,
        color: str | Literal[_Keywords.NO_VALUE] | None = _Keywords.NO_VALUE,
    ):
        # If color is not specified (NO_VALUE) map it to None so that
        # it gets filtered out in postprocess. This will mean the color
        # will not be updated in the front-end
        if color is _Keywords.NO_VALUE:
            color = None
        # If the color was specified by the developer as None
        # Map is so that the color is updated to be transparent,
        # e.g. no background default state.
        elif color is None:
            color = "transparent"
        return {
            "label": label,
            "show_label": show_label,
            "visible": visible,
            "value": value,
            "color": color,
            "__type__": "update",
        }

    def style(
        self,
        *,
        container: bool | None = None,
    ):
        """
        This method can be used to change the appearance of the label component.
        Parameters:
            container: If True, will add a container to the label - providing some extra padding around the border.
        """
        Component.style(self, container=container)
        return self


@document("style")
class HighlightedText(Changeable, Selectable, IOComponent, JSONSerializable):
    """
    Displays text that contains spans that are highlighted by category or numerical value.
    Preprocessing: this component does *not* accept input.
    Postprocessing: expects a {List[Tuple[str, float | str]]]} consisting of spans of text and their associated labels, or a {Dict} with two keys: (1) "text" whose value is the complete text, and "entities", which is a list of dictionaries, each of which have the keys: "entity" (consisting of the entity label), "start" (the character index where the label starts), and "end" (the character index where the label ends). Entities should not overlap.

    Demos: diff_texts, text_analysis
    Guides: named-entity-recognition
    """

    def __init__(
        self,
        value: list[tuple[str, str | float | None]] | dict | Callable | None = None,
        *,
        color_map: dict[str, str]
        | None = None,  # Parameter moved to HighlightedText.style()
        show_legend: bool = False,
        combine_adjacent: bool = False,
        adjacent_separator: str = "",
        label: str | None = None,
        every: float | None = None,
        show_label: bool = True,
        visible: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        **kwargs,
    ):
        """
        Parameters:
            value: Default value to show. If callable, the function will be called whenever the app loads to set the initial value of the component.
            show_legend: whether to show span categories in a separate legend or inline.
            combine_adjacent: If True, will merge the labels of adjacent tokens belonging to the same category.
            adjacent_separator: Specifies the separator to be used between tokens if combine_adjacent is True.
            label: component name in interface.
            every: If `value` is a callable, run the function 'every' number of seconds while the client connection is open. Has no effect otherwise. Queue must be enabled. The event can be accessed (e.g. to cancel it) via this component's .load_event attribute.
            show_label: if True, will display label.
            visible: If False, component will be hidden.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.
        """
        self.color_map = color_map
        if color_map is not None:
            warnings.warn(
                "The 'color_map' parameter has been moved from the constructor to `HighlightedText.style()` ",
            )
        self.show_legend = show_legend
        self.combine_adjacent = combine_adjacent
        self.adjacent_separator = adjacent_separator
        self.select: EventListenerMethod
        """
        Event listener for when the user selects Highlighted text span.
        Uses event data gradio.SelectData to carry `value` referring to selected [text, label] tuple, and `index` to refer to span index.
        See EventData documentation on how to use this event data.
        """
        IOComponent.__init__(
            self,
            label=label,
            every=every,
            show_label=show_label,
            visible=visible,
            elem_id=elem_id,
            elem_classes=elem_classes,
            value=value,
            **kwargs,
        )

    def get_config(self):
        return {
            "color_map": self.color_map,
            "show_legend": self.show_legend,
            "value": self.value,
            "selectable": self.selectable,
            **IOComponent.get_config(self),
        }

    @staticmethod
    def update(
        value: list[tuple[str, str | float | None]]
        | dict
        | Literal[_Keywords.NO_VALUE]
        | None = _Keywords.NO_VALUE,
        color_map: dict[str, str] | None = None,
        show_legend: bool | None = None,
        label: str | None = None,
        show_label: bool | None = None,
        visible: bool | None = None,
    ):
        updated_config = {
            "color_map": color_map,
            "show_legend": show_legend,
            "label": label,
            "show_label": show_label,
            "visible": visible,
            "value": value,
            "__type__": "update",
        }
        return updated_config

    def postprocess(
        self, y: list[tuple[str, str | float | None]] | dict | None
    ) -> list[tuple[str, str | float | None]] | None:
        """
        Parameters:
            y: List of (word, category) tuples
        Returns:
            List of (word, category) tuples
        """
        if y is None:
            return None
        if isinstance(y, dict):
            try:
                text = y["text"]
                entities = y["entities"]
            except KeyError as ke:
                raise ValueError(
                    "Expected a dictionary with keys 'text' and 'entities' "
                    "for the value of the HighlightedText component."
                ) from ke
            if len(entities) == 0:
                y = [(text, None)]
            else:
                list_format = []
                index = 0
                entities = sorted(entities, key=lambda x: x["start"])
                for entity in entities:
                    list_format.append((text[index : entity["start"]], None))
                    list_format.append(
                        (text[entity["start"] : entity["end"]], entity["entity"])
                    )
                    index = entity["end"]
                list_format.append((text[index:], None))
                y = list_format
        if self.combine_adjacent:
            output = []
            running_text, running_category = None, None
            for text, category in y:
                if running_text is None:
                    running_text = text
                    running_category = category
                elif category == running_category:
                    running_text += self.adjacent_separator + text
                elif not text:
                    # Skip fully empty item, these get added in processing
                    # of dictionaries.
                    pass
                else:
                    output.append((running_text, running_category))
                    running_text = text
                    running_category = category
            if running_text is not None:
                output.append((running_text, running_category))
            return output
        else:
            return y

    def style(
        self,
        *,
        color_map: dict[str, str] | None = None,
        container: bool | None = None,
        **kwargs,
    ):
        """
        This method can be used to change the appearance of the HighlightedText component.
        Parameters:
            color_map: Map between category and respective colors.
            container: If True, will place the component in a container - providing some extra padding around the border.
        """
        if color_map is not None:
            self._style["color_map"] = color_map

        Component.style(self, container=container, **kwargs)
        return self


@document("style")
class AnnotatedImage(Selectable, IOComponent, JSONSerializable):
    """
    Displays a base image and colored subsections on top of that image. Subsections can take the from of rectangles (e.g. object detection) or masks (e.g. image segmentation).
    Preprocessing: this component does *not* accept input.
    Postprocessing: expects a {Tuple[numpy.ndarray | PIL.Image | str, List[Tuple[numpy.ndarray | Tuple[int, int, int, int], str]]]} consisting of a base image and a list of subsections, that are either (x1, y1, x2, y2) tuples identifying object boundaries, or 0-1 confidence masks of the same shape as the image. A label is provided for each subsection.

    Demos: image_segmentation
    """

    def __init__(
        self,
        value: tuple[
            np.ndarray | _Image.Image | str,
            list[tuple[np.ndarray | tuple[int, int, int, int], str]],
        ]
        | None = None,
        *,
        show_legend: bool = True,
        label: str | None = None,
        every: float | None = None,
        show_label: bool = True,
        visible: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        **kwargs,
    ):
        """
        Parameters:
            value: Tuple of base image and list of (subsection, label) pairs.
            show_legend: If True, will show a legend of the subsections.
            label: component name in interface.
            every: If `value` is a callable, run the function 'every' number of seconds while the client connection is open. Has no effect otherwise. Queue must be enabled. The event can be accessed (e.g. to cancel it) via this component's .load_event attribute.
            show_label: if True, will display label.
            visible: If False, component will be hidden.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.
        """
        self.show_legend = show_legend
        self.select: EventListenerMethod
        """
        Event listener for when the user selects Image subsection.
        Uses event data gradio.SelectData to carry `value` referring to selected subsection label, and `index` to refer to subsection index.
        See EventData documentation on how to use this event data.
        """
        IOComponent.__init__(
            self,
            label=label,
            every=every,
            show_label=show_label,
            visible=visible,
            elem_id=elem_id,
            elem_classes=elem_classes,
            value=value,
            **kwargs,
        )

    def get_config(self):
        return {
            "show_legend": self.show_legend,
            "value": self.value,
            "selectable": self.selectable,
            **IOComponent.get_config(self),
        }

    @staticmethod
    def update(
        value: tuple[
            np.ndarray | _Image.Image | str,
            list[tuple[np.ndarray | tuple[int, int, int, int], str]],
        ]
        | Literal[_Keywords.NO_VALUE] = _Keywords.NO_VALUE,
        show_legend: bool | None = None,
        label: str | None = None,
        show_label: bool | None = None,
        visible: bool | None = None,
    ):
        updated_config = {
            "show_legend": show_legend,
            "label": label,
            "show_label": show_label,
            "visible": visible,
            "value": value,
            "__type__": "update",
        }
        return updated_config

    def postprocess(
        self,
        y: tuple[
            np.ndarray | _Image.Image | str,
            list[tuple[np.ndarray | tuple[int, int, int, int], str]],
        ],
    ) -> tuple[dict, list[tuple[dict, str]]] | None:
        """
        Parameters:
            y: Tuple of base image and list of subsections, with each subsection a two-part tuple where the first element is a 4 element bounding box or a 0-1 confidence mask, and the second element is the label.
        Returns:
            Tuple of base image file and list of subsections, with each subsection a two-part tuple where the first element image path of the mask, and the second element is the label.
        """
        if y is None:
            return None
        base_img = y[0]
        if isinstance(base_img, str):
            base_img_path = base_img
            base_img = np.array(_Image.open(base_img))
        elif isinstance(base_img, np.ndarray):
            base_file = self.img_array_to_temp_file(base_img, dir=self.DEFAULT_TEMP_DIR)
            base_img_path = str(utils.abspath(base_file))
        elif isinstance(base_img, _Image.Image):
            base_file = self.pil_to_temp_file(base_img, dir=self.DEFAULT_TEMP_DIR)
            base_img_path = str(utils.abspath(base_file))
            base_img = np.array(base_img)
        else:
            raise ValueError(
                "AnnotatedImage only accepts filepaths, PIL images or numpy arrays for the base image."
            )
        self.temp_files.add(base_img_path)

        sections = []
        color_map = self._style.get("color_map", {})

        def hex_to_rgb(value):
            value = value.lstrip("#")
            lv = len(value)
            return [int(value[i : i + lv // 3], 16) for i in range(0, lv, lv // 3)]

        for mask, label in y[1]:
            mask_array = np.zeros((base_img.shape[0], base_img.shape[1]))
            if isinstance(mask, np.ndarray):
                mask_array = mask
            else:
                x1, y1, x2, y2 = mask
                border_width = 3
                mask_array[y1:y2, x1:x2] = 0.5
                mask_array[y1:y2, x1 : x1 + border_width] = 1
                mask_array[y1:y2, x2 - border_width : x2] = 1
                mask_array[y1 : y1 + border_width, x1:x2] = 1
                mask_array[y2 - border_width : y2, x1:x2] = 1

            if label in color_map:
                rgb_color = hex_to_rgb(color_map[label])
            else:
                rgb_color = [255, 0, 0]
            colored_mask = np.zeros((base_img.shape[0], base_img.shape[1], 4))
            solid_mask = np.copy(mask_array)
            solid_mask[solid_mask > 0] = 1

            colored_mask[:, :, 0] = rgb_color[0] * solid_mask
            colored_mask[:, :, 1] = rgb_color[1] * solid_mask
            colored_mask[:, :, 2] = rgb_color[2] * solid_mask
            colored_mask[:, :, 3] = mask_array * 255

            colored_mask_img = _Image.fromarray((colored_mask).astype(np.uint8))

            mask_file = self.pil_to_temp_file(
                colored_mask_img, dir=self.DEFAULT_TEMP_DIR
            )
            mask_file_path = str(utils.abspath(mask_file))
            self.temp_files.add(mask_file_path)

            sections.append(
                ({"name": mask_file_path, "data": None, "is_file": True}, label)
            )

        return {"name": base_img_path, "data": None, "is_file": True}, sections

    def style(
        self,
        *,
        height: int | None = None,
        width: int | None = None,
        color_map: dict[str, str] | None = None,
        **kwargs,
    ):
        """
        This method can be used to change the appearance of the Image component.
        Parameters:
            height: Height of the image.
            width: Width of the image.
            color_map: A dictionary mapping labels to colors. The colors must be specified as hex codes.
        """
        self._style["height"] = height
        self._style["width"] = width
        self._style["color_map"] = color_map
        Component.style(
            self,
            **kwargs,
        )
        return self


@document("style")
class JSON(Changeable, IOComponent, JSONSerializable):
    """
    Used to display arbitrary JSON output prettily.
    Preprocessing: this component does *not* accept input.
    Postprocessing: expects a {str} filepath to a file containing valid JSON -- or a {list} or {dict} that is valid JSON

    Demos: zip_to_json, blocks_xray
    """

    def __init__(
        self,
        value: str | dict | list | Callable | None = None,
        *,
        label: str | None = None,
        every: float | None = None,
        show_label: bool = True,
        visible: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        **kwargs,
    ):
        """
        Parameters:
            value: Default value. If callable, the function will be called whenever the app loads to set the initial value of the component.
            label: component name in interface.
            every: If `value` is a callable, run the function 'every' number of seconds while the client connection is open. Has no effect otherwise. Queue must be enabled. The event can be accessed (e.g. to cancel it) via this component's .load_event attribute.
            show_label: if True, will display label.
            visible: If False, component will be hidden.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.
        """
        IOComponent.__init__(
            self,
            label=label,
            every=every,
            show_label=show_label,
            visible=visible,
            elem_id=elem_id,
            elem_classes=elem_classes,
            value=value,
            **kwargs,
        )

    def get_config(self):
        return {
            "value": self.value,
            **IOComponent.get_config(self),
        }

    @staticmethod
    def update(
        value: Any | Literal[_Keywords.NO_VALUE] | None = _Keywords.NO_VALUE,
        label: str | None = None,
        show_label: bool | None = None,
        visible: bool | None = None,
    ):
        updated_config = {
            "label": label,
            "show_label": show_label,
            "visible": visible,
            "value": value,
            "__type__": "update",
        }
        return updated_config

    def postprocess(self, y: dict | list | str | None) -> dict | list | None:
        """
        Parameters:
            y: either a string filepath to a JSON file, or a Python list or dict that can be converted to JSON
        Returns:
            JSON output in Python list or dict format
        """
        if y is None:
            return None
        if isinstance(y, str):
            return json.loads(y)
        else:
            return y

    def style(self, *, container: bool | None = None, **kwargs):
        """
        This method can be used to change the appearance of the JSON component.
        Parameters:
            container: If True, will place the JSON in a container - providing some extra padding around the border.
        """
        Component.style(self, container=container, **kwargs)
        return self


@document()
class HTML(Changeable, IOComponent, StringSerializable):
    """
    Used to display arbitrary HTML output.
    Preprocessing: this component does *not* accept input.
    Postprocessing: expects a valid HTML {str}.

    Demos: text_analysis
    Guides: key-features
    """

    def __init__(
        self,
        value: str | Callable = "",
        *,
        label: str | None = None,
        every: float | None = None,
        show_label: bool = True,
        visible: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        **kwargs,
    ):
        """
        Parameters:
            value: Default value. If callable, the function will be called whenever the app loads to set the initial value of the component.
            label: component name in interface.
            every: If `value` is a callable, run the function 'every' number of seconds while the client connection is open. Has no effect otherwise. Queue must be enabled. The event can be accessed (e.g. to cancel it) via this component's .load_event attribute.
            show_label: if True, will display label.
            visible: If False, component will be hidden.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.
        """
        IOComponent.__init__(
            self,
            label=label,
            every=every,
            show_label=show_label,
            visible=visible,
            elem_id=elem_id,
            elem_classes=elem_classes,
            value=value,
            **kwargs,
        )

    def get_config(self):
        return {
            "value": self.value,
            **IOComponent.get_config(self),
        }

    @staticmethod
    def update(
        value: Any | Literal[_Keywords.NO_VALUE] | None = _Keywords.NO_VALUE,
        label: str | None = None,
        show_label: bool | None = None,
        visible: bool | None = None,
    ):
        updated_config = {
            "label": label,
            "show_label": show_label,
            "visible": visible,
            "value": value,
            "__type__": "update",
        }
        return updated_config

    def style(self):
        return self


@document("style")
class Gallery(IOComponent, GallerySerializable, Selectable):
    """
    Used to display a list of images as a gallery that can be scrolled through.
    Preprocessing: this component does *not* accept input.
    Postprocessing: expects a list of images in any format, {List[numpy.array | PIL.Image | str]}, or a {List} of (image, {str} caption) tuples and displays them.

    Demos: fake_gan
    """

    def __init__(
        self,
        value: list[np.ndarray | _Image.Image | str | tuple] | Callable | None = None,
        *,
        label: str | None = None,
        every: float | None = None,
        show_label: bool = True,
        visible: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        **kwargs,
    ):
        """
        Parameters:
            value: List of images to display in the gallery by default. If callable, the function will be called whenever the app loads to set the initial value of the component.
            label: component name in interface.
            every: If `value` is a callable, run the function 'every' number of seconds while the client connection is open. Has no effect otherwise. Queue must be enabled. The event can be accessed (e.g. to cancel it) via this component's .load_event attribute.
            show_label: if True, will display label.
            visible: If False, component will be hidden.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.
        """
        self.select: EventListenerMethod
        """
        Event listener for when the user selects image within Gallery.
        Uses event data gradio.SelectData to carry `value` referring to caption of selected image, and `index` to refer to index.
        See EventData documentation on how to use this event data.
        """
        IOComponent.__init__(
            self,
            label=label,
            every=every,
            show_label=show_label,
            visible=visible,
            elem_id=elem_id,
            elem_classes=elem_classes,
            value=value,
            **kwargs,
        )

    @staticmethod
    def update(
        value: Any | Literal[_Keywords.NO_VALUE] | None = _Keywords.NO_VALUE,
        label: str | None = None,
        show_label: bool | None = None,
        visible: bool | None = None,
    ):
        updated_config = {
            "label": label,
            "show_label": show_label,
            "visible": visible,
            "value": value,
            "__type__": "update",
        }
        return updated_config

    def get_config(self):
        return {
            "value": self.value,
            **IOComponent.get_config(self),
        }

    def postprocess(
        self,
        y: list[np.ndarray | _Image.Image | str]
        | list[tuple[np.ndarray | _Image.Image | str, str]]
        | None,
    ) -> list[str]:
        """
        Parameters:
            y: list of images, or list of (image, caption) tuples
        Returns:
            list of string file paths to images in temp directory
        """
        if y is None:
            return []
        output = []
        for img in y:
            caption = None
            if isinstance(img, (tuple, list)):
                img, caption = img
            if isinstance(img, np.ndarray):
                file = self.img_array_to_temp_file(img, dir=self.DEFAULT_TEMP_DIR)
                file_path = str(utils.abspath(file))
                self.temp_files.add(file_path)
            elif isinstance(img, _Image.Image):
                file = self.pil_to_temp_file(img, dir=self.DEFAULT_TEMP_DIR)
                file_path = str(utils.abspath(file))
                self.temp_files.add(file_path)
            elif isinstance(img, str):
                if utils.validate_url(img):
                    file_path = img
                else:
                    file_path = self.make_temp_copy_if_needed(img)
            else:
                raise ValueError(f"Cannot process type as image: {type(img)}")

            if caption is not None:
                output.append(
                    [{"name": file_path, "data": None, "is_file": True}, caption]
                )
            else:
                output.append({"name": file_path, "data": None, "is_file": True})

        return output

    def style(
        self,
        *,
        grid: int | tuple | None = None,
        columns: int | tuple | None = None,
        rows: int | tuple | None = None,
        height: str | None = None,
        container: bool | None = None,
        preview: bool | None = None,
        object_fit: str | None = None,
        **kwargs,
    ):
        """
        This method can be used to change the appearance of the gallery component.
        Parameters:
            grid: ('grid' has been renamed to 'columns') Represents the number of images that should be shown in one row, for each of the six standard screen sizes (<576px, <768px, <992px, <1200px, <1400px, >1400px). if fewer that 6 are given then the last will be used for all subsequent breakpoints
            columns: Represents the number of columns in the image grid, for each of the six standard screen sizes (<576px, <768px, <992px, <1200px, <1400px, >1400px). if fewer that 6 are given then the last will be used for all subsequent breakpoints
            rows: Represents the number of rows in the image grid, for each of the six standard screen sizes (<576px, <768px, <992px, <1200px, <1400px, >1400px). if fewer that 6 are given then the last will be used for all subsequent breakpoints
            height: Height of the gallery.
            container: If True, will place gallery in a container - providing some extra padding around the border.
            preview: If True, will display the Gallery in preview mode, which shows all of the images as thumbnails and allows the user to click on them to view them in full size.
            object_fit: CSS object-fit property for the thumbnail images in the gallery. Can be "contain", "cover", "fill", "none", or "scale-down".
        """
        if grid is not None:
            warnings.warn(
                "The 'grid' parameter will be deprecated. Please use 'columns' instead.",
            )
            self._style["grid_cols"] = grid
        if columns is not None:
            self._style["grid_cols"] = columns
        if rows is not None:
            self._style["grid_rows"] = rows
        if height is not None:
            self._style["height"] = height
        if preview is not None:
            self._style["preview"] = preview
        if object_fit is not None:
            self._style["object_fit"] = object_fit

        Component.style(self, container=container, **kwargs)
        return self


class Carousel(IOComponent, Changeable, SimpleSerializable):
    """
    Deprecated Component
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        raise DeprecationWarning(
            "The Carousel component is deprecated. Please consider using the Gallery "
            "component, which can be used to display images (and optional captions).",
        )


@document("style")
class Chatbot(Changeable, Selectable, IOComponent, JSONSerializable):
    """
    Displays a chatbot output showing both user submitted messages and responses. Supports a subset of Markdown including bold, italics, code, and images.
    Preprocessing: this component does *not* accept input.
    Postprocessing: expects function to return a {List[List[str | None | Tuple]]}, a list of lists. The inner list should have 2 elements: the user message and the response message. Messages should be strings, tuples, or Nones. If the message is a string, it can include Markdown. If it is a tuple, it should consist of (string filepath to image/video/audio, [optional string alt text]). Messages that are `None` are not displayed.

    Demos: chatbot_simple, chatbot_multimodal
    Guides: creating-a-chatbot
    """

    def __init__(
        self,
        value: list[list[str | tuple[str] | tuple[str, str] | None]]
        | Callable
        | None = None,
        color_map: dict[str, str] | None = None,  # Parameter moved to Chatbot.style()
        *,
        label: str | None = None,
        every: float | None = None,
        show_label: bool = True,
        visible: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        **kwargs,
    ):
        """
        Parameters:
            value: Default value to show in chatbot. If callable, the function will be called whenever the app loads to set the initial value of the component.
            label: component name in interface.
            every: If `value` is a callable, run the function 'every' number of seconds while the client connection is open. Has no effect otherwise. Queue must be enabled. The event can be accessed (e.g. to cancel it) via this component's .load_event attribute.
            show_label: if True, will display label.
            visible: If False, component will be hidden.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.
        """
        if color_map is not None:
            warnings.warn(
                "The 'color_map' parameter has been deprecated.",
            )
        self.select: EventListenerMethod
        """
        Event listener for when the user selects message from Chatbot.
        Uses event data gradio.SelectData to carry `value` referring to text of selected message, and `index` tuple to refer to [message, participant] index.
        See EventData documentation on how to use this event data.
        """

        IOComponent.__init__(
            self,
            label=label,
            every=every,
            show_label=show_label,
            visible=visible,
            elem_id=elem_id,
            elem_classes=elem_classes,
            value=value,
            **kwargs,
        )

    def get_config(self):
        return {
            "value": self.value,
            "selectable": self.selectable,
            **IOComponent.get_config(self),
        }

    @staticmethod
    def update(
        value: list[list[str | tuple[str] | tuple[str, str] | None]]
        | Literal[_Keywords.NO_VALUE]
        | None = _Keywords.NO_VALUE,
        label: str | None = None,
        show_label: bool | None = None,
        visible: bool | None = None,
    ):
        updated_config = {
            "label": label,
            "show_label": show_label,
            "visible": visible,
            "value": value,
            "__type__": "update",
        }
        return updated_config

    def _preprocess_chat_messages(
        self, chat_message: str | dict | None
    ) -> str | tuple[str] | tuple[str, str] | None:
        if chat_message is None:
            return None
        elif isinstance(chat_message, dict):
            if chat_message["alt_text"] is not None:
                return (chat_message["name"], chat_message["alt_text"])
            else:
                return (chat_message["name"],)
        else:  # string
            return chat_message

    def preprocess(
        self,
        y: list[list[str | dict | None] | tuple[str | dict | None, str | dict | None]],
    ) -> list[list[str | tuple[str] | tuple[str, str] | None]]:
        if y is None:
            return y
        processed_messages = []
        for message_pair in y:
            assert isinstance(
                message_pair, (tuple, list)
            ), f"Expected a list of lists or list of tuples. Received: {message_pair}"
            assert (
                len(message_pair) == 2
            ), f"Expected a list of lists of length 2 or list of tuples of length 2. Received: {message_pair}"
            processed_messages.append(
                [
                    self._preprocess_chat_messages(message_pair[0]),
                    self._preprocess_chat_messages(message_pair[1]),
                ]
            )
        return processed_messages

    def _postprocess_chat_messages(
        self, chat_message: str | tuple | list | None
    ) -> str | dict | None:
        if chat_message is None:
            return None
        elif isinstance(chat_message, (tuple, list)):
            file_uri = chat_message[0]
            if utils.validate_url(file_uri):
                filepath = file_uri
            else:
                filepath = self.make_temp_copy_if_needed(file_uri)

            mime_type = client_utils.get_mimetype(filepath)
            return {
                "name": filepath,
                "mime_type": mime_type,
                "alt_text": chat_message[1] if len(chat_message) > 1 else None,
                "data": None,  # These last two fields are filled in by the frontend
                "is_file": True,
            }
        elif isinstance(chat_message, str):
            chat_message = inspect.cleandoc(chat_message)
            return chat_message
        else:
            raise ValueError(f"Invalid message for Chatbot component: {chat_message}")

    def postprocess(
        self,
        y: list[list[str | tuple[str] | tuple[str, str] | None] | tuple],
    ) -> list[list[str | dict | None]]:
        """
        Parameters:
            y: List of lists representing the message and response pairs. Each message and response should be a string, which may be in Markdown format.  It can also be a tuple whose first element is a string filepath or URL to an image/video/audio, and second (optional) element is the alt text, in which case the media file is displayed. It can also be None, in which case that message is not displayed.
        Returns:
            List of lists representing the message and response. Each message and response will be a string of HTML, or a dictionary with media information. Or None if the message is not to be displayed.
        """
        if y is None:
            return []
        processed_messages = []
        for message_pair in y:
            assert isinstance(
                message_pair, (tuple, list)
            ), f"Expected a list of lists or list of tuples. Received: {message_pair}"
            assert (
                len(message_pair) == 2
            ), f"Expected a list of lists of length 2 or list of tuples of length 2. Received: {message_pair}"
            processed_messages.append(
                [
                    self._postprocess_chat_messages(message_pair[0]),
                    self._postprocess_chat_messages(message_pair[1]),
                ]
            )
        return processed_messages

    def style(self, height: int | None = None, **kwargs):
        """
        This method can be used to change the appearance of the Chatbot component.
        """
        if height is not None:
            self._style["height"] = height
        if kwargs.get("color_map") is not None:
            warnings.warn("The 'color_map' parameter has been deprecated.")

        Component.style(
            self,
            **kwargs,
        )
        return self


@document("style")
class Model3D(
    Changeable, Uploadable, Editable, Clearable, IOComponent, FileSerializable
):
    """
    Component allows users to upload or view 3D Model files (.obj, .glb, or .gltf).
    Preprocessing: This component passes the uploaded file as a {str} filepath.
    Postprocessing: expects function to return a {str} path to a file of type (.obj, glb, or .gltf)

    Demos: model3D
    Guides: how-to-use-3D-model-component
    """

    def __init__(
        self,
        value: str | Callable | None = None,
        *,
        clear_color: list[float] | None = None,
        label: str | None = None,
        every: float | None = None,
        show_label: bool = True,
        visible: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        **kwargs,
    ):
        """
        Parameters:
            value: path to (.obj, glb, or .gltf) file to show in model3D viewer. If callable, the function will be called whenever the app loads to set the initial value of the component.
            clear_color: background color of scene
            label: component name in interface.
            every: If `value` is a callable, run the function 'every' number of seconds while the client connection is open. Has no effect otherwise. Queue must be enabled. The event can be accessed (e.g. to cancel it) via this component's .load_event attribute.
            show_label: if True, will display label.
            visible: If False, component will be hidden.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.
        """
        self.clear_color = clear_color or [0, 0, 0, 0]
        IOComponent.__init__(
            self,
            label=label,
            every=every,
            show_label=show_label,
            visible=visible,
            elem_id=elem_id,
            elem_classes=elem_classes,
            value=value,
            **kwargs,
        )

    def get_config(self):
        return {
            "clearColor": self.clear_color,
            "value": self.value,
            **IOComponent.get_config(self),
        }

    def example_inputs(self) -> dict[str, Any]:
        return {
            "raw": {"is_file": False, "data": media_data.BASE64_MODEL3D},
            "serialized": "https://github.com/gradio-app/gradio/raw/main/test/test_files/Box.gltf",
        }

    @staticmethod
    def update(
        value: Any | Literal[_Keywords.NO_VALUE] | None = _Keywords.NO_VALUE,
        label: str | None = None,
        show_label: bool | None = None,
        visible: bool | None = None,
    ):
        updated_config = {
            "label": label,
            "show_label": show_label,
            "visible": visible,
            "value": value,
            "__type__": "update",
        }
        return updated_config

    def preprocess(self, x: dict[str, str] | None) -> str | None:
        """
        Parameters:
            x: JSON object with filename as 'name' property and base64 data as 'data' property
        Returns:
            string file path to temporary file with the 3D image model
        """
        if x is None:
            return x
        file_name, file_data, is_file = (
            x["name"],
            x["data"],
            x.get("is_file", False),
        )
        if is_file:
            temp_file_path = self.make_temp_copy_if_needed(file_name)
        else:
            temp_file_path = self.base64_to_temp_file_if_needed(file_data, file_name)

        return temp_file_path

    def postprocess(self, y: str | None) -> dict[str, str] | None:
        """
        Parameters:
            y: path to the model
        Returns:
            file name mapped to base64 url data
        """
        if y is None:
            return y
        data = {
            "name": self.make_temp_copy_if_needed(y),
            "data": None,
            "is_file": True,
        }
        return data

    def style(self, **kwargs):
        """
        This method can be used to change the appearance of the Model3D component.
        """
        Component.style(
            self,
            **kwargs,
        )
        return self

    def as_example(self, input_data: str | None) -> str:
        return Path(input_data).name if input_data else ""


@document()
class Plot(Changeable, Clearable, IOComponent, JSONSerializable):
    """
    Used to display various kinds of plots (matplotlib, plotly, or bokeh are supported)
    Preprocessing: this component does *not* accept input.
    Postprocessing: expects either a {matplotlib.figure.Figure}, a {plotly.graph_objects._figure.Figure}, or a {dict} corresponding to a bokeh plot (json_item format)

    Demos: altair_plot, outbreak_forecast, blocks_kinematics, stock_forecast, map_airbnb
    Guides: plot-component-for-maps
    """

    def __init__(
        self,
        value: Callable | None | pd.DataFrame = None,
        *,
        label: str | None = None,
        every: float | None = None,
        show_label: bool = True,
        visible: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        **kwargs,
    ):
        """
        Parameters:
            value: Optionally, supply a default plot object to display, must be a matplotlib, plotly, altair, or bokeh figure, or a callable. If callable, the function will be called whenever the app loads to set the initial value of the component.
            label: component name in interface.
            every: If `value` is a callable, run the function 'every' number of seconds while the client connection is open. Has no effect otherwise. Queue must be enabled. The event can be accessed (e.g. to cancel it) via this component's .load_event attribute.
            show_label: if True, will display label.
            visible: If False, component will be hidden.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.
        """
        IOComponent.__init__(
            self,
            label=label,
            every=every,
            show_label=show_label,
            visible=visible,
            elem_id=elem_id,
            elem_classes=elem_classes,
            value=value,
            **kwargs,
        )

    def get_config(self):
        try:
            import bokeh  # type: ignore

            bokeh_version = bokeh.__version__
        except ImportError:
            bokeh_version = None
        return {
            "value": self.value,
            "bokeh_version": bokeh_version,
            **IOComponent.get_config(self),
        }

    @staticmethod
    def update(
        value: Any | Literal[_Keywords.NO_VALUE] | None = _Keywords.NO_VALUE,
        label: str | None = None,
        show_label: bool | None = None,
        visible: bool | None = None,
    ):
        updated_config = {
            "label": label,
            "show_label": show_label,
            "visible": visible,
            "value": value,
            "__type__": "update",
        }
        return updated_config

    def postprocess(self, y) -> dict[str, str] | None:
        """
        Parameters:
            y: plot data
        Returns:
            plot type mapped to plot base64 data
        """
        import matplotlib.figure

        if y is None:
            return None
        if isinstance(y, (ModuleType, matplotlib.figure.Figure)):  # type: ignore
            dtype = "matplotlib"
            out_y = processing_utils.encode_plot_to_base64(y)
        elif "bokeh" in y.__module__:
            dtype = "bokeh"
            from bokeh.embed import json_item  # type: ignore

            out_y = json.dumps(json_item(y))
        else:
            is_altair = "altair" in y.__module__
            dtype = "altair" if is_altair else "plotly"
            out_y = y.to_json()
        return {"type": dtype, "plot": out_y}

    def style(self, container: bool | None = None):
        Component.style(
            self,
            container=container,
        )
        return self


class AltairPlot:
    @staticmethod
    def create_legend(position, title):
        if position == "none":
            legend = None
        else:
            position = {"orient": position} if position else {}
            legend = {"title": title, **position}

        return legend

    @staticmethod
    def create_scale(limit):
        return alt.Scale(domain=limit) if limit else alt.Undefined


@document()
class ScatterPlot(Plot):
    """
    Create a scatter plot.

    Preprocessing: this component does *not* accept input.
    Postprocessing: expects a pandas dataframe with the data to plot.

    Demos: native_plots
    Guides: creating-a-dashboard-from-bigquery-data
    """

    def __init__(
        self,
        value: pd.DataFrame | Callable | None = None,
        x: str | None = None,
        y: str | None = None,
        *,
        color: str | None = None,
        size: str | None = None,
        shape: str | None = None,
        title: str | None = None,
        tooltip: list[str] | str | None = None,
        x_title: str | None = None,
        y_title: str | None = None,
        color_legend_title: str | None = None,
        size_legend_title: str | None = None,
        shape_legend_title: str | None = None,
        color_legend_position: str | None = None,
        size_legend_position: str | None = None,
        shape_legend_position: str | None = None,
        height: int | None = None,
        width: int | None = None,
        x_lim: list[int | float] | None = None,
        y_lim: list[int | float] | None = None,
        caption: str | None = None,
        interactive: bool | None = True,
        label: str | None = None,
        every: float | None = None,
        show_label: bool = True,
        visible: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
    ):
        """
        Parameters:
            value: The pandas dataframe containing the data to display in a scatter plot, or a callable. If callable, the function will be called whenever the app loads to set the initial value of the component.
            x: Column corresponding to the x axis.
            y: Column corresponding to the y axis.
            color: The column to determine the point color. If the column contains numeric data, gradio will interpolate the column data so that small values correspond to light colors and large values correspond to dark values.
            size: The column used to determine the point size. Should contain numeric data so that gradio can map the data to the point size.
            shape: The column used to determine the point shape. Should contain categorical data. Gradio will map each unique value to a different shape.
            title: The title to display on top of the chart.
            tooltip: The column (or list of columns) to display on the tooltip when a user hovers a point on the plot.
            x_title: The title given to the x axis. By default, uses the value of the x parameter.
            y_title: The title given to the y axis. By default, uses the value of the y parameter.
            color_legend_title: The title given to the color legend. By default, uses the value of color parameter.
            size_legend_title: The title given to the size legend. By default, uses the value of the size parameter.
            shape_legend_title: The title given to the shape legend. By default, uses the value of the shape parameter.
            color_legend_position: The position of the color legend. If the string value 'none' is passed, this legend is omitted. For other valid position values see: https://vega.github.io/vega/docs/legends/#orientation.
            size_legend_position: The position of the size legend. If the string value 'none' is passed, this legend is omitted. For other valid position values see: https://vega.github.io/vega/docs/legends/#orientation.
            shape_legend_position: The position of the shape legend. If the string value 'none' is passed, this legend is omitted. For other valid position values see: https://vega.github.io/vega/docs/legends/#orientation.
            height: The height of the plot in pixels.
            width: The width of the plot in pixels.
            x_lim: A tuple or list containing the limits for the x-axis, specified as [x_min, x_max].
            y_lim: A tuple of list containing the limits for the y-axis, specified as [y_min, y_max].
            caption: The (optional) caption to display below the plot.
            interactive: Whether users should be able to interact with the plot by panning or zooming with their mouse or trackpad.
            label: The (optional) label to display on the top left corner of the plot.
            every:  If `value` is a callable, run the function 'every' number of seconds while the client connection is open. Has no effect otherwise. Queue must be enabled. The event can be accessed (e.g. to cancel it) via this component's .load_event attribute.
            show_label: Whether the label should be displayed.
            visible: Whether the plot should be visible.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.
        """
        self.x = x
        self.y = y
        self.color = color
        self.size = size
        self.shape = shape
        self.tooltip = tooltip
        self.title = title
        self.x_title = x_title
        self.y_title = y_title
        self.color_legend_title = color_legend_title
        self.color_legend_position = color_legend_position
        self.size_legend_title = size_legend_title
        self.size_legend_position = size_legend_position
        self.shape_legend_title = shape_legend_title
        self.shape_legend_position = shape_legend_position
        self.caption = caption
        self.interactive_chart = interactive
        self.width = width
        self.height = height
        self.x_lim = x_lim
        self.y_lim = y_lim
        super().__init__(
            value=value,
            label=label,
            every=every,
            show_label=show_label,
            visible=visible,
            elem_id=elem_id,
            elem_classes=elem_classes,
        )

    def get_config(self):
        config = super().get_config()
        config["caption"] = self.caption
        return config

    def get_block_name(self) -> str:
        return "plot"

    @staticmethod
    def update(
        value: DataFrame | dict | Literal[_Keywords.NO_VALUE] = _Keywords.NO_VALUE,
        x: str | None = None,
        y: str | None = None,
        color: str | None = None,
        size: str | None = None,
        shape: str | None = None,
        title: str | None = None,
        tooltip: list[str] | str | None = None,
        x_title: str | None = None,
        y_title: str | None = None,
        color_legend_title: str | None = None,
        size_legend_title: str | None = None,
        shape_legend_title: str | None = None,
        color_legend_position: str | None = None,
        size_legend_position: str | None = None,
        shape_legend_position: str | None = None,
        height: int | None = None,
        width: int | None = None,
        x_lim: list[int | float] | None = None,
        y_lim: list[int | float] | None = None,
        interactive: bool | None = None,
        caption: str | None = None,
        label: str | None = None,
        show_label: bool | None = None,
        visible: bool | None = None,
    ):
        """Update an existing plot component.

        If updating any of the plot properties (color, size, etc) the value, x, and y parameters must be specified.

        Parameters:
            value: The pandas dataframe containing the data to display in a scatter plot.
            x: Column corresponding to the x axis.
            y: Column corresponding to the y axis.
            color: The column to determine the point color. If the column contains numeric data, gradio will interpolate the column data so that small values correspond to light colors and large values correspond to dark values.
            size: The column used to determine the point size. Should contain numeric data so that gradio can map the data to the point size.
            shape: The column used to determine the point shape. Should contain categorical data. Gradio will map each unique value to a different shape.
            title: The title to display on top of the chart.
            tooltip: The column (or list of columns) to display on the tooltip when a user hovers a point on the plot.
            x_title: The title given to the x axis. By default, uses the value of the x parameter.
            y_title: The title given to the y axis. By default, uses the value of the y parameter.
            color_legend_title: The title given to the color legend. By default, uses the value of color parameter.
            size_legend_title: The title given to the size legend. By default, uses the value of the size parameter.
            shape_legend_title: The title given to the shape legend. By default, uses the value of the shape parameter.
            color_legend_position: The position of the color legend. If the string value 'none' is passed, this legend is omitted. For other valid position values see: https://vega.github.io/vega/docs/legends/#orientation.
            size_legend_position: The position of the size legend. If the string value 'none' is passed, this legend is omitted. For other valid position values see: https://vega.github.io/vega/docs/legends/#orientation.
            shape_legend_position: The position of the shape legend. If the string value 'none' is passed, this legend is omitted. For other valid position values see: https://vega.github.io/vega/docs/legends/#orientation.
            height: The height of the plot in pixels.
            width: The width of the plot in pixels.
            x_lim: A tuple or list containing the limits for the x-axis, specified as [x_min, x_max].
            y_lim: A tuple of list containing the limits for the y-axis, specified as [y_min, y_max].
            interactive: Whether users should be able to interact with the plot by panning or zooming with their mouse or trackpad.
            caption: The (optional) caption to display below the plot.
            label: The (optional) label to display in the top left corner of the plot.
            show_label: Whether the label should be displayed.
            visible: Whether the plot should be visible.
        """
        properties = [
            x,
            y,
            color,
            size,
            shape,
            title,
            tooltip,
            x_title,
            y_title,
            color_legend_title,
            size_legend_title,
            shape_legend_title,
            color_legend_position,
            size_legend_position,
            shape_legend_position,
            height,
            width,
            x_lim,
            y_lim,
            interactive,
        ]
        if any(properties):
            if not isinstance(value, pd.DataFrame):
                raise ValueError(
                    "In order to update plot properties the value parameter "
                    "must be provided, and it must be a Dataframe. Please pass a value "
                    "parameter to gr.ScatterPlot.update."
                )
            if x is None or y is None:
                raise ValueError(
                    "In order to update plot properties, the x and y axis data "
                    "must be specified. Please pass valid values for x an y to "
                    "gr.ScatterPlot.update."
                )
            chart = ScatterPlot.create_plot(value, *properties)
            value = {"type": "altair", "plot": chart.to_json(), "chart": "scatter"}

        updated_config = {
            "label": label,
            "show_label": show_label,
            "visible": visible,
            "value": value,
            "caption": caption,
            "__type__": "update",
        }
        return updated_config

    @staticmethod
    def create_plot(
        value: pd.DataFrame,
        x: str,
        y: str,
        color: str | None = None,
        size: str | None = None,
        shape: str | None = None,
        title: str | None = None,
        tooltip: list[str] | str | None = None,
        x_title: str | None = None,
        y_title: str | None = None,
        color_legend_title: str | None = None,
        size_legend_title: str | None = None,
        shape_legend_title: str | None = None,
        color_legend_position: str | None = None,
        size_legend_position: str | None = None,
        shape_legend_position: str | None = None,
        height: int | None = None,
        width: int | None = None,
        x_lim: list[int | float] | None = None,
        y_lim: list[int | float] | None = None,
        interactive: bool | None = True,
    ):
        """Helper for creating the scatter plot."""
        interactive = True if interactive is None else interactive
        encodings = {
            "x": alt.X(
                x,  # type: ignore
                title=x_title or x,  # type: ignore
                scale=AltairPlot.create_scale(x_lim),  # type: ignore
            ),  # ignore: type
            "y": alt.Y(
                y,  # type: ignore
                title=y_title or y,  # type: ignore
                scale=AltairPlot.create_scale(y_lim),  # type: ignore
            ),
        }
        properties = {}
        if title:
            properties["title"] = title
        if height:
            properties["height"] = height
        if width:
            properties["width"] = width
        if color:
            if is_numeric_dtype(value[color]):
                domain = [value[color].min(), value[color].max()]
                range_ = [0, 1]
                type_ = "quantitative"
            else:
                domain = value[color].unique().tolist()
                range_ = list(range(len(domain)))
                type_ = "nominal"

            encodings["color"] = {
                "field": color,
                "type": type_,
                "legend": AltairPlot.create_legend(
                    position=color_legend_position, title=color_legend_title or color
                ),
                "scale": {"domain": domain, "range": range_},
            }
        if tooltip:
            encodings["tooltip"] = tooltip
        if size:
            encodings["size"] = {
                "field": size,
                "type": "quantitative" if is_numeric_dtype(value[size]) else "nominal",
                "legend": AltairPlot.create_legend(
                    position=size_legend_position, title=size_legend_title or size
                ),
            }
        if shape:
            encodings["shape"] = {
                "field": shape,
                "type": "quantitative" if is_numeric_dtype(value[shape]) else "nominal",
                "legend": AltairPlot.create_legend(
                    position=shape_legend_position, title=shape_legend_title or shape
                ),
            }
        chart = (
            alt.Chart(value)  # type: ignore
            .mark_point(clip=True)  # type: ignore
            .encode(**encodings)
            .properties(background="transparent", **properties)
        )
        if interactive:
            chart = chart.interactive()

        return chart

    def postprocess(self, y: pd.DataFrame | dict | None) -> dict[str, str] | None:
        # if None or update
        if y is None or isinstance(y, Dict):
            return y
        if self.x is None or self.y is None:
            raise ValueError("No value provided for required parameters `x` and `y`.")
        chart = self.create_plot(
            value=y,
            x=self.x,
            y=self.y,
            color=self.color,
            size=self.size,
            shape=self.shape,
            title=self.title,
            tooltip=self.tooltip,
            x_title=self.x_title,
            y_title=self.y_title,
            color_legend_title=self.color_legend_title,
            size_legend_title=self.size_legend_title,
            shape_legend_title=self.size_legend_title,
            color_legend_position=self.color_legend_position,
            size_legend_position=self.size_legend_position,
            shape_legend_position=self.shape_legend_position,
            interactive=self.interactive_chart,
            height=self.height,
            width=self.width,
            x_lim=self.x_lim,
            y_lim=self.y_lim,
        )

        return {"type": "altair", "plot": chart.to_json(), "chart": "scatter"}


@document()
class LinePlot(Plot):
    """
    Create a line plot.

    Preprocessing: this component does *not* accept input.
    Postprocessing: expects a pandas dataframe with the data to plot.

    Demos: native_plots, live_dashboard
    """

    def __init__(
        self,
        value: pd.DataFrame | Callable | None = None,
        x: str | None = None,
        y: str | None = None,
        *,
        color: str | None = None,
        stroke_dash: str | None = None,
        overlay_point: bool | None = None,
        title: str | None = None,
        tooltip: list[str] | str | None = None,
        x_title: str | None = None,
        y_title: str | None = None,
        color_legend_title: str | None = None,
        stroke_dash_legend_title: str | None = None,
        color_legend_position: str | None = None,
        stroke_dash_legend_position: str | None = None,
        height: int | None = None,
        width: int | None = None,
        x_lim: list[int] | None = None,
        y_lim: list[int] | None = None,
        caption: str | None = None,
        interactive: bool | None = True,
        label: str | None = None,
        show_label: bool = True,
        every: float | None = None,
        visible: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
    ):
        """
        Parameters:
            value: The pandas dataframe containing the data to display in a scatter plot.
            x: Column corresponding to the x axis.
            y: Column corresponding to the y axis.
            color: The column to determine the point color. If the column contains numeric data, gradio will interpolate the column data so that small values correspond to light colors and large values correspond to dark values.
            stroke_dash: The column to determine the symbol used to draw the line, e.g. dashed lines, dashed lines with points.
            overlay_point: Whether to draw a point on the line for each (x, y) coordinate pair.
            title: The title to display on top of the chart.
            tooltip: The column (or list of columns) to display on the tooltip when a user hovers a point on the plot.
            x_title: The title given to the x axis. By default, uses the value of the x parameter.
            y_title: The title given to the y axis. By default, uses the value of the y parameter.
            color_legend_title: The title given to the color legend. By default, uses the value of color parameter.
            stroke_dash_legend_title: The title given to the stroke_dash legend. By default, uses the value of the stroke_dash parameter.
            color_legend_position: The position of the color legend. If the string value 'none' is passed, this legend is omitted. For other valid position values see: https://vega.github.io/vega/docs/legends/#orientation.
            stroke_dash_legend_position: The position of the stoke_dash legend. If the string value 'none' is passed, this legend is omitted. For other valid position values see: https://vega.github.io/vega/docs/legends/#orientation.
            height: The height of the plot in pixels.
            width: The width of the plot in pixels.
            x_lim: A tuple or list containing the limits for the x-axis, specified as [x_min, x_max].
            y_lim: A tuple of list containing the limits for the y-axis, specified as [y_min, y_max].
            caption: The (optional) caption to display below the plot.
            interactive: Whether users should be able to interact with the plot by panning or zooming with their mouse or trackpad.
            label: The (optional) label to display on the top left corner of the plot.
            show_label: Whether the label should be displayed.
            every: If `value` is a callable, run the function 'every' number of seconds while the client connection is open. Has no effect otherwise. Queue must be enabled. The event can be accessed (e.g. to cancel it) via this component's .load_event attribute.
            visible: Whether the plot should be visible.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.
        """
        self.x = x
        self.y = y
        self.color = color
        self.stroke_dash = stroke_dash
        self.tooltip = tooltip
        self.title = title
        self.x_title = x_title
        self.y_title = y_title
        self.color_legend_title = color_legend_title
        self.stroke_dash_legend_title = stroke_dash_legend_title
        self.color_legend_position = color_legend_position
        self.stroke_dash_legend_position = stroke_dash_legend_position
        self.overlay_point = overlay_point
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.caption = caption
        self.interactive_chart = interactive
        self.width = width
        self.height = height
        super().__init__(
            value=value,
            label=label,
            show_label=show_label,
            visible=visible,
            elem_id=elem_id,
            elem_classes=elem_classes,
            every=every,
        )

    def get_config(self):
        config = super().get_config()
        config["caption"] = self.caption
        return config

    def get_block_name(self) -> str:
        return "plot"

    @staticmethod
    def update(
        value: pd.DataFrame | dict | Literal[_Keywords.NO_VALUE] = _Keywords.NO_VALUE,
        x: str | None = None,
        y: str | None = None,
        color: str | None = None,
        stroke_dash: str | None = None,
        overlay_point: bool | None = None,
        title: str | None = None,
        tooltip: list[str] | str | None = None,
        x_title: str | None = None,
        y_title: str | None = None,
        color_legend_title: str | None = None,
        stroke_dash_legend_title: str | None = None,
        color_legend_position: str | None = None,
        stroke_dash_legend_position: str | None = None,
        height: int | None = None,
        width: int | None = None,
        x_lim: list[int] | None = None,
        y_lim: list[int] | None = None,
        interactive: bool | None = None,
        caption: str | None = None,
        label: str | None = None,
        show_label: bool | None = None,
        visible: bool | None = None,
    ):
        """Update an existing plot component.

        If updating any of the plot properties (color, size, etc) the value, x, and y parameters must be specified.

        Parameters:
            value: The pandas dataframe containing the data to display in a scatter plot.
            x: Column corresponding to the x axis.
            y: Column corresponding to the y axis.
            color: The column to determine the point color. If the column contains numeric data, gradio will interpolate the column data so that small values correspond to light colors and large values correspond to dark values.
            stroke_dash: The column to determine the symbol used to draw the line, e.g. dashed lines, dashed lines with points.
            overlay_point: Whether to draw a point on the line for each (x, y) coordinate pair.
            title: The title to display on top of the chart.
            tooltip: The column (or list of columns) to display on the tooltip when a user hovers a point on the plot.
            x_title: The title given to the x axis. By default, uses the value of the x parameter.
            y_title: The title given to the y axis. By default, uses the value of the y parameter.
            color_legend_title: The title given to the color legend. By default, uses the value of color parameter.
            stroke_dash_legend_title: The title given to the stroke legend. By default, uses the value of stroke parameter.
            color_legend_position: The position of the color legend. If the string value 'none' is passed, this legend is omitted. For other valid position values see: https://vega.github.io/vega/docs/legends/#orientation
            stroke_dash_legend_position: The position of the stoke_dash legend. If the string value 'none' is passed, this legend is omitted. For other valid position values see: https://vega.github.io/vega/docs/legends/#orientation
            height: The height of the plot in pixels.
            width: The width of the plot in pixels.
            x_lim: A tuple or list containing the limits for the x-axis, specified as [x_min, x_max].
            y_lim: A tuple of list containing the limits for the y-axis, specified as [y_min, y_max].
            caption: The (optional) caption to display below the plot.
            interactive: Whether users should be able to interact with the plot by panning or zooming with their mouse or trackpad.
            label: The (optional) label to display in the top left corner of the plot.
            show_label: Whether the label should be displayed.
            visible: Whether the plot should be visible.
        """
        properties = [
            x,
            y,
            color,
            stroke_dash,
            overlay_point,
            title,
            tooltip,
            x_title,
            y_title,
            color_legend_title,
            stroke_dash_legend_title,
            color_legend_position,
            stroke_dash_legend_position,
            height,
            width,
            x_lim,
            y_lim,
            interactive,
        ]
        if any(properties):
            if not isinstance(value, pd.DataFrame):
                raise ValueError(
                    "In order to update plot properties the value parameter "
                    "must be provided, and it must be a Dataframe. Please pass a value "
                    "parameter to gr.LinePlot.update."
                )
            if x is None or y is None:
                raise ValueError(
                    "In order to update plot properties, the x and y axis data "
                    "must be specified. Please pass valid values for x an y to "
                    "gr.LinePlot.update."
                )
            chart = LinePlot.create_plot(value, *properties)
            value = {"type": "altair", "plot": chart.to_json(), "chart": "line"}

        updated_config = {
            "label": label,
            "show_label": show_label,
            "visible": visible,
            "value": value,
            "caption": caption,
            "__type__": "update",
        }
        return updated_config

    @staticmethod
    def create_plot(
        value: pd.DataFrame,
        x: str,
        y: str,
        color: str | None = None,
        stroke_dash: str | None = None,
        overlay_point: bool | None = None,
        title: str | None = None,
        tooltip: list[str] | str | None = None,
        x_title: str | None = None,
        y_title: str | None = None,
        color_legend_title: str | None = None,
        stroke_dash_legend_title: str | None = None,
        color_legend_position: str | None = None,
        stroke_dash_legend_position: str | None = None,
        height: int | None = None,
        width: int | None = None,
        x_lim: list[int] | None = None,
        y_lim: list[int] | None = None,
        interactive: bool | None = None,
    ):
        """Helper for creating the scatter plot."""
        interactive = True if interactive is None else interactive
        encodings = {
            "x": alt.X(
                x,  # type: ignore
                title=x_title or x,  # type: ignore
                scale=AltairPlot.create_scale(x_lim),  # type: ignore
            ),
            "y": alt.Y(
                y,  # type: ignore
                title=y_title or y,  # type: ignore
                scale=AltairPlot.create_scale(y_lim),  # type: ignore
            ),
        }
        properties = {}
        if title:
            properties["title"] = title
        if height:
            properties["height"] = height
        if width:
            properties["width"] = width

        if color:
            domain = value[color].unique().tolist()
            range_ = list(range(len(domain)))
            encodings["color"] = {
                "field": color,
                "type": "nominal",
                "scale": {"domain": domain, "range": range_},
                "legend": AltairPlot.create_legend(
                    position=color_legend_position, title=color_legend_title or color
                ),
            }

        highlight = None
        if interactive and any([color, stroke_dash]):
            highlight = alt.selection(
                type="single",  # type: ignore
                on="mouseover",
                fields=[c for c in [color, stroke_dash] if c],
                nearest=True,
            )

        if stroke_dash:
            stroke_dash = {
                "field": stroke_dash,  # type: ignore
                "legend": AltairPlot.create_legend(  # type: ignore
                    position=stroke_dash_legend_position,  # type: ignore
                    title=stroke_dash_legend_title or stroke_dash,  # type: ignore
                ),  # type: ignore
            }  # type: ignore
        else:
            stroke_dash = alt.value(alt.Undefined)  # type: ignore

        if tooltip:
            encodings["tooltip"] = tooltip

        chart = alt.Chart(value).encode(**encodings)  # type: ignore

        points = chart.mark_point(clip=True).encode(
            opacity=alt.value(alt.Undefined) if overlay_point else alt.value(0),
        )
        lines = chart.mark_line(clip=True).encode(strokeDash=stroke_dash)

        if highlight:
            points = points.add_selection(highlight)

            lines = lines.encode(
                size=alt.condition(highlight, alt.value(4), alt.value(1)),
            )

        chart = (lines + points).properties(background="transparent", **properties)
        if interactive:
            chart = chart.interactive()

        return chart

    def postprocess(self, y: pd.DataFrame | dict | None) -> dict[str, str] | None:
        # if None or update
        if y is None or isinstance(y, Dict):
            return y
        if self.x is None or self.y is None:
            raise ValueError("No value provided for required parameters `x` and `y`.")
        chart = self.create_plot(
            value=y,
            x=self.x,
            y=self.y,
            color=self.color,
            overlay_point=self.overlay_point,
            title=self.title,
            tooltip=self.tooltip,
            x_title=self.x_title,
            y_title=self.y_title,
            color_legend_title=self.color_legend_title,
            color_legend_position=self.color_legend_position,
            stroke_dash_legend_title=self.stroke_dash_legend_title,
            stroke_dash_legend_position=self.stroke_dash_legend_position,
            x_lim=self.x_lim,
            y_lim=self.y_lim,
            stroke_dash=self.stroke_dash,
            interactive=self.interactive_chart,
            height=self.height,
            width=self.width,
        )

        return {"type": "altair", "plot": chart.to_json(), "chart": "line"}


@document()
class BarPlot(Plot):
    """
    Create a bar plot.

    Preprocessing: this component does *not* accept input.
    Postprocessing: expects a pandas dataframe with the data to plot.

    Demos: native_plots, chicago-bikeshare-dashboard
    """

    def __init__(
        self,
        value: pd.DataFrame | Callable | None = None,
        x: str | None = None,
        y: str | None = None,
        *,
        color: str | None = None,
        vertical: bool = True,
        group: str | None = None,
        title: str | None = None,
        tooltip: list[str] | str | None = None,
        x_title: str | None = None,
        y_title: str | None = None,
        color_legend_title: str | None = None,
        group_title: str | None = None,
        color_legend_position: str | None = None,
        height: int | None = None,
        width: int | None = None,
        y_lim: list[int] | None = None,
        caption: str | None = None,
        interactive: bool | None = True,
        label: str | None = None,
        show_label: bool = True,
        every: float | None = None,
        visible: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
    ):
        """
        Parameters:
            value: The pandas dataframe containing the data to display in a scatter plot.
            x: Column corresponding to the x axis.
            y: Column corresponding to the y axis.
            color: The column to determine the bar color. Must be categorical (discrete values).
            vertical: If True, the bars will be displayed vertically. If False, the x and y axis will be switched, displaying the bars horizontally. Default is True.
            group: The column with which to split the overall plot into smaller subplots.
            title: The title to display on top of the chart.
            tooltip: The column (or list of columns) to display on the tooltip when a user hovers over a bar.
            x_title: The title given to the x axis. By default, uses the value of the x parameter.
            y_title: The title given to the y axis. By default, uses the value of the y parameter.
            color_legend_title: The title given to the color legend. By default, uses the value of color parameter.
            group_title: The label displayed on top of the subplot columns (or rows if vertical=True). Use an empty string to omit.
            color_legend_position: The position of the color legend. If the string value 'none' is passed, this legend is omitted. For other valid position values see: https://vega.github.io/vega/docs/legends/#orientation.
            height: The height of the plot in pixels.
            width: The width of the plot in pixels.
            y_lim: A tuple of list containing the limits for the y-axis, specified as [y_min, y_max].
            caption: The (optional) caption to display below the plot.
            interactive: Whether users should be able to interact with the plot by panning or zooming with their mouse or trackpad.
            label: The (optional) label to display on the top left corner of the plot.
            show_label: Whether the label should be displayed.
            every: If `value` is a callable, run the function 'every' number of seconds while the client connection is open. Has no effect otherwise. Queue must be enabled. The event can be accessed (e.g. to cancel it) via this component's .load_event attribute.
            visible: Whether the plot should be visible.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.
        """
        self.x = x
        self.y = y
        self.color = color
        self.vertical = vertical
        self.group = group
        self.group_title = group_title
        self.tooltip = tooltip
        self.title = title
        self.x_title = x_title
        self.y_title = y_title
        self.color_legend_title = color_legend_title
        self.group_title = group_title
        self.color_legend_position = color_legend_position
        self.y_lim = y_lim
        self.caption = caption
        self.interactive_chart = interactive
        self.width = width
        self.height = height
        super().__init__(
            value=value,
            label=label,
            show_label=show_label,
            visible=visible,
            elem_id=elem_id,
            elem_classes=elem_classes,
            every=every,
        )

    def get_config(self):
        config = super().get_config()
        config["caption"] = self.caption
        return config

    def get_block_name(self) -> str:
        return "plot"

    @staticmethod
    def update(
        value: pd.DataFrame | dict | Literal[_Keywords.NO_VALUE] = _Keywords.NO_VALUE,
        x: str | None = None,
        y: str | None = None,
        color: str | None = None,
        vertical: bool = True,
        group: str | None = None,
        title: str | None = None,
        tooltip: list[str] | str | None = None,
        x_title: str | None = None,
        y_title: str | None = None,
        color_legend_title: str | None = None,
        group_title: str | None = None,
        color_legend_position: str | None = None,
        height: int | None = None,
        width: int | None = None,
        y_lim: list[int] | None = None,
        caption: str | None = None,
        interactive: bool | None = None,
        label: str | None = None,
        show_label: bool = True,
        visible: bool = True,
    ):
        """Update an existing BarPlot component.

        If updating any of the plot properties (color, size, etc) the value, x, and y parameters must be specified.

        Parameters:
            value: The pandas dataframe containing the data to display in a scatter plot.
            x: Column corresponding to the x axis.
            y: Column corresponding to the y axis.
            color: The column to determine the bar color. Must be categorical (discrete values).
            vertical: If True, the bars will be displayed vertically. If False, the x and y axis will be switched, displaying the bars horizontally. Default is True.
            group: The column with which to split the overall plot into smaller subplots.
            title: The title to display on top of the chart.
            tooltip: The column (or list of columns) to display on the tooltip when a user hovers over a bar.
            x_title: The title given to the x axis. By default, uses the value of the x parameter.
            y_title: The title given to the y axis. By default, uses the value of the y parameter.
            color_legend_title: The title given to the color legend. By default, uses the value of color parameter.
            group_title: The label displayed on top of the subplot columns (or rows if vertical=True). Use an empty string to omit.
            color_legend_position: The position of the color legend. If the string value 'none' is passed, this legend is omitted. For other valid position values see: https://vega.github.io/vega/docs/legends/#orientation.
            height: The height of the plot in pixels.
            width: The width of the plot in pixels.
            y_lim: A tuple of list containing the limits for the y-axis, specified as [y_min, y_max].
            caption: The (optional) caption to display below the plot.
            interactive: Whether users should be able to interact with the plot by panning or zooming with their mouse or trackpad.
            label: The (optional) label to display on the top left corner of the plot.
            show_label: Whether the label should be displayed.
            visible: Whether the plot should be visible.
        """
        properties = [
            x,
            y,
            color,
            vertical,
            group,
            title,
            tooltip,
            x_title,
            y_title,
            color_legend_title,
            group_title,
            color_legend_position,
            height,
            width,
            y_lim,
            interactive,
        ]
        if any(properties):
            if not isinstance(value, pd.DataFrame):
                raise ValueError(
                    "In order to update plot properties the value parameter "
                    "must be provided, and it must be a Dataframe. Please pass a value "
                    "parameter to gr.BarPlot.update."
                )
            if x is None or y is None:
                raise ValueError(
                    "In order to update plot properties, the x and y axis data "
                    "must be specified. Please pass valid values for x an y to "
                    "gr.BarPlot.update."
                )
            chart = BarPlot.create_plot(value, *properties)
            value = {"type": "altair", "plot": chart.to_json(), "chart": "bar"}

        updated_config = {
            "label": label,
            "show_label": show_label,
            "visible": visible,
            "value": value,
            "caption": caption,
            "__type__": "update",
        }
        return updated_config

    @staticmethod
    def create_plot(
        value: pd.DataFrame,
        x: str,
        y: str,
        color: str | None = None,
        vertical: bool = True,
        group: str | None = None,
        title: str | None = None,
        tooltip: list[str] | str | None = None,
        x_title: str | None = None,
        y_title: str | None = None,
        color_legend_title: str | None = None,
        group_title: str | None = None,
        color_legend_position: str | None = None,
        height: int | None = None,
        width: int | None = None,
        y_lim: list[int] | None = None,
        interactive: bool | None = True,
    ):
        """Helper for creating the bar plot."""
        interactive = True if interactive is None else interactive
        orientation = (
            {"field": group, "title": group_title if group_title is not None else group}
            if group
            else {}
        )

        x_title = x_title or x
        y_title = y_title or y

        # If horizontal, switch x and y
        if not vertical:
            y, x = x, y
            x = f"sum({x}):Q"
            y_title, x_title = x_title, y_title
            orientation = {"row": alt.Row(**orientation)} if orientation else {}  # type: ignore
            x_lim = y_lim
            y_lim = None
        else:
            y = f"sum({y}):Q"
            x_lim = None
            orientation = {"column": alt.Column(**orientation)} if orientation else {}  # type: ignore

        encodings = dict(
            x=alt.X(
                x,  # type: ignore
                title=x_title,  # type: ignore
                scale=AltairPlot.create_scale(x_lim),  # type: ignore
            ),
            y=alt.Y(
                y,  # type: ignore
                title=y_title,  # type: ignore
                scale=AltairPlot.create_scale(y_lim),  # type: ignore
            ),
            **orientation,
        )
        properties = {}
        if title:
            properties["title"] = title
        if height:
            properties["height"] = height
        if width:
            properties["width"] = width

        if color:
            domain = value[color].unique().tolist()
            range_ = list(range(len(domain)))
            encodings["color"] = {
                "field": color,
                "type": "nominal",
                "scale": {"domain": domain, "range": range_},
                "legend": AltairPlot.create_legend(
                    position=color_legend_position, title=color_legend_title or color
                ),
            }

        if tooltip:
            encodings["tooltip"] = tooltip

        chart = (
            alt.Chart(value)  # type: ignore
            .mark_bar()  # type: ignore
            .encode(**encodings)
            .properties(background="transparent", **properties)
        )
        if interactive:
            chart = chart.interactive()

        return chart

    def postprocess(self, y: pd.DataFrame | dict | None) -> dict[str, str] | None:
        # if None or update
        if y is None or isinstance(y, Dict):
            return y
        if self.x is None or self.y is None:
            raise ValueError("No value provided for required parameters `x` and `y`.")
        chart = self.create_plot(
            value=y,
            x=self.x,
            y=self.y,
            color=self.color,
            vertical=self.vertical,
            group=self.group,
            title=self.title,
            tooltip=self.tooltip,
            x_title=self.x_title,
            y_title=self.y_title,
            color_legend_title=self.color_legend_title,
            color_legend_position=self.color_legend_position,
            group_title=self.group_title,
            y_lim=self.y_lim,
            interactive=self.interactive_chart,
            height=self.height,
            width=self.width,
        )

        return {"type": "altair", "plot": chart.to_json(), "chart": "bar"}


@document()
class Markdown(IOComponent, Changeable, StringSerializable):
    """
    Used to render arbitrary Markdown output. Can also render latex enclosed by dollar signs.
    Preprocessing: this component does *not* accept input.
    Postprocessing: expects a valid {str} that can be rendered as Markdown.

    Demos: blocks_hello, blocks_kinematics
    Guides: key-features
    """

    def __init__(
        self,
        value: str | Callable = "",
        *,
        visible: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        **kwargs,
    ):
        """
        Parameters:
            value: Value to show in Markdown component. If callable, the function will be called whenever the app loads to set the initial value of the component.
            visible: If False, component will be hidden.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.
        """
        self.md = utils.get_markdown_parser()
        IOComponent.__init__(
            self,
            visible=visible,
            elem_id=elem_id,
            elem_classes=elem_classes,
            value=value,
            **kwargs,
        )

    def postprocess(self, y: str | None) -> str | None:
        """
        Parameters:
            y: markdown representation
        Returns:
            HTML rendering of markdown
        """
        if y is None:
            return None
        unindented_y = inspect.cleandoc(y)
        return self.md.render(unindented_y)

    def get_config(self):
        return {
            "value": self.value,
            **Component.get_config(self),
        }

    @staticmethod
    def update(
        value: Any | Literal[_Keywords.NO_VALUE] | None = _Keywords.NO_VALUE,
        visible: bool | None = None,
    ):
        updated_config = {
            "visible": visible,
            "value": value,
            "__type__": "update",
        }
        return updated_config

    def style(self):
        return self

    def as_example(self, input_data: str | None) -> str:
        postprocessed = self.postprocess(input_data)
        return postprocessed if postprocessed else ""


@document("languages")
class Code(Changeable, Inputable, IOComponent, StringSerializable):
    """
    Creates a Code editor for entering, editing or viewing code.
    Preprocessing: passes a {str} of code into the function.
    Postprocessing: expects the function to return a {str} of code or a single-elment {tuple}: (string filepath,)
    """

    languages = [
        "python",
        "markdown",
        "json",
        "html",
        "css",
        "javascript",
        "typescript",
        "yaml",
        "dockerfile",
        "shell",
        "r",
        None,
    ]

    def __init__(
        self,
        value: str | tuple[str] | None = None,
        language: str | None = None,
        *,
        lines: int = 5,
        label: str | None = None,
        interactive: bool | None = None,
        show_label: bool = True,
        visible: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        **kwargs,
    ):
        """
        Parameters:
            value: Default value to show in the code editor. If callable, the function will be called whenever the app loads to set the initial value of the component.
            language: The language to display the code as. Supported languages listed in `gr.Code.languages`.
            label: component name in interface.
            interactive: Whether user should be able to enter code or only view it.
            show_label: if True, will display label.
            visible: If False, component will be hidden.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.
        """
        assert language in Code.languages, f"Language {language} not supported."
        self.language = language
        self.lines = lines
        IOComponent.__init__(
            self,
            label=label,
            interactive=interactive,
            show_label=show_label,
            visible=visible,
            elem_id=elem_id,
            elem_classes=elem_classes,
            value=value,
            **kwargs,
        )

    def get_config(self):
        return {
            "value": self.value,
            "language": self.language,
            "lines": self.lines,
            **IOComponent.get_config(self),
        }

    def postprocess(self, y):
        if y is None:
            return None
        elif isinstance(y, tuple):
            with open(y[0]) as file_data:
                return file_data.read()
        else:
            return y.strip()

    @staticmethod
    def update(
        value: str
        | tuple[str]
        | None
        | Literal[_Keywords.NO_VALUE] = _Keywords.NO_VALUE,
        label: str | None = None,
        show_label: bool | None = None,
        visible: bool | None = None,
        language: str | None = None,
        interactive: bool | None = None,
    ):
        return {
            "label": label,
            "show_label": show_label,
            "visible": visible,
            "value": value,
            "language": language,
            "interactive": interactive,
            "__type__": "update",
        }

    def style(self):
        return self


############################
# Special Components
############################


@document("style")
class Dataset(Clickable, Selectable, Component, StringSerializable):
    """
    Used to create an output widget for showing datasets. Used to render the examples
    box.
    Preprocessing: passes the selected sample either as a {list} of data (if type="value") or as an {int} index (if type="index")
    Postprocessing: expects a {list} of {lists} corresponding to the dataset data.
    """

    def __init__(
        self,
        *,
        label: str | None = None,
        components: list[IOComponent] | list[str],
        samples: list[list[Any]] | None = None,
        headers: list[str] | None = None,
        type: str = "values",
        samples_per_page: int = 10,
        visible: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        **kwargs,
    ):
        """
        Parameters:
            components: Which component types to show in this dataset widget, can be passed in as a list of string names or Components instances. The following components are supported in a Dataset: Audio, Checkbox, CheckboxGroup, ColorPicker, Dataframe, Dropdown, File, HTML, Image, Markdown, Model3D, Number, Radio, Slider, Textbox, TimeSeries, Video
            samples: a nested list of samples. Each sublist within the outer list represents a data sample, and each element within the sublist represents an value for each component
            headers: Column headers in the Dataset widget, should be the same len as components. If not provided, inferred from component labels
            type: 'values' if clicking on a sample should pass the value of the sample, or "index" if it should pass the index of the sample
            samples_per_page: how many examples to show per page.
            visible: If False, component will be hidden.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.
        """
        Component.__init__(
            self, visible=visible, elem_id=elem_id, elem_classes=elem_classes, **kwargs
        )
        self.components = [get_component_instance(c, render=False) for c in components]

        # Narrow type to IOComponent
        assert all(
            isinstance(c, IOComponent) for c in self.components
        ), "All components in a `Dataset` must be subclasses of `IOComponent`"
        self.components = [c for c in self.components if isinstance(c, IOComponent)]
        for component in self.components:
            component.root_url = self.root_url

        self.samples = [[]] if samples is None else samples
        for example in self.samples:
            for i, (component, ex) in enumerate(zip(self.components, example)):
                example[i] = component.as_example(ex)
        self.type = type
        self.label = label
        if headers is not None:
            self.headers = headers
        elif all(c.label is None for c in self.components):
            self.headers = []
        else:
            self.headers = [c.label or "" for c in self.components]
        self.samples_per_page = samples_per_page

    def get_config(self):
        return {
            "components": [component.get_block_name() for component in self.components],
            "headers": self.headers,
            "samples": self.samples,
            "type": self.type,
            "label": self.label,
            "samples_per_page": self.samples_per_page,
            **Component.get_config(self),
        }

    @staticmethod
    def update(
        samples: Any | Literal[_Keywords.NO_VALUE] | None = _Keywords.NO_VALUE,
        visible: bool | None = None,
        label: str | None = None,
    ):
        return {
            "samples": samples,
            "visible": visible,
            "label": label,
            "__type__": "update",
        }

    def preprocess(self, x: Any) -> Any:
        """
        Any preprocessing needed to be performed on function input.
        """
        if self.type == "index":
            return x
        elif self.type == "values":
            return self.samples[x]

    def postprocess(self, samples: list[list[Any]]) -> dict:
        return {
            "samples": samples,
            "__type__": "update",
        }

    def style(self, **kwargs):
        """
        This method can be used to change the appearance of the Dataset component.
        """
        Component.style(self, **kwargs)
        return self


@document()
class Interpretation(Component, SimpleSerializable):
    """
    Used to create an interpretation widget for a component.
    Preprocessing: this component does *not* accept input.
    Postprocessing: expects a {dict} with keys "original" and "interpretation".

    Guides: custom-interpretations-with-blocks
    """

    def __init__(
        self,
        component: Component,
        *,
        visible: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        **kwargs,
    ):
        """
        Parameters:
            component: Which component to show in the interpretation widget.
            visible: Whether or not the interpretation is visible.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.
        """
        Component.__init__(
            self, visible=visible, elem_id=elem_id, elem_classes=elem_classes, **kwargs
        )
        self.component = component

    def get_config(self):
        return {
            "component": self.component.get_block_name(),
            "component_props": self.component.get_config(),
        }

    @staticmethod
    def update(
        value: Any | Literal[_Keywords.NO_VALUE] | None = _Keywords.NO_VALUE,
        visible: bool | None = None,
    ):
        return {
            "visible": visible,
            "value": value,
            "__type__": "update",
        }

    def style(self):
        return self


class StatusTracker(Component, SimpleSerializable):
    def __init__(
        self,
        **kwargs,
    ):
        warnings.warn("The StatusTracker component is deprecated.")


def component(cls_name: str) -> Component:
    obj = utils.component_or_layout_class(cls_name)()
    if isinstance(obj, BlockContext):
        raise ValueError(f"Invalid component: {obj.__class__}")
    return obj


def get_component_instance(comp: str | dict | Component, render=True) -> Component:
    if isinstance(comp, str):
        component_obj = component(comp)
        if not (render):
            component_obj.unrender()
        return component_obj
    elif isinstance(comp, dict):
        name = comp.pop("name")
        component_cls = utils.component_or_layout_class(name)
        component_obj = component_cls(**comp)
        if isinstance(component_obj, BlockContext):
            raise ValueError(f"Invalid component: {name}")
        if not (render):
            component_obj.unrender()
        return component_obj
    elif isinstance(comp, Component):
        return comp
    else:
        raise ValueError(
            f"Component must provided as a `str` or `dict` or `Component` but is {comp}"
        )


Text = Textbox
DataFrame = Dataframe
Highlightedtext = HighlightedText
Annotatedimage = AnnotatedImage
Highlight = HighlightedText
Checkboxgroup = CheckboxGroup
TimeSeries = Timeseries
Json = JSON
