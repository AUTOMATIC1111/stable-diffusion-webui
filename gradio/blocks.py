from __future__ import annotations

import copy
import inspect
import json
import os
import random
import secrets
import sys
import time
import warnings
import webbrowser
from abc import abstractmethod
from types import ModuleType
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable

import anyio
import requests
from anyio import CapacityLimiter
from gradio_client import serializing
from gradio_client import utils as client_utils
from gradio_client.documentation import document, set_documentation_group
from packaging import version
from typing_extensions import Literal

from gradio import (
    analytics,
    components,
    external,
    networking,
    queueing,
    routes,
    strings,
    themes,
    utils,
)
from gradio.context import Context
from gradio.deprecation import check_deprecated_parameters
from gradio.exceptions import DuplicateBlockError, InvalidApiNameError
from gradio.helpers import EventData, create_tracker, skip, special_args
from gradio.themes import Default as DefaultTheme
from gradio.themes import ThemeClass as Theme
from gradio.tunneling import BINARY_PATH, CURRENT_TUNNELS
from gradio.utils import (
    GRADIO_VERSION,
    TupleNoPrint,
    check_function_inputs_match,
    component_or_layout_class,
    delete_none,
    get_cancel_function,
    get_continuous_fn,
)

set_documentation_group("blocks")

if TYPE_CHECKING:  # Only import for type checking (is False at runtime).
    from fastapi.applications import FastAPI

    from gradio.components import Component

BUILT_IN_THEMES: dict[str, Theme] = {
    t.name: t
    for t in [
        themes.Base(),
        themes.Default(),
        themes.Monochrome(),
        themes.Soft(),
        themes.Glass(),
    ]
}


class Block:
    def __init__(
        self,
        *,
        render: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        visible: bool = True,
        root_url: str | None = None,  # URL that is prepended to all file paths
        _skip_init_processing: bool = False,  # Used for loading from Spaces
        **kwargs,
    ):
        self._id = Context.id
        Context.id += 1
        self.visible = visible
        self.elem_id = elem_id
        self.elem_classes = (
            [elem_classes] if isinstance(elem_classes, str) else elem_classes
        )
        self.root_url = root_url
        self.share_token = secrets.token_urlsafe(32)
        self._skip_init_processing = _skip_init_processing
        self._style = {}
        self.parent: BlockContext | None = None

        if render:
            self.render()
        check_deprecated_parameters(
            self.__class__.__name__, stacklevel=6, kwargs=kwargs
        )

    def render(self):
        """
        Adds self into appropriate BlockContext
        """
        if Context.root_block is not None and self._id in Context.root_block.blocks:
            raise DuplicateBlockError(
                f"A block with id: {self._id} has already been rendered in the current Blocks."
            )
        if Context.block is not None:
            Context.block.add(self)
        if Context.root_block is not None:
            Context.root_block.blocks[self._id] = self
            if isinstance(self, components.IOComponent):
                Context.root_block.temp_file_sets.append(self.temp_files)
        return self

    def unrender(self):
        """
        Removes self from BlockContext if it has been rendered (otherwise does nothing).
        Removes self from the layout and collection of blocks, but does not delete any event triggers.
        """
        if Context.block is not None:
            try:
                Context.block.children.remove(self)
            except ValueError:
                pass
        if Context.root_block is not None:
            try:
                del Context.root_block.blocks[self._id]
            except KeyError:
                pass
        return self

    def get_block_name(self) -> str:
        """
        Gets block's class name.

        If it is template component it gets the parent's class name.

        @return: class name
        """
        return (
            self.__class__.__base__.__name__.lower()
            if hasattr(self, "is_template")
            else self.__class__.__name__.lower()
        )

    def get_expected_parent(self) -> type[BlockContext] | None:
        return None

    def set_event_trigger(
        self,
        event_name: str,
        fn: Callable | None,
        inputs: Component | list[Component] | set[Component] | None,
        outputs: Component | list[Component] | None,
        preprocess: bool = True,
        postprocess: bool = True,
        scroll_to_output: bool = False,
        show_progress: bool = True,
        api_name: str | None = None,
        js: str | None = None,
        no_target: bool = False,
        queue: bool | None = None,
        batch: bool = False,
        max_batch_size: int = 4,
        cancels: list[int] | None = None,
        every: float | None = None,
        collects_event_data: bool | None = None,
        trigger_after: int | None = None,
        trigger_only_on_success: bool = False,
    ) -> tuple[dict[str, Any], int]:
        """
        Adds an event to the component's dependencies.
        Parameters:
            event_name: event name
            fn: Callable function
            inputs: input list
            outputs: output list
            preprocess: whether to run the preprocess methods of components
            postprocess: whether to run the postprocess methods of components
            scroll_to_output: whether to scroll to output of dependency on trigger
            show_progress: whether to show progress animation while running.
            api_name: Defining this parameter exposes the endpoint in the api docs
            js: Experimental parameter (API may change): Optional frontend js method to run before running 'fn'. Input arguments for js method are values of 'inputs' and 'outputs', return should be a list of values for output components
            no_target: if True, sets "targets" to [], used for Blocks "load" event
            queue: If True, will place the request on the queue, if the queue has been enabled. If False, will not put this event on the queue, even if the queue has been enabled. If None, will use the queue setting of the gradio app.
            batch: whether this function takes in a batch of inputs
            max_batch_size: the maximum batch size to send to the function
            cancels: a list of other events to cancel when this event is triggered. For example, setting cancels=[click_event] will cancel the click_event, where click_event is the return value of another components .click method.
            every: Run this event 'every' number of seconds while the client connection is open. Interpreted in seconds. Queue must be enabled.
            collects_event_data: whether to collect event data for this event
            trigger_after: if set, this event will be triggered after 'trigger_after' function index
            trigger_only_on_success: if True, this event will only be triggered if the previous event was successful (only applies if `trigger_after` is set)
        Returns: dependency information, dependency index
        """
        # Support for singular parameter
        if isinstance(inputs, set):
            inputs_as_dict = True
            inputs = sorted(inputs, key=lambda x: x._id)
        else:
            inputs_as_dict = False
            if inputs is None:
                inputs = []
            elif not isinstance(inputs, list):
                inputs = [inputs]

        if isinstance(outputs, set):
            outputs = sorted(outputs, key=lambda x: x._id)
        else:
            if outputs is None:
                outputs = []
            elif not isinstance(outputs, list):
                outputs = [outputs]

        if fn is not None and not cancels:
            check_function_inputs_match(fn, inputs, inputs_as_dict)

        if Context.root_block is None:
            raise AttributeError(
                f"{event_name}() and other events can only be called within a Blocks context."
            )
        if every is not None and every <= 0:
            raise ValueError("Parameter every must be positive or None")
        if every and batch:
            raise ValueError(
                f"Cannot run {event_name} event in a batch and every {every} seconds. "
                "Either batch is True or every is non-zero but not both."
            )

        if every and fn:
            fn = get_continuous_fn(fn, every)
        elif every:
            raise ValueError("Cannot set a value for `every` without a `fn`.")

        _, progress_index, event_data_index = (
            special_args(fn) if fn else (None, None, None)
        )
        Context.root_block.fns.append(
            BlockFunction(
                fn,
                inputs,
                outputs,
                preprocess,
                postprocess,
                inputs_as_dict,
                progress_index is not None,
            )
        )
        if api_name is not None:
            api_name_ = utils.append_unique_suffix(
                api_name, [dep["api_name"] for dep in Context.root_block.dependencies]
            )
            if api_name != api_name_:
                warnings.warn(f"api_name {api_name} already exists, using {api_name_}")
                api_name = api_name_

        if collects_event_data is None:
            collects_event_data = event_data_index is not None

        dependency = {
            "targets": [self._id] if not no_target else [],
            "trigger": event_name,
            "inputs": [block._id for block in inputs],
            "outputs": [block._id for block in outputs],
            "backend_fn": fn is not None,
            "js": js,
            "queue": False if fn is None else queue,
            "api_name": api_name,
            "scroll_to_output": scroll_to_output,
            "show_progress": show_progress,
            "every": every,
            "batch": batch,
            "max_batch_size": max_batch_size,
            "cancels": cancels or [],
            "types": {
                "continuous": bool(every),
                "generator": inspect.isgeneratorfunction(fn) or bool(every),
            },
            "collects_event_data": collects_event_data,
            "trigger_after": trigger_after,
            "trigger_only_on_success": trigger_only_on_success,
        }
        Context.root_block.dependencies.append(dependency)
        return dependency, len(Context.root_block.dependencies) - 1

    def get_config(self):
        return {
            "visible": self.visible,
            "elem_id": self.elem_id,
            "elem_classes": self.elem_classes,
            "style": self._style,
            "root_url": self.root_url,
        }

    @staticmethod
    @abstractmethod
    def update(**kwargs) -> dict:
        return {}

    @classmethod
    def get_specific_update(cls, generic_update: dict[str, Any]) -> dict:
        generic_update = generic_update.copy()
        del generic_update["__type__"]
        specific_update = cls.update(**generic_update)
        return specific_update


class BlockContext(Block):
    def __init__(
        self,
        visible: bool = True,
        render: bool = True,
        **kwargs,
    ):
        """
        Parameters:
            visible: If False, this will be hidden but included in the Blocks config file (its visibility can later be updated).
            render: If False, this will not be included in the Blocks config file at all.
        """
        self.children: list[Block] = []
        Block.__init__(self, visible=visible, render=render, **kwargs)

    def __enter__(self):
        self.parent = Context.block
        Context.block = self
        return self

    def add(self, child: Block):
        child.parent = self
        self.children.append(child)

    def fill_expected_parents(self):
        children = []
        pseudo_parent = None
        for child in self.children:
            expected_parent = child.get_expected_parent()
            if not expected_parent or isinstance(self, expected_parent):
                pseudo_parent = None
                children.append(child)
            else:
                if pseudo_parent is not None and isinstance(
                    pseudo_parent, expected_parent
                ):
                    pseudo_parent.children.append(child)
                else:
                    pseudo_parent = expected_parent(render=False)
                    children.append(pseudo_parent)
                    pseudo_parent.children = [child]
                    if Context.root_block:
                        Context.root_block.blocks[pseudo_parent._id] = pseudo_parent
                child.parent = pseudo_parent
        self.children = children

    def __exit__(self, *args):
        if getattr(self, "allow_expected_parents", True):
            self.fill_expected_parents()
        Context.block = self.parent

    def postprocess(self, y):
        """
        Any postprocessing needed to be performed on a block context.
        """
        return y


class BlockFunction:
    def __init__(
        self,
        fn: Callable | None,
        inputs: list[Component],
        outputs: list[Component],
        preprocess: bool,
        postprocess: bool,
        inputs_as_dict: bool,
        tracks_progress: bool = False,
    ):
        self.fn = fn
        self.inputs = inputs
        self.outputs = outputs
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.tracks_progress = tracks_progress
        self.total_runtime = 0
        self.total_runs = 0
        self.inputs_as_dict = inputs_as_dict
        self.name = getattr(fn, "__name__", "fn") if fn is not None else None

    def __str__(self):
        return str(
            {
                "fn": self.name,
                "preprocess": self.preprocess,
                "postprocess": self.postprocess,
            }
        )

    def __repr__(self):
        return str(self)


class class_or_instancemethod(classmethod):  # noqa: N801
    def __get__(self, instance, type_):
        descr_get = super().__get__ if instance is None else self.__func__.__get__
        return descr_get(instance, type_)


def postprocess_update_dict(block: Block, update_dict: dict, postprocess: bool = True):
    """
    Converts a dictionary of updates into a format that can be sent to the frontend.
    E.g. {"__type__": "generic_update", "value": "2", "interactive": False}
    Into -> {"__type__": "update", "value": 2.0, "mode": "static"}

    Parameters:
        block: The Block that is being updated with this update dictionary.
        update_dict: The original update dictionary
        postprocess: Whether to postprocess the "value" key of the update dictionary.
    """
    if update_dict.get("__type__", "") == "generic_update":
        update_dict = block.get_specific_update(update_dict)
    if update_dict.get("value") is components._Keywords.NO_VALUE:
        update_dict.pop("value")
    interactive = update_dict.pop("interactive", None)
    if interactive is not None:
        update_dict["mode"] = "dynamic" if interactive else "static"
    prediction_value = delete_none(update_dict, skip_value=True)
    if "value" in prediction_value and postprocess:
        assert isinstance(
            block, components.IOComponent
        ), f"Component {block.__class__} does not support value"
        prediction_value["value"] = block.postprocess(prediction_value["value"])
    return prediction_value


def convert_component_dict_to_list(
    outputs_ids: list[int], predictions: dict
) -> list | dict:
    """
    Converts a dictionary of component updates into a list of updates in the order of
    the outputs_ids and including every output component. Leaves other types of dictionaries unchanged.
    E.g. {"textbox": "hello", "number": {"__type__": "generic_update", "value": "2"}}
    Into -> ["hello", {"__type__": "generic_update"}, {"__type__": "generic_update", "value": "2"}]
    """
    keys_are_blocks = [isinstance(key, Block) for key in predictions]
    if all(keys_are_blocks):
        reordered_predictions = [skip() for _ in outputs_ids]
        for component, value in predictions.items():
            if component._id not in outputs_ids:
                raise ValueError(
                    f"Returned component {component} not specified as output of function."
                )
            output_index = outputs_ids.index(component._id)
            reordered_predictions[output_index] = value
        predictions = utils.resolve_singleton(reordered_predictions)
    elif any(keys_are_blocks):
        raise ValueError(
            "Returned dictionary included some keys as Components. Either all keys must be Components to assign Component values, or return a List of values to assign output values in order."
        )
    return predictions


def get_api_info(config: dict, serialize: bool = True):
    """
    Gets the information needed to generate the API docs from a Blocks config.
    Parameters:
        config: a Blocks config dictionary
        serialize: If True, returns the serialized version of the typed information. If False, returns the raw version.
    """
    api_info = {"named_endpoints": {}, "unnamed_endpoints": {}}
    mode = config.get("mode", None)
    after_new_format = version.parse(config.get("version", "2.0")) > version.Version(
        "3.28.3"
    )

    for d, dependency in enumerate(config["dependencies"]):
        dependency_info = {"parameters": [], "returns": []}
        skip_endpoint = False
        skip_components = ["state"]

        inputs = dependency["inputs"]
        for i in inputs:
            for component in config["components"]:
                if component["id"] == i:
                    break
            else:
                skip_endpoint = True  # if component not found, skip endpoint
                break
            type = component["type"]
            if (
                not component.get("serializer")
                and type not in serializing.COMPONENT_MAPPING
            ):
                skip_endpoint = True  # if component not serializable, skip endpoint
                break
            if type in skip_components:
                continue
            label = component["props"].get("label", f"parameter_{i}")
            # The config has the most specific API info (taking into account the parameters
            # of the component), so we use that if it exists. Otherwise, we fallback to the
            # Serializer's API info.
            serializer = serializing.COMPONENT_MAPPING[type]()
            if component.get("api_info") and after_new_format:
                info = component["api_info"]
                example = component["example_inputs"]["serialized"]
            else:
                assert isinstance(serializer, serializing.Serializable)
                info = serializer.api_info()
                example = serializer.example_inputs()["raw"]
            python_info = info["info"]
            if serialize and info["serialized_info"]:
                python_info = serializer.serialized_info()
                if (
                    isinstance(serializer, serializing.FileSerializable)
                    and component["props"].get("file_count", "single") != "single"
                ):
                    python_info = serializer._multiple_file_serialized_info()

            python_type = client_utils.json_schema_to_python_type(python_info)
            serializer_name = serializing.COMPONENT_MAPPING[type].__name__
            dependency_info["parameters"].append(
                {
                    "label": label,
                    "type": info["info"],
                    "python_type": {
                        "type": python_type,
                        "description": python_info.get("description", ""),
                    },
                    "component": type.capitalize(),
                    "example_input": example,
                    "serializer": serializer_name,
                }
            )

        outputs = dependency["outputs"]
        for o in outputs:
            for component in config["components"]:
                if component["id"] == o:
                    break
            else:
                skip_endpoint = True  # if component not found, skip endpoint
                break
            type = component["type"]
            if (
                not component.get("serializer")
                and type not in serializing.COMPONENT_MAPPING
            ):
                skip_endpoint = True  # if component not serializable, skip endpoint
                break
            if type in skip_components:
                continue
            label = component["props"].get("label", f"value_{o}")
            serializer = serializing.COMPONENT_MAPPING[type]()
            if component.get("api_info") and after_new_format:
                info = component["api_info"]
                example = component["example_inputs"]["serialized"]
            else:
                assert isinstance(serializer, serializing.Serializable)
                info = serializer.api_info()
                example = serializer.example_inputs()["raw"]
            python_info = info["info"]
            if serialize and info["serialized_info"]:
                python_info = serializer.serialized_info()
                if (
                    isinstance(serializer, serializing.FileSerializable)
                    and component["props"].get("file_count", "single") != "single"
                ):
                    python_info = serializer._multiple_file_serialized_info()
            python_type = client_utils.json_schema_to_python_type(python_info)
            serializer_name = serializing.COMPONENT_MAPPING[type].__name__
            dependency_info["returns"].append(
                {
                    "label": label,
                    "type": info["info"],
                    "python_type": {
                        "type": python_type,
                        "description": python_info.get("description", ""),
                    },
                    "component": type.capitalize(),
                    "serializer": serializer_name,
                }
            )

        if not dependency["backend_fn"]:
            skip_endpoint = True

        if skip_endpoint:
            continue
        if dependency["api_name"]:
            api_info["named_endpoints"][f"/{dependency['api_name']}"] = dependency_info
        elif mode == "interface" or mode == "tabbed_interface":
            pass  # Skip unnamed endpoints in interface mode
        else:
            api_info["unnamed_endpoints"][str(d)] = dependency_info

    return api_info


@document("launch", "queue", "integrate", "load")
class Blocks(BlockContext):
    """
    Blocks is Gradio's low-level API that allows you to create more custom web
    applications and demos than Interfaces (yet still entirely in Python).


    Compared to the Interface class, Blocks offers more flexibility and control over:
    (1) the layout of components (2) the events that
    trigger the execution of functions (3) data flows (e.g. inputs can trigger outputs,
    which can trigger the next level of outputs). Blocks also offers ways to group
    together related demos such as with tabs.


    The basic usage of Blocks is as follows: create a Blocks object, then use it as a
    context (with the "with" statement), and then define layouts, components, or events
    within the Blocks context. Finally, call the launch() method to launch the demo.

    Example:
        import gradio as gr
        def update(name):
            return f"Welcome to Gradio, {name}!"

        with gr.Blocks() as demo:
            gr.Markdown("Start typing below and then click **Run** to see the output.")
            with gr.Row():
                inp = gr.Textbox(placeholder="What is your name?")
                out = gr.Textbox()
            btn = gr.Button("Run")
            btn.click(fn=update, inputs=inp, outputs=out)

        demo.launch()
    Demos: blocks_hello, blocks_flipper, blocks_speech_text_sentiment, generate_english_german, sound_alert
    Guides: blocks-and-event-listeners, controlling-layout, state-in-blocks, custom-CSS-and-JS, custom-interpretations-with-blocks, using-blocks-like-functions
    """

    def __init__(
        self,
        theme: Theme | str | None = None,
        analytics_enabled: bool | None = None,
        mode: str = "blocks",
        title: str = "Gradio",
        css: str | None = None,
        **kwargs,
    ):
        """
        Parameters:
            theme: a Theme object or a string representing a theme. If a string, will look for a built-in theme with that name (e.g. "soft" or "default"), or will attempt to load a theme from the HF Hub (e.g. "gradio/monochrome"). If None, will use the Default theme.
            analytics_enabled: whether to allow basic telemetry. If None, will use GRADIO_ANALYTICS_ENABLED environment variable or default to True.
            mode: a human-friendly name for the kind of Blocks or Interface being created.
            title: The tab title to display when this is opened in a browser window.
            css: custom css or path to custom css file to apply to entire Blocks
        """
        # Cleanup shared parameters with Interface #TODO: is this part still necessary after Interface with Blocks?
        self.limiter = None
        self.save_to = None
        if theme is None:
            theme = DefaultTheme()
        elif isinstance(theme, str):
            if theme.lower() in BUILT_IN_THEMES:
                theme = BUILT_IN_THEMES[theme.lower()]
            else:
                try:
                    theme = Theme.from_hub(theme)
                except Exception as e:
                    warnings.warn(f"Cannot load {theme}. Caught Exception: {str(e)}")
                    theme = DefaultTheme()
        if not isinstance(theme, Theme):
            warnings.warn("Theme should be a class loaded from gradio.themes")
            theme = DefaultTheme()
        self.theme: Theme = theme
        self.theme_css = theme._get_theme_css()
        self.stylesheets = theme._stylesheets
        self.encrypt = False
        self.share = False
        self.enable_queue = None
        self.max_threads = 40
        self.show_error = True
        if css is not None and os.path.exists(css):
            with open(css) as css_file:
                self.css = css_file.read()
        else:
            self.css = css

        # For analytics_enabled and allow_flagging: (1) first check for
        # parameter, (2) check for env variable, (3) default to True/"manual"
        self.analytics_enabled = (
            analytics_enabled
            if analytics_enabled is not None
            else analytics.analytics_enabled()
        )
        if not self.analytics_enabled:
            os.environ["HF_HUB_DISABLE_TELEMETRY"] = "True"
        super().__init__(render=False, **kwargs)
        self.blocks: dict[int, Block] = {}
        self.fns: list[BlockFunction] = []
        self.dependencies = []
        self.mode = mode

        self.is_running = False
        self.local_url = None
        self.share_url = None
        self.width = None
        self.height = None
        self.api_open = True

        self.is_space = os.getenv("SYSTEM") == "spaces"
        self.favicon_path = None
        self.auth = None
        self.dev_mode = True
        self.app_id = random.getrandbits(64)
        self.temp_file_sets = []
        self.title = title
        self.show_api = True

        # Only used when an Interface is loaded from a config
        self.predict = None
        self.input_components = None
        self.output_components = None
        self.__name__ = None
        self.api_mode = None
        self.progress_tracking = None
        self.ssl_verify = True

        self.allowed_paths = []
        self.blocked_paths = []
        self.root_path = ""

        if self.analytics_enabled:
            is_custom_theme = not any(
                self.theme.to_dict() == built_in_theme.to_dict()
                for built_in_theme in BUILT_IN_THEMES.values()
            )
            data = {
                "mode": self.mode,
                "custom_css": self.css is not None,
                "theme": self.theme.name,
                "is_custom_theme": is_custom_theme,
                "version": GRADIO_VERSION,
            }
            analytics.initiated_analytics(data)

    @classmethod
    def from_config(
        cls,
        config: dict,
        fns: list[Callable],
        root_url: str | None = None,
    ) -> Blocks:
        """
        Factory method that creates a Blocks from a config and list of functions.

        Parameters:
        config: a dictionary containing the configuration of the Blocks.
        fns: a list of functions that are used in the Blocks. Must be in the same order as the dependencies in the config.
        root_url: an optional root url to use for the components in the Blocks. Allows serving files from an external URL.
        """
        config = copy.deepcopy(config)
        components_config = config["components"]
        theme = config.get("theme", "default")
        original_mapping: dict[int, Block] = {}

        def get_block_instance(id: int) -> Block:
            for block_config in components_config:
                if block_config["id"] == id:
                    break
            else:
                raise ValueError(f"Cannot find block with id {id}")
            cls = component_or_layout_class(block_config["type"])
            block_config["props"].pop("type", None)
            block_config["props"].pop("name", None)
            style = block_config["props"].pop("style", None)
            if block_config["props"].get("root_url") is None and root_url:
                block_config["props"]["root_url"] = f"{root_url}/"
            # Any component has already processed its initial value, so we skip that step here
            block = cls(**block_config["props"], _skip_init_processing=True)
            if style and isinstance(block, components.IOComponent):
                block.style(**style)
            return block

        def iterate_over_children(children_list):
            for child_config in children_list:
                id = child_config["id"]
                block = get_block_instance(id)

                original_mapping[id] = block

                children = child_config.get("children")
                if children is not None:
                    assert isinstance(
                        block, BlockContext
                    ), f"Invalid config, Block with id {id} has children but is not a BlockContext."
                    with block:
                        iterate_over_children(children)

        derived_fields = ["types"]

        with Blocks(theme=theme) as blocks:
            # ID 0 should be the root Blocks component
            original_mapping[0] = Context.root_block or blocks

            iterate_over_children(config["layout"]["children"])

            first_dependency = None

            # add the event triggers
            for dependency, fn in zip(config["dependencies"], fns):
                # We used to add a "fake_event" to the config to cache examples
                # without removing it. This was causing bugs in calling gr.load
                # We fixed the issue by removing "fake_event" from the config in examples.py
                # but we still need to skip these events when loading the config to support
                # older demos
                if dependency["trigger"] == "fake_event":
                    continue
                for field in derived_fields:
                    dependency.pop(field, None)
                targets = dependency.pop("targets")
                trigger = dependency.pop("trigger")
                dependency.pop("backend_fn")
                dependency.pop("documentation", None)
                dependency["inputs"] = [
                    original_mapping[i] for i in dependency["inputs"]
                ]
                dependency["outputs"] = [
                    original_mapping[o] for o in dependency["outputs"]
                ]
                dependency.pop("status_tracker", None)
                dependency["preprocess"] = False
                dependency["postprocess"] = False

                for target in targets:
                    dependency = original_mapping[target].set_event_trigger(
                        event_name=trigger, fn=fn, **dependency
                    )[0]
                    if first_dependency is None:
                        first_dependency = dependency

            # Allows some use of Interface-specific methods with loaded Spaces
            if first_dependency and Context.root_block:
                blocks.predict = [fns[0]]
                blocks.input_components = [
                    Context.root_block.blocks[i] for i in first_dependency["inputs"]
                ]
                blocks.output_components = [
                    Context.root_block.blocks[o] for o in first_dependency["outputs"]
                ]
                blocks.__name__ = "Interface"
                blocks.api_mode = True

        return blocks

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        num_backend_fns = len([d for d in self.dependencies if d["backend_fn"]])
        repr = f"Gradio Blocks instance: {num_backend_fns} backend functions"
        repr += f"\n{'-' * len(repr)}"
        for d, dependency in enumerate(self.dependencies):
            if dependency["backend_fn"]:
                repr += f"\nfn_index={d}"
                repr += "\n inputs:"
                for input_id in dependency["inputs"]:
                    block = self.blocks[input_id]
                    repr += f"\n |-{block}"
                repr += "\n outputs:"
                for output_id in dependency["outputs"]:
                    block = self.blocks[output_id]
                    repr += f"\n |-{block}"
        return repr

    def render(self):
        if Context.root_block is not None:
            if self._id in Context.root_block.blocks:
                raise DuplicateBlockError(
                    f"A block with id: {self._id} has already been rendered in the current Blocks."
                )
            overlapping_ids = set(Context.root_block.blocks).intersection(self.blocks)
            for id in overlapping_ids:
                # State components are allowed to be reused between Blocks
                if not isinstance(self.blocks[id], components.State):
                    raise DuplicateBlockError(
                        "At least one block in this Blocks has already been rendered."
                    )

            Context.root_block.blocks.update(self.blocks)
            Context.root_block.fns.extend(self.fns)
            dependency_offset = len(Context.root_block.dependencies)
            for i, dependency in enumerate(self.dependencies):
                api_name = dependency["api_name"]
                if api_name is not None:
                    api_name_ = utils.append_unique_suffix(
                        api_name,
                        [dep["api_name"] for dep in Context.root_block.dependencies],
                    )
                    if api_name != api_name_:
                        warnings.warn(
                            f"api_name {api_name} already exists, using {api_name_}"
                        )
                        dependency["api_name"] = api_name_
                dependency["cancels"] = [
                    c + dependency_offset for c in dependency["cancels"]
                ]
                if dependency.get("trigger_after") is not None:
                    dependency["trigger_after"] += dependency_offset
                # Recreate the cancel function so that it has the latest
                # dependency fn indices. This is necessary to properly cancel
                # events in the backend
                if dependency["cancels"]:
                    updated_cancels = [
                        Context.root_block.dependencies[i]
                        for i in dependency["cancels"]
                    ]
                    new_fn = BlockFunction(
                        get_cancel_function(updated_cancels)[0],
                        [],
                        [],
                        False,
                        True,
                        False,
                    )
                    Context.root_block.fns[dependency_offset + i] = new_fn
                Context.root_block.dependencies.append(dependency)
            Context.root_block.temp_file_sets.extend(self.temp_file_sets)

        if Context.block is not None:
            Context.block.children.extend(self.children)
        return self

    def is_callable(self, fn_index: int = 0) -> bool:
        """Checks if a particular Blocks function is callable (i.e. not stateful or a generator)."""
        block_fn = self.fns[fn_index]
        dependency = self.dependencies[fn_index]

        if inspect.isasyncgenfunction(block_fn.fn):
            return False
        if inspect.isgeneratorfunction(block_fn.fn):
            return False
        for input_id in dependency["inputs"]:
            block = self.blocks[input_id]
            if getattr(block, "stateful", False):
                return False
        for output_id in dependency["outputs"]:
            block = self.blocks[output_id]
            if getattr(block, "stateful", False):
                return False

        return True

    def __call__(self, *inputs, fn_index: int = 0, api_name: str | None = None):
        """
        Allows Blocks objects to be called as functions. Supply the parameters to the
        function as positional arguments. To choose which function to call, use the
        fn_index parameter, which must be a keyword argument.

        Parameters:
        *inputs: the parameters to pass to the function
        fn_index: the index of the function to call (defaults to 0, which for Interfaces, is the default prediction function)
        api_name: The api_name of the dependency to call. Will take precedence over fn_index.
        """
        if api_name is not None:
            inferred_fn_index = next(
                (
                    i
                    for i, d in enumerate(self.dependencies)
                    if d.get("api_name") == api_name
                ),
                None,
            )
            if inferred_fn_index is None:
                raise InvalidApiNameError(
                    f"Cannot find a function with api_name {api_name}"
                )
            fn_index = inferred_fn_index
        if not (self.is_callable(fn_index)):
            raise ValueError(
                "This function is not callable because it is either stateful or is a generator. Please use the .launch() method instead to create an interactive user interface."
            )

        inputs = list(inputs)
        processed_inputs = self.serialize_data(fn_index, inputs)
        batch = self.dependencies[fn_index]["batch"]
        if batch:
            processed_inputs = [[inp] for inp in processed_inputs]

        outputs = client_utils.synchronize_async(
            self.process_api,
            fn_index=fn_index,
            inputs=processed_inputs,
            request=None,
            state={},
        )
        outputs = outputs["data"]

        if batch:
            outputs = [out[0] for out in outputs]

        processed_outputs = self.deserialize_data(fn_index, outputs)
        processed_outputs = utils.resolve_singleton(processed_outputs)

        return processed_outputs

    async def call_function(
        self,
        fn_index: int,
        processed_input: list[Any],
        iterator: AsyncIterator[Any] | None = None,
        requests: routes.Request | list[routes.Request] | None = None,
        event_id: str | None = None,
        event_data: EventData | None = None,
    ):
        """
        Calls function with given index and preprocessed input, and measures process time.
        Parameters:
            fn_index: index of function to call
            processed_input: preprocessed input to pass to function
            iterator: iterator to use if function is a generator
            requests: requests to pass to function
            event_id: id of event in queue
            event_data: data associated with event trigger
        """
        block_fn = self.fns[fn_index]
        assert block_fn.fn, f"function with index {fn_index} not defined."
        is_generating = False

        if block_fn.inputs_as_dict:
            processed_input = [dict(zip(block_fn.inputs, processed_input))]

        request = requests[0] if isinstance(requests, list) else requests
        processed_input, progress_index, _ = special_args(
            block_fn.fn, processed_input, request, event_data
        )
        progress_tracker = (
            processed_input[progress_index] if progress_index is not None else None
        )

        start = time.time()

        if iterator is None:  # If not a generator function that has already run
            if progress_tracker is not None and progress_index is not None:
                progress_tracker, fn = create_tracker(
                    self, event_id, block_fn.fn, progress_tracker.track_tqdm
                )
                processed_input[progress_index] = progress_tracker
            else:
                fn = block_fn.fn

            if inspect.iscoroutinefunction(fn):
                prediction = await fn(*processed_input)
            else:
                prediction = await anyio.to_thread.run_sync(
                    fn, *processed_input, limiter=self.limiter
                )
        else:
            prediction = None

        if inspect.isgeneratorfunction(block_fn.fn) or inspect.isasyncgenfunction(
            block_fn.fn
        ):
            if not self.enable_queue:
                raise ValueError("Need to enable queue to use generators.")
            try:
                if iterator is None:
                    iterator = prediction
                if inspect.isgenerator(iterator):
                    iterator = utils.SyncToAsyncIterator(iterator, self.limiter)
                prediction = await utils.async_iteration(iterator)
                is_generating = True
            except StopAsyncIteration:
                n_outputs = len(self.dependencies[fn_index].get("outputs"))
                prediction = (
                    components._Keywords.FINISHED_ITERATING
                    if n_outputs == 1
                    else (components._Keywords.FINISHED_ITERATING,) * n_outputs
                )
                iterator = None

        duration = time.time() - start

        return {
            "prediction": prediction,
            "duration": duration,
            "is_generating": is_generating,
            "iterator": iterator,
        }

    def serialize_data(self, fn_index: int, inputs: list[Any]) -> list[Any]:
        dependency = self.dependencies[fn_index]
        processed_input = []

        for i, input_id in enumerate(dependency["inputs"]):
            block = self.blocks[input_id]
            assert isinstance(
                block, components.IOComponent
            ), f"{block.__class__} Component with id {input_id} not a valid input component."
            serialized_input = block.serialize(inputs[i])
            processed_input.append(serialized_input)

        return processed_input

    def deserialize_data(self, fn_index: int, outputs: list[Any]) -> list[Any]:
        dependency = self.dependencies[fn_index]
        predictions = []

        for o, output_id in enumerate(dependency["outputs"]):
            block = self.blocks[output_id]
            assert isinstance(
                block, components.IOComponent
            ), f"{block.__class__} Component with id {output_id} not a valid output component."
            deserialized = block.deserialize(
                outputs[o], root_url=block.root_url, hf_token=Context.hf_token
            )
            predictions.append(deserialized)

        return predictions

    def validate_inputs(self, fn_index: int, inputs: list[Any]):
        block_fn = self.fns[fn_index]
        dependency = self.dependencies[fn_index]

        dep_inputs = dependency["inputs"]

        # This handles incorrect inputs when args are changed by a JS function
        # Only check not enough args case, ignore extra arguments (for now)
        # TODO: make this stricter?
        if len(inputs) < len(dep_inputs):
            name = (
                f" ({block_fn.name})"
                if block_fn.name and block_fn.name != "<lambda>"
                else ""
            )

            wanted_args = []
            received_args = []
            for input_id in dep_inputs:
                block = self.blocks[input_id]
                wanted_args.append(str(block))
            for inp in inputs:
                v = f'"{inp}"' if isinstance(inp, str) else str(inp)
                received_args.append(v)

            wanted = ", ".join(wanted_args)
            received = ", ".join(received_args)

            # JS func didn't pass enough arguments
            raise ValueError(
                f"""An event handler{name} didn't receive enough input values (needed: {len(dep_inputs)}, got: {len(inputs)}).
Check if the event handler calls a Javascript function, and make sure its return value is correct.
Wanted inputs:
    [{wanted}]
Received inputs:
    [{received}]"""
            )

    def preprocess_data(self, fn_index: int, inputs: list[Any], state: dict[int, Any]):
        block_fn = self.fns[fn_index]
        dependency = self.dependencies[fn_index]

        self.validate_inputs(fn_index, inputs)

        if block_fn.preprocess:
            processed_input = []
            for i, input_id in enumerate(dependency["inputs"]):
                block = self.blocks[input_id]
                assert isinstance(
                    block, components.Component
                ), f"{block.__class__} Component with id {input_id} not a valid input component."
                if getattr(block, "stateful", False):
                    processed_input.append(state.get(input_id))
                else:
                    processed_input.append(block.preprocess(inputs[i]))
        else:
            processed_input = inputs
        return processed_input

    def validate_outputs(self, fn_index: int, predictions: Any | list[Any]):
        block_fn = self.fns[fn_index]
        dependency = self.dependencies[fn_index]

        dep_outputs = dependency["outputs"]

        if type(predictions) is not list and type(predictions) is not tuple:
            predictions = [predictions]

        if len(predictions) < len(dep_outputs):
            name = (
                f" ({block_fn.name})"
                if block_fn.name and block_fn.name != "<lambda>"
                else ""
            )

            wanted_args = []
            received_args = []
            for output_id in dep_outputs:
                block = self.blocks[output_id]
                wanted_args.append(str(block))
            for pred in predictions:
                v = f'"{pred}"' if isinstance(pred, str) else str(pred)
                received_args.append(v)

            wanted = ", ".join(wanted_args)
            received = ", ".join(received_args)

            raise ValueError(
                f"""An event handler{name} didn't receive enough output values (needed: {len(dep_outputs)}, received: {len(predictions)}).
Wanted outputs:
    [{wanted}]
Received outputs:
    [{received}]"""
            )

    def postprocess_data(
        self, fn_index: int, predictions: list | dict, state: dict[int, Any]
    ):
        block_fn = self.fns[fn_index]
        dependency = self.dependencies[fn_index]
        batch = dependency["batch"]

        if type(predictions) is dict and len(predictions) > 0:
            predictions = convert_component_dict_to_list(
                dependency["outputs"], predictions
            )

        if len(dependency["outputs"]) == 1 and not (batch):
            predictions = [
                predictions,
            ]

        self.validate_outputs(fn_index, predictions)  # type: ignore

        output = []
        for i, output_id in enumerate(dependency["outputs"]):
            try:
                if predictions[i] is components._Keywords.FINISHED_ITERATING:
                    output.append(None)
                    continue
            except (IndexError, KeyError) as err:
                raise ValueError(
                    "Number of output components does not match number "
                    f"of values returned from from function {block_fn.name}"
                ) from err
            block = self.blocks[output_id]
            if getattr(block, "stateful", False):
                if not utils.is_update(predictions[i]):
                    state[output_id] = predictions[i]
                output.append(None)
            else:
                prediction_value = predictions[i]
                if utils.is_update(prediction_value):
                    assert isinstance(prediction_value, dict)
                    prediction_value = postprocess_update_dict(
                        block=block,
                        update_dict=prediction_value,
                        postprocess=block_fn.postprocess,
                    )
                elif block_fn.postprocess:
                    assert isinstance(
                        block, components.Component
                    ), f"{block.__class__} Component with id {output_id} not a valid output component."
                    prediction_value = block.postprocess(prediction_value)
                output.append(prediction_value)

        return output

    async def process_api(
        self,
        fn_index: int,
        inputs: list[Any],
        state: dict[int, Any],
        request: routes.Request | list[routes.Request] | None = None,
        iterators: dict[int, Any] | None = None,
        event_id: str | None = None,
        event_data: EventData | None = None,
    ) -> dict[str, Any]:
        """
        Processes API calls from the frontend. First preprocesses the data,
        then runs the relevant function, then postprocesses the output.
        Parameters:
            fn_index: Index of function to run.
            inputs: input data received from the frontend
            state: data stored from stateful components for session (key is input block id)
            request: the gr.Request object containing information about the network request (e.g. IP address, headers, query parameters, username)
            iterators: the in-progress iterators for each generator function (key is function index)
            event_id: id of event that triggered this API call
            event_data: data associated with the event trigger itself
        Returns: None
        """
        block_fn = self.fns[fn_index]
        batch = self.dependencies[fn_index]["batch"]

        if batch:
            max_batch_size = self.dependencies[fn_index]["max_batch_size"]
            batch_sizes = [len(inp) for inp in inputs]
            batch_size = batch_sizes[0]
            if inspect.isasyncgenfunction(block_fn.fn) or inspect.isgeneratorfunction(
                block_fn.fn
            ):
                raise ValueError("Gradio does not support generators in batch mode.")
            if not all(x == batch_size for x in batch_sizes):
                raise ValueError(
                    f"All inputs to a batch function must have the same length but instead have sizes: {batch_sizes}."
                )
            if batch_size > max_batch_size:
                raise ValueError(
                    f"Batch size ({batch_size}) exceeds the max_batch_size for this function ({max_batch_size})"
                )

            inputs = [
                self.preprocess_data(fn_index, list(i), state) for i in zip(*inputs)
            ]
            result = await self.call_function(
                fn_index, list(zip(*inputs)), None, request, event_id, event_data
            )
            preds = result["prediction"]
            data = [
                self.postprocess_data(fn_index, list(o), state) for o in zip(*preds)
            ]
            data = list(zip(*data))
            is_generating, iterator = None, None
        else:
            inputs = self.preprocess_data(fn_index, inputs, state)
            iterator = iterators.get(fn_index, None) if iterators else None
            result = await self.call_function(
                fn_index, inputs, iterator, request, event_id, event_data
            )
            data = self.postprocess_data(fn_index, result["prediction"], state)
            is_generating, iterator = result["is_generating"], result["iterator"]

        block_fn.total_runtime += result["duration"]
        block_fn.total_runs += 1
        return {
            "data": data,
            "is_generating": is_generating,
            "iterator": iterator,
            "duration": result["duration"],
            "average_duration": block_fn.total_runtime / block_fn.total_runs,
        }

    async def create_limiter(self):
        self.limiter = (
            None
            if self.max_threads == 40
            else CapacityLimiter(total_tokens=self.max_threads)
        )

    def get_config(self):
        return {"type": "column"}

    def get_config_file(self):
        config = {
            "version": routes.VERSION,
            "mode": self.mode,
            "dev_mode": self.dev_mode,
            "analytics_enabled": self.analytics_enabled,
            "components": [],
            "css": self.css,
            "title": self.title or "Gradio",
            "is_space": self.is_space,
            "enable_queue": getattr(self, "enable_queue", False),  # launch attributes
            "show_error": getattr(self, "show_error", False),
            "show_api": self.show_api,
            "is_colab": utils.colab_check(),
            "stylesheets": self.stylesheets,
            "theme": self.theme.name,
        }

        def get_layout(block):
            if not isinstance(block, BlockContext):
                return {"id": block._id}
            children_layout = []
            for child in block.children:
                children_layout.append(get_layout(child))
            return {"id": block._id, "children": children_layout}

        config["layout"] = get_layout(self)

        for _id, block in self.blocks.items():
            props = block.get_config() if hasattr(block, "get_config") else {}
            block_config = {
                "id": _id,
                "type": block.get_block_name(),
                "props": utils.delete_none(props),
            }
            serializer = utils.get_serializer_name(block)
            if serializer:
                assert isinstance(block, serializing.Serializable)
                block_config["serializer"] = serializer
                block_config["api_info"] = block.api_info()  # type: ignore
                block_config["example_inputs"] = block.example_inputs()  # type: ignore
            config["components"].append(block_config)
        config["dependencies"] = self.dependencies
        return config

    def __enter__(self):
        if Context.block is None:
            Context.root_block = self
        self.parent = Context.block
        Context.block = self
        self.exited = False
        return self

    def __exit__(self, *args):
        super().fill_expected_parents()
        Context.block = self.parent
        # Configure the load events before root_block is reset
        self.attach_load_events()
        if self.parent is None:
            Context.root_block = None
        else:
            self.parent.children.extend(self.children)
        self.config = self.get_config_file()
        self.app = routes.App.create_app(self)
        self.progress_tracking = any(block_fn.tracks_progress for block_fn in self.fns)
        self.exited = True

    @class_or_instancemethod
    def load(
        self_or_cls,  # noqa: N805
        fn: Callable | None = None,
        inputs: list[Component] | None = None,
        outputs: list[Component] | None = None,
        api_name: str | None = None,
        scroll_to_output: bool = False,
        show_progress: bool = True,
        queue=None,
        batch: bool = False,
        max_batch_size: int = 4,
        preprocess: bool = True,
        postprocess: bool = True,
        every: float | None = None,
        _js: str | None = None,
        *,
        name: str | None = None,
        src: str | None = None,
        api_key: str | None = None,
        alias: str | None = None,
        **kwargs,
    ) -> Blocks | dict[str, Any] | None:
        """
        For reverse compatibility reasons, this is both a class method and an instance
        method, the two of which, confusingly, do two completely different things.


        Class method: loads a demo from a Hugging Face Spaces repo and creates it locally and returns a block instance. Warning: this method will be deprecated. Use the equivalent `gradio.load()` instead.


        Instance method: adds event that runs as soon as the demo loads in the browser. Example usage below.
        Parameters:
            name: Class Method - the name of the model (e.g. "gpt2" or "facebook/bart-base") or space (e.g. "flax-community/spanish-gpt2"), can include the `src` as prefix (e.g. "models/facebook/bart-base")
            src: Class Method - the source of the model: `models` or `spaces` (or leave empty if source is provided as a prefix in `name`)
            api_key: Class Method - optional access token for loading private Hugging Face Hub models or spaces. Find your token here: https://huggingface.co/settings/tokens. Warning: only provide this if you are loading a trusted private Space as it can be read by the Space you are loading.
            alias: Class Method - optional string used as the name of the loaded model instead of the default name (only applies if loading a Space running Gradio 2.x)
            fn: Instance Method - the function to wrap an interface around. Often a machine learning model's prediction function. Each parameter of the function corresponds to one input component, and the function should return a single value or a tuple of values, with each element in the tuple corresponding to one output component.
            inputs: Instance Method - List of gradio.components to use as inputs. If the function takes no inputs, this should be an empty list.
            outputs: Instance Method - List of gradio.components to use as inputs. If the function returns no outputs, this should be an empty list.
            api_name: Instance Method - Defining this parameter exposes the endpoint in the api docs
            scroll_to_output: Instance Method - If True, will scroll to output component on completion
            show_progress: Instance Method - If True, will show progress animation while pending
            queue: Instance Method - If True, will place the request on the queue, if the queue exists
            batch: Instance Method - If True, then the function should process a batch of inputs, meaning that it should accept a list of input values for each parameter. The lists should be of equal length (and be up to length `max_batch_size`). The function is then *required* to return a tuple of lists (even if there is only 1 output component), with each list in the tuple corresponding to one output component.
            max_batch_size: Instance Method - Maximum number of inputs to batch together if this is called from the queue (only relevant if batch=True)
            preprocess: Instance Method - If False, will not run preprocessing of component data before running 'fn' (e.g. leaving it as a base64 string if this method is called with the `Image` component).
            postprocess: Instance Method - If False, will not run postprocessing of component data before returning 'fn' output to the browser.
            every: Instance Method - Run this event 'every' number of seconds. Interpreted in seconds. Queue must be enabled.
        Example:
            import gradio as gr
            import datetime
            with gr.Blocks() as demo:
                def get_time():
                    return datetime.datetime.now().time()
                dt = gr.Textbox(label="Current time")
                demo.load(get_time, inputs=None, outputs=dt)
            demo.launch()
        """
        if isinstance(self_or_cls, type):
            warnings.warn("gr.Blocks.load() will be deprecated. Use gr.load() instead.")
            if name is None:
                raise ValueError(
                    "Blocks.load() requires passing parameters as keyword arguments"
                )
            return external.load(
                name=name, src=src, hf_token=api_key, alias=alias, **kwargs
            )
        else:
            from gradio.events import Dependency

            dep, dep_index = self_or_cls.set_event_trigger(
                event_name="load",
                fn=fn,
                inputs=inputs,
                outputs=outputs,
                api_name=api_name,
                preprocess=preprocess,
                postprocess=postprocess,
                scroll_to_output=scroll_to_output,
                show_progress=show_progress,
                js=_js,
                queue=queue,
                batch=batch,
                max_batch_size=max_batch_size,
                every=every,
                no_target=True,
            )
            return Dependency(self_or_cls, dep, dep_index)

    def clear(self):
        """Resets the layout of the Blocks object."""
        self.blocks = {}
        self.fns = []
        self.dependencies = []
        self.children = []
        return self

    @document()
    def queue(
        self,
        concurrency_count: int = 1,
        status_update_rate: float | Literal["auto"] = "auto",
        client_position_to_load_data: int | None = None,
        default_enabled: bool | None = None,
        api_open: bool = True,
        max_size: int | None = None,
    ):
        """
        You can control the rate of processed requests by creating a queue. This will allow you to set the number of requests to be processed at one time, and will let users know their position in the queue.
        Parameters:
            concurrency_count: Number of worker threads that will be processing requests from the queue concurrently. Increasing this number will increase the rate at which requests are processed, but will also increase the memory usage of the queue.
            status_update_rate: If "auto", Queue will send status estimations to all clients whenever a job is finished. Otherwise Queue will send status at regular intervals set by this parameter as the number of seconds.
            client_position_to_load_data: DEPRECATED. This parameter is deprecated and has no effect.
            default_enabled: Deprecated and has no effect.
            api_open: If True, the REST routes of the backend will be open, allowing requests made directly to those endpoints to skip the queue.
            max_size: The maximum number of events the queue will store at any given moment. If the queue is full, new events will not be added and a user will receive a message saying that the queue is full. If None, the queue size will be unlimited.
        Example: (Blocks)
            with gr.Blocks() as demo:
                button = gr.Button(label="Generate Image")
                button.click(fn=image_generator, inputs=gr.Textbox(), outputs=gr.Image())
            demo.queue(concurrency_count=3)
            demo.launch()
        Example: (Interface)
            demo = gr.Interface(image_generator, gr.Textbox(), gr.Image())
            demo.queue(concurrency_count=3)
            demo.launch()
        """
        if default_enabled is not None:
            warnings.warn(
                "The default_enabled parameter of queue has no effect and will be removed "
                "in a future version of gradio."
            )
        self.enable_queue = True
        self.api_open = api_open
        if client_position_to_load_data is not None:
            warnings.warn("The client_position_to_load_data parameter is deprecated.")
        self._queue = queueing.Queue(
            live_updates=status_update_rate == "auto",
            concurrency_count=concurrency_count,
            update_intervals=status_update_rate if status_update_rate != "auto" else 1,
            max_size=max_size,
            blocks_dependencies=self.dependencies,
        )
        self.config = self.get_config_file()
        self.app = routes.App.create_app(self)
        return self

    def validate_queue_settings(self):
        if not self.enable_queue and self.progress_tracking:
            raise ValueError("Progress tracking requires queuing to be enabled.")

        for fn_index, dep in enumerate(self.dependencies):
            if not self.enable_queue and self.queue_enabled_for_fn(fn_index):
                raise ValueError(
                    f"The queue is enabled for event {dep['api_name'] if dep['api_name'] else fn_index} "
                    "but the queue has not been enabled for the app. Please call .queue() "
                    "on your app. Consult https://gradio.app/docs/#blocks-queue for information on how "
                    "to configure the queue."
                )
            for i in dep["cancels"]:
                if not self.queue_enabled_for_fn(i):
                    raise ValueError(
                        "Queue needs to be enabled! "
                        "You may get this error by either 1) passing a function that uses the yield keyword "
                        "into an interface without enabling the queue or 2) defining an event that cancels "
                        "another event without enabling the queue. Both can be solved by calling .queue() "
                        "before .launch()"
                    )
            if dep["batch"] and (
                dep["queue"] is False
                or (dep["queue"] is None and not self.enable_queue)
            ):
                raise ValueError("In order to use batching, the queue must be enabled.")

    def launch(
        self,
        inline: bool | None = None,
        inbrowser: bool = False,
        share: bool | None = None,
        debug: bool = False,
        enable_queue: bool | None = None,
        max_threads: int = 40,
        auth: Callable | tuple[str, str] | list[tuple[str, str]] | None = None,
        auth_message: str | None = None,
        prevent_thread_lock: bool = False,
        show_error: bool = False,
        server_name: str | None = None,
        server_port: int | None = None,
        show_tips: bool = False,
        height: int = 500,
        width: int | str = "100%",
        encrypt: bool | None = None,
        favicon_path: str | None = None,
        ssl_keyfile: str | None = None,
        ssl_certfile: str | None = None,
        ssl_keyfile_password: str | None = None,
        ssl_verify: bool = True,
        quiet: bool = False,
        show_api: bool = True,
        file_directories: list[str] | None = None,
        allowed_paths: list[str] | None = None,
        blocked_paths: list[str] | None = None,
        root_path: str = "",
        _frontend: bool = True,
        app_kwargs: dict[str, Any] | None = None,
    ) -> tuple[FastAPI, str, str]:
        """
        Launches a simple web server that serves the demo. Can also be used to create a
        public link used by anyone to access the demo from their browser by setting share=True.

        Parameters:
            inline: whether to display in the interface inline in an iframe. Defaults to True in python notebooks; False otherwise.
            inbrowser: whether to automatically launch the interface in a new tab on the default browser.
            share: whether to create a publicly shareable link for the interface. Creates an SSH tunnel to make your UI accessible from anywhere. If not provided, it is set to False by default every time, except when running in Google Colab. When localhost is not accessible (e.g. Google Colab), setting share=False is not supported.
            debug: if True, blocks the main thread from running. If running in Google Colab, this is needed to print the errors in the cell output.
            auth: If provided, username and password (or list of username-password tuples) required to access interface. Can also provide function that takes username and password and returns True if valid login.
            auth_message: If provided, HTML message provided on login page.
            prevent_thread_lock: If True, the interface will block the main thread while the server is running.
            show_error: If True, any errors in the interface will be displayed in an alert modal and printed in the browser console log
            server_port: will start gradio app on this port (if available). Can be set by environment variable GRADIO_SERVER_PORT. If None, will search for an available port starting at 7860.
            server_name: to make app accessible on local network, set this to "0.0.0.0". Can be set by environment variable GRADIO_SERVER_NAME. If None, will use "127.0.0.1".
            show_tips: if True, will occasionally show tips about new Gradio features
            enable_queue: DEPRECATED (use .queue() method instead.) if True, inference requests will be served through a queue instead of with parallel threads. Required for longer inference times (> 1min) to prevent timeout. The default option in HuggingFace Spaces is True. The default option elsewhere is False.
            max_threads: the maximum number of total threads that the Gradio app can generate in parallel. The default is inherited from the starlette library (currently 40). Applies whether the queue is enabled or not. But if queuing is enabled, this parameter is increaseed to be at least the concurrency_count of the queue.
            width: The width in pixels of the iframe element containing the interface (used if inline=True)
            height: The height in pixels of the iframe element containing the interface (used if inline=True)
            encrypt: DEPRECATED. Has no effect.
            favicon_path: If a path to a file (.png, .gif, or .ico) is provided, it will be used as the favicon for the web page.
            ssl_keyfile: If a path to a file is provided, will use this as the private key file to create a local server running on https.
            ssl_certfile: If a path to a file is provided, will use this as the signed certificate for https. Needs to be provided if ssl_keyfile is provided.
            ssl_keyfile_password: If a password is provided, will use this with the ssl certificate for https.
            ssl_verify: If False, skips certificate validation which allows self-signed certificates to be used.
            quiet: If True, suppresses most print statements.
            show_api: If True, shows the api docs in the footer of the app. Default True. If the queue is enabled, then api_open parameter of .queue() will determine if the api docs are shown, independent of the value of show_api.
            file_directories: This parameter has been renamed to `allowed_paths`. It will be removed in a future version.
            allowed_paths: List of complete filepaths or parent directories that gradio is allowed to serve (in addition to the directory containing the gradio python file). Must be absolute paths. Warning: if you provide directories, any files in these directories or their subdirectories are accessible to all users of your app.
            blocked_paths: List of complete filepaths or parent directories that gradio is not allowed to serve (i.e. users of your app are not allowed to access). Must be absolute paths. Warning: takes precedence over `allowed_paths` and all other directories exposed by Gradio by default.
            root_path: The root path (or "mount point") of the application, if it's not served from the root ("/") of the domain. Often used when the application is behind a reverse proxy that forwards requests to the application. For example, if the application is served at "https://example.com/myapp", the `root_path` should be set to "/myapp".
            app_kwargs: Additional keyword arguments to pass to the underlying FastAPI app as a dictionary of parameter keys and argument values. For example, `{"docs_url": "/docs"}`
        Returns:
            app: FastAPI app object that is running the demo
            local_url: Locally accessible link to the demo
            share_url: Publicly accessible link to the demo (if share=True, otherwise None)
        Example: (Blocks)
            import gradio as gr
            def reverse(text):
                return text[::-1]
            with gr.Blocks() as demo:
                button = gr.Button(value="Reverse")
                button.click(reverse, gr.Textbox(), gr.Textbox())
            demo.launch(share=True, auth=("username", "password"))
        Example:  (Interface)
            import gradio as gr
            def reverse(text):
                return text[::-1]
            demo = gr.Interface(reverse, "text", "text")
            demo.launch(share=True, auth=("username", "password"))
        """
        if not self.exited:
            self.__exit__()

        self.dev_mode = False
        if (
            auth
            and not callable(auth)
            and not isinstance(auth[0], tuple)
            and not isinstance(auth[0], list)
        ):
            self.auth = [auth]
        else:
            self.auth = auth
        self.auth_message = auth_message
        self.show_tips = show_tips
        self.show_error = show_error
        self.height = height
        self.width = width
        self.favicon_path = favicon_path
        self.ssl_verify = ssl_verify
        self.root_path = root_path

        if enable_queue is not None:
            self.enable_queue = enable_queue
            warnings.warn(
                "The `enable_queue` parameter has been deprecated. Please use the `.queue()` method instead.",
                DeprecationWarning,
            )
        if encrypt is not None:
            warnings.warn(
                "The `encrypt` parameter has been deprecated and has no effect.",
                DeprecationWarning,
            )

        if self.is_space:
            self.enable_queue = self.enable_queue is not False
        else:
            self.enable_queue = self.enable_queue is True
        if self.enable_queue and not hasattr(self, "_queue"):
            self.queue()
        self.show_api = self.api_open if self.enable_queue else show_api

        if file_directories is not None:
            warnings.warn(
                "The `file_directories` parameter has been renamed to `allowed_paths`. Please use that instead.",
                DeprecationWarning,
            )
            if allowed_paths is None:
                allowed_paths = file_directories
        self.allowed_paths = allowed_paths or []
        self.blocked_paths = blocked_paths or []

        if not isinstance(self.allowed_paths, list):
            raise ValueError("`allowed_paths` must be a list of directories.")
        if not isinstance(self.blocked_paths, list):
            raise ValueError("`blocked_paths` must be a list of directories.")

        self.validate_queue_settings()

        self.config = self.get_config_file()
        self.max_threads = max(
            self._queue.max_thread_count if self.enable_queue else 0, max_threads
        )

        if self.is_running:
            assert isinstance(
                self.local_url, str
            ), f"Invalid local_url: {self.local_url}"
            if not (quiet):
                print(
                    "Rerunning server... use `close()` to stop if you need to change `launch()` parameters.\n----"
                )
        else:
            server_name, server_port, local_url, app, server = networking.start_server(
                self,
                server_name,
                server_port,
                ssl_keyfile,
                ssl_certfile,
                ssl_keyfile_password,
                app_kwargs=app_kwargs,
            )
            self.server_name = server_name
            self.local_url = local_url
            self.server_port = server_port
            self.server_app = app
            self.server = server
            self.is_running = True
            self.is_colab = utils.colab_check()
            self.is_kaggle = utils.kaggle_check()
            self.is_sagemaker = utils.sagemaker_check()

            self.protocol = (
                "https"
                if self.local_url.startswith("https") or self.is_colab
                else "http"
            )

            if self.enable_queue:
                self._queue.set_url(self.local_url)

            # Cannot run async functions in background other than app's scope.
            # Workaround by triggering the app endpoint
            requests.get(f"{self.local_url}startup-events", verify=ssl_verify)

        utils.launch_counter()

        if share is None:
            if self.is_colab and self.enable_queue:
                if not quiet:
                    print(
                        "Setting queue=True in a Colab notebook requires sharing enabled. Setting `share=True` (you can turn this off by setting `share=False` in `launch()` explicitly).\n"
                    )
                self.share = True
            elif self.is_kaggle:
                if not quiet:
                    print(
                        "Kaggle notebooks require sharing enabled. Setting `share=True` (you can turn this off by setting `share=False` in `launch()` explicitly).\n"
                    )
                self.share = True
            elif self.is_sagemaker:
                if not quiet:
                    print(
                        "Sagemaker notebooks may require sharing enabled. Setting `share=True` (you can turn this off by setting `share=False` in `launch()` explicitly).\n"
                    )
                self.share = True
            else:
                self.share = False
        else:
            self.share = share

        # If running in a colab or not able to access localhost,
        # a shareable link must be created.
        if _frontend and (not networking.url_ok(self.local_url)) and (not self.share):
            raise ValueError(
                "When localhost is not accessible, a shareable link must be created. Please set share=True or check your proxy settings to allow access to localhost."
            )

        if self.is_colab:
            if not quiet:
                if debug:
                    print(strings.en["COLAB_DEBUG_TRUE"])
                else:
                    print(strings.en["COLAB_DEBUG_FALSE"])
                if not self.share:
                    print(strings.en["COLAB_WARNING"].format(self.server_port))
            if self.enable_queue and not self.share:
                raise ValueError(
                    "When using queueing in Colab, a shareable link must be created. Please set share=True."
                )
        else:
            print(
                strings.en["RUNNING_LOCALLY_SEPARATED"].format(
                    self.protocol, self.server_name, self.server_port
                )
            )

        if self.share:
            if self.is_space:
                raise RuntimeError("Share is not supported when you are in Spaces")
            try:
                if self.share_url is None:
                    self.share_url = networking.setup_tunnel(
                        self.server_name, self.server_port, self.share_token
                    )
                print(strings.en["SHARE_LINK_DISPLAY"].format(self.share_url))
                if not (quiet):
                    print(strings.en["SHARE_LINK_MESSAGE"])
            except (RuntimeError, requests.exceptions.ConnectionError):
                if self.analytics_enabled:
                    analytics.error_analytics("Not able to set up tunnel")
                self.share_url = None
                self.share = False
                print(strings.en["COULD_NOT_GET_SHARE_LINK"].format(BINARY_PATH))
        else:
            if not (quiet):
                print(strings.en["PUBLIC_SHARE_TRUE"])
            self.share_url = None

        if inbrowser:
            link = self.share_url if self.share and self.share_url else self.local_url
            webbrowser.open(link)

        # Check if running in a Python notebook in which case, display inline
        if inline is None:
            inline = utils.ipython_check() and (self.auth is None)
        if inline:
            if self.auth is not None:
                print(
                    "Warning: authentication is not supported inline. Please"
                    "click the link to access the interface in a new tab."
                )
            try:
                from IPython.display import HTML, Javascript, display  # type: ignore

                if self.share and self.share_url:
                    while not networking.url_ok(self.share_url):
                        time.sleep(0.25)
                    display(
                        HTML(
                            f'<div><iframe src="{self.share_url}" width="{self.width}" height="{self.height}" allow="autoplay; camera; microphone; clipboard-read; clipboard-write;" frameborder="0" allowfullscreen></iframe></div>'
                        )
                    )
                elif self.is_colab:
                    # modified from /usr/local/lib/python3.7/dist-packages/google/colab/output/_util.py within Colab environment
                    code = """(async (port, path, width, height, cache, element) => {
                        if (!google.colab.kernel.accessAllowed && !cache) {
                            return;
                        }
                        element.appendChild(document.createTextNode(''));
                        const url = await google.colab.kernel.proxyPort(port, {cache});

                        const external_link = document.createElement('div');
                        external_link.innerHTML = `
                            <div style="font-family: monospace; margin-bottom: 0.5rem">
                                Running on <a href=${new URL(path, url).toString()} target="_blank">
                                    https://localhost:${port}${path}
                                </a>
                            </div>
                        `;
                        element.appendChild(external_link);

                        const iframe = document.createElement('iframe');
                        iframe.src = new URL(path, url).toString();
                        iframe.height = height;
                        iframe.allow = "autoplay; camera; microphone; clipboard-read; clipboard-write;"
                        iframe.width = width;
                        iframe.style.border = 0;
                        element.appendChild(iframe);
                    })""" + "({port}, {path}, {width}, {height}, {cache}, window.element)".format(
                        port=json.dumps(self.server_port),
                        path=json.dumps("/"),
                        width=json.dumps(self.width),
                        height=json.dumps(self.height),
                        cache=json.dumps(False),
                    )

                    display(Javascript(code))
                else:
                    display(
                        HTML(
                            f'<div><iframe src="{self.local_url}" width="{self.width}" height="{self.height}" allow="autoplay; camera; microphone; clipboard-read; clipboard-write;" frameborder="0" allowfullscreen></iframe></div>'
                        )
                    )
            except ImportError:
                pass

        if getattr(self, "analytics_enabled", False):
            data = {
                "launch_method": "browser" if inbrowser else "inline",
                "is_google_colab": self.is_colab,
                "is_sharing_on": self.share,
                "share_url": self.share_url,
                "enable_queue": self.enable_queue,
                "show_tips": self.show_tips,
                "server_name": server_name,
                "server_port": server_port,
                "is_spaces": self.is_space,
                "mode": self.mode,
            }
            analytics.launched_analytics(self, data)

        utils.show_tip(self)

        # Block main thread if debug==True
        if debug or int(os.getenv("GRADIO_DEBUG", 0)) == 1:
            self.block_thread()
        # Block main thread if running in a script to stop script from exiting
        is_in_interactive_mode = bool(getattr(sys, "ps1", sys.flags.interactive))

        if not prevent_thread_lock and not is_in_interactive_mode:
            self.block_thread()

        return TupleNoPrint((self.server_app, self.local_url, self.share_url))

    def integrate(
        self,
        comet_ml=None,
        wandb: ModuleType | None = None,
        mlflow: ModuleType | None = None,
    ) -> None:
        """
        A catch-all method for integrating with other libraries. This method should be run after launch()
        Parameters:
            comet_ml: If a comet_ml Experiment object is provided, will integrate with the experiment and appear on Comet dashboard
            wandb: If the wandb module is provided, will integrate with it and appear on WandB dashboard
            mlflow: If the mlflow module  is provided, will integrate with the experiment and appear on ML Flow dashboard
        """
        analytics_integration = ""
        if comet_ml is not None:
            analytics_integration = "CometML"
            comet_ml.log_other("Created from", "Gradio")
            if self.share_url is not None:
                comet_ml.log_text(f"gradio: {self.share_url}")
                comet_ml.end()
            elif self.local_url:
                comet_ml.log_text(f"gradio: {self.local_url}")
                comet_ml.end()
            else:
                raise ValueError("Please run `launch()` first.")
        if wandb is not None:
            analytics_integration = "WandB"
            if self.share_url is not None:
                wandb.log(
                    {
                        "Gradio panel": wandb.Html(
                            '<iframe src="'
                            + self.share_url
                            + '" width="'
                            + str(self.width)
                            + '" height="'
                            + str(self.height)
                            + '" frameBorder="0"></iframe>'
                        )
                    }
                )
            else:
                print(
                    "The WandB integration requires you to "
                    "`launch(share=True)` first."
                )
        if mlflow is not None:
            analytics_integration = "MLFlow"
            if self.share_url is not None:
                mlflow.log_param("Gradio Interface Share Link", self.share_url)
            else:
                mlflow.log_param("Gradio Interface Local Link", self.local_url)
        if self.analytics_enabled and analytics_integration:
            data = {"integration": analytics_integration}
            analytics.integration_analytics(data)

    def close(self, verbose: bool = True) -> None:
        """
        Closes the Interface that was launched and frees the port.
        """
        try:
            if self.enable_queue:
                self._queue.close()
            self.server.close()
            self.is_running = False
            # So that the startup events (starting the queue)
            # happen the next time the app is launched
            self.app.startup_events_triggered = False
            if verbose:
                print(f"Closing server running on port: {self.server_port}")
        except (AttributeError, OSError):  # can't close if not running
            pass

    def block_thread(
        self,
    ) -> None:
        """Block main thread until interrupted by user."""
        try:
            while True:
                time.sleep(0.1)
        except (KeyboardInterrupt, OSError):
            print("Keyboard interruption in main thread... closing server.")
            self.server.close()
            for tunnel in CURRENT_TUNNELS:
                tunnel.kill()

    def attach_load_events(self):
        """Add a load event for every component whose initial value should be randomized."""
        if Context.root_block:
            for component in Context.root_block.blocks.values():
                if (
                    isinstance(component, components.IOComponent)
                    and component.load_event_to_attach
                ):
                    load_fn, every = component.load_event_to_attach
                    # Use set_event_trigger to avoid ambiguity between load class/instance method
                    dep = self.set_event_trigger(
                        "load",
                        load_fn,
                        None,
                        component,
                        no_target=True,
                        # If every is None, for sure skip the queue
                        # else, let the enable_queue parameter take precedence
                        # this will raise a nice error message is every is used
                        # without queue
                        queue=False if every is None else None,
                        every=every,
                    )[0]
                    component.load_event = dep

    def startup_events(self):
        """Events that should be run when the app containing this block starts up."""

        if self.enable_queue:
            utils.run_coro_in_background(
                self._queue.start, self.progress_tracking, self.ssl_verify
            )
            # So that processing can resume in case the queue was stopped
            self._queue.stopped = False
        utils.run_coro_in_background(self.create_limiter)

    def queue_enabled_for_fn(self, fn_index: int):
        if self.dependencies[fn_index]["queue"] is None:
            return self.enable_queue
        return self.dependencies[fn_index]["queue"]
