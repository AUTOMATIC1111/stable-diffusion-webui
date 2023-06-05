"""
This is the core file in the `gradio` package, and defines the Interface class,
including various methods for constructing an interface and then launching it.
"""

from __future__ import annotations

import inspect
import json
import os
import warnings
import weakref
from typing import TYPE_CHECKING, Any, Callable

from gradio_client.documentation import document, set_documentation_group

from gradio import Examples, analytics, external, interpretation, utils
from gradio.blocks import Blocks
from gradio.components import (
    Button,
    Interpretation,
    IOComponent,
    Markdown,
    State,
    get_component_instance,
)
from gradio.data_classes import InterfaceTypes
from gradio.events import Changeable, Streamable, Submittable
from gradio.flagging import CSVLogger, FlaggingCallback, FlagMethod
from gradio.layouts import Column, Row, Tab, Tabs
from gradio.pipelines import load_from_pipeline
from gradio.themes import ThemeClass as Theme

set_documentation_group("interface")

if TYPE_CHECKING:  # Only import for type checking (is False at runtime).
    from transformers.pipelines.base import Pipeline


@document("launch", "load", "from_pipeline", "integrate", "queue")
class Interface(Blocks):
    """
    Interface is Gradio's main high-level class, and allows you to create a web-based GUI / demo
    around a machine learning model (or any Python function) in a few lines of code.
    You must specify three parameters: (1) the function to create a GUI for (2) the desired input components and
    (3) the desired output components. Additional parameters can be used to control the appearance
    and behavior of the demo.

    Example:
        import gradio as gr

        def image_classifier(inp):
            return {'cat': 0.3, 'dog': 0.7}

        demo = gr.Interface(fn=image_classifier, inputs="image", outputs="label")
        demo.launch()
    Demos: hello_world, hello_world_3, gpt_j
    Guides: quickstart, key-features, sharing-your-app, interface-state, reactive-interfaces, advanced-interface-features, setting-up-a-gradio-demo-for-maximum-performance
    """

    # stores references to all currently existing Interface instances
    instances: weakref.WeakSet = weakref.WeakSet()

    @classmethod
    def get_instances(cls) -> list[Interface]:
        """
        :return: list of all current instances.
        """
        return list(Interface.instances)

    @classmethod
    def load(
        cls,
        name: str,
        src: str | None = None,
        api_key: str | None = None,
        alias: str | None = None,
        **kwargs,
    ) -> Blocks:
        """
        Warning: this method will be deprecated. Use the equivalent `gradio.load()` instead. This is a class
        method that constructs a Blocks from a Hugging Face repo. Can accept
        model repos (if src is "models") or Space repos (if src is "spaces"). The input
        and output components are automatically loaded from the repo.
        Parameters:
            name: the name of the model (e.g. "gpt2" or "facebook/bart-base") or space (e.g. "flax-community/spanish-gpt2"), can include the `src` as prefix (e.g. "models/facebook/bart-base")
            src: the source of the model: `models` or `spaces` (or leave empty if source is provided as a prefix in `name`)
            api_key: optional access token for loading private Hugging Face Hub models or spaces. Find your token here: https://huggingface.co/settings/tokens. Warning: only provide this if you are loading a trusted private Space as it can be read by the Space you are loading.
            alias: optional string used as the name of the loaded model instead of the default name (only applies if loading a Space running Gradio 2.x)
        Returns:
            a Gradio Interface object for the given model
        """
        warnings.warn("gr.Interface.load() will be deprecated. Use gr.load() instead.")
        return external.load(
            name=name, src=src, hf_token=api_key, alias=alias, **kwargs
        )

    @classmethod
    def from_pipeline(cls, pipeline: Pipeline, **kwargs) -> Interface:
        """
        Class method that constructs an Interface from a Hugging Face transformers.Pipeline object.
        The input and output components are automatically determined from the pipeline.
        Parameters:
            pipeline: the pipeline object to use.
        Returns:
            a Gradio Interface object from the given Pipeline
        Example:
            import gradio as gr
            from transformers import pipeline
            pipe = pipeline("image-classification")
            gr.Interface.from_pipeline(pipe).launch()
        """
        interface_info = load_from_pipeline(pipeline)
        kwargs = dict(interface_info, **kwargs)
        interface = cls(**kwargs)
        return interface

    def __init__(
        self,
        fn: Callable,
        inputs: str | IOComponent | list[str | IOComponent] | None,
        outputs: str | IOComponent | list[str | IOComponent] | None,
        examples: list[Any] | list[list[Any]] | str | None = None,
        cache_examples: bool | None = None,
        examples_per_page: int = 10,
        live: bool = False,
        interpretation: Callable | str | None = None,
        num_shap: float = 2.0,
        title: str | None = None,
        description: str | None = None,
        article: str | None = None,
        thumbnail: str | None = None,
        theme: Theme | str | None = None,
        css: str | None = None,
        allow_flagging: str | None = None,
        flagging_options: list[str] | list[tuple[str, str]] | None = None,
        flagging_dir: str = "flagged",
        flagging_callback: FlaggingCallback = CSVLogger(),
        analytics_enabled: bool | None = None,
        batch: bool = False,
        max_batch_size: int = 4,
        _api_mode: bool = False,
        **kwargs,
    ):
        """
        Parameters:
            fn: the function to wrap an interface around. Often a machine learning model's prediction function. Each parameter of the function corresponds to one input component, and the function should return a single value or a tuple of values, with each element in the tuple corresponding to one output component.
            inputs: a single Gradio component, or list of Gradio components. Components can either be passed as instantiated objects, or referred to by their string shortcuts. The number of input components should match the number of parameters in fn. If set to None, then only the output components will be displayed.
            outputs: a single Gradio component, or list of Gradio components. Components can either be passed as instantiated objects, or referred to by their string shortcuts. The number of output components should match the number of values returned by fn. If set to None, then only the input components will be displayed.
            examples: sample inputs for the function; if provided, appear below the UI components and can be clicked to populate the interface. Should be nested list, in which the outer list consists of samples and each inner list consists of an input corresponding to each input component. A string path to a directory of examples can also be provided, but it should be within the directory with the python file running the gradio app. If there are multiple input components and a directory is provided, a log.csv file must be present in the directory to link corresponding inputs.
            cache_examples: If True, caches examples in the server for fast runtime in examples. The default option in HuggingFace Spaces is True. The default option elsewhere is False.
            examples_per_page: If examples are provided, how many to display per page.
            live: whether the interface should automatically rerun if any of the inputs change.
            interpretation: function that provides interpretation explaining prediction output. Pass "default" to use simple built-in interpreter, "shap" to use a built-in shapley-based interpreter, or your own custom interpretation function. For more information on the different interpretation methods, see the Advanced Interface Features guide.
            num_shap: a multiplier that determines how many examples are computed for shap-based interpretation. Increasing this value will increase shap runtime, but improve results. Only applies if interpretation is "shap".
            title: a title for the interface; if provided, appears above the input and output components in large font. Also used as the tab title when opened in a browser window.
            description: a description for the interface; if provided, appears above the input and output components and beneath the title in regular font. Accepts Markdown and HTML content.
            article: an expanded article explaining the interface; if provided, appears below the input and output components in regular font. Accepts Markdown and HTML content.
            thumbnail: path or url to image to use as display image when the web demo is shared on social media.
            theme: Theme to use, loaded from gradio.themes.
            css: custom css or path to custom css file to use with interface.
            allow_flagging: one of "never", "auto", or "manual". If "never" or "auto", users will not see a button to flag an input and output. If "manual", users will see a button to flag. If "auto", every input the user submits will be automatically flagged (outputs are not flagged). If "manual", both the input and outputs are flagged when the user clicks flag button. This parameter can be set with environmental variable GRADIO_ALLOW_FLAGGING; otherwise defaults to "manual".
            flagging_options: if provided, allows user to select from the list of options when flagging. Only applies if allow_flagging is "manual". Can either be a list of tuples of the form (label, value), where label is the string that will be displayed on the button and value is the string that will be stored in the flagging CSV; or it can be a list of strings ["X", "Y"], in which case the values will be the list of strings and the labels will ["Flag as X", "Flag as Y"], etc.
            flagging_dir: what to name the directory where flagged data is stored.
            flagging_callback: An instance of a subclass of FlaggingCallback which will be called when a sample is flagged. By default logs to a local CSV file.
            analytics_enabled: Whether to allow basic telemetry. If None, will use GRADIO_ANALYTICS_ENABLED environment variable if defined, or default to True.
            batch: If True, then the function should process a batch of inputs, meaning that it should accept a list of input values for each parameter. The lists should be of equal length (and be up to length `max_batch_size`). The function is then *required* to return a tuple of lists (even if there is only 1 output component), with each list in the tuple corresponding to one output component.
            max_batch_size: Maximum number of inputs to batch together if this is called from the queue (only relevant if batch=True)
        """
        super().__init__(
            analytics_enabled=analytics_enabled,
            mode="interface",
            css=css,
            title=title or "Gradio",
            theme=theme,
            **kwargs,
        )

        if isinstance(fn, list):
            raise DeprecationWarning(
                "The `fn` parameter only accepts a single function, support for a list "
                "of functions has been deprecated. Please use gradio.mix.Parallel "
                "instead."
            )

        self.interface_type = InterfaceTypes.STANDARD
        if (inputs is None or inputs == []) and (outputs is None or outputs == []):
            raise ValueError("Must provide at least one of `inputs` or `outputs`")
        elif outputs is None or outputs == []:
            outputs = []
            self.interface_type = InterfaceTypes.INPUT_ONLY
        elif inputs is None or inputs == []:
            inputs = []
            self.interface_type = InterfaceTypes.OUTPUT_ONLY

        assert isinstance(inputs, (str, list, IOComponent))
        assert isinstance(outputs, (str, list, IOComponent))

        if not isinstance(inputs, list):
            inputs = [inputs]
        if not isinstance(outputs, list):
            outputs = [outputs]

        if self.is_space and cache_examples is None:
            self.cache_examples = True
        else:
            self.cache_examples = cache_examples or False

        state_input_indexes = [
            idx for idx, i in enumerate(inputs) if i == "state" or isinstance(i, State)
        ]
        state_output_indexes = [
            idx for idx, o in enumerate(outputs) if o == "state" or isinstance(o, State)
        ]

        if len(state_input_indexes) == 0 and len(state_output_indexes) == 0:
            pass
        elif len(state_input_indexes) != 1 or len(state_output_indexes) != 1:
            raise ValueError(
                "If using 'state', there must be exactly one state input and one state output."
            )
        else:
            state_input_index = state_input_indexes[0]
            state_output_index = state_output_indexes[0]
            if inputs[state_input_index] == "state":
                default = utils.get_default_args(fn)[state_input_index]
                state_variable = State(value=default)  # type: ignore
            else:
                state_variable = inputs[state_input_index]

            inputs[state_input_index] = state_variable
            outputs[state_output_index] = state_variable

            if cache_examples:
                warnings.warn(
                    "Cache examples cannot be used with state inputs and outputs."
                    "Setting cache_examples to False."
                )
            self.cache_examples = False

        self.input_components = [
            get_component_instance(i, render=False) for i in inputs  # type: ignore
        ]
        self.output_components = [
            get_component_instance(o, render=False) for o in outputs  # type: ignore
        ]

        for component in self.input_components + self.output_components:
            if not (isinstance(component, IOComponent)):
                raise ValueError(
                    f"{component} is not a valid input/output component for Interface."
                )

        if len(self.input_components) == len(self.output_components):
            same_components = [
                i is o for i, o in zip(self.input_components, self.output_components)
            ]
            if all(same_components):
                self.interface_type = InterfaceTypes.UNIFIED

        if self.interface_type in [
            InterfaceTypes.STANDARD,
            InterfaceTypes.OUTPUT_ONLY,
        ]:
            for o in self.output_components:
                assert isinstance(o, IOComponent)
                if o.interactive is None:
                    # Unless explicitly otherwise specified, force output components to
                    # be non-interactive
                    o.interactive = False
        if (
            interpretation is None
            or isinstance(interpretation, list)
            or callable(interpretation)
        ):
            self.interpretation = interpretation
        elif isinstance(interpretation, str):
            self.interpretation = [
                interpretation.lower() for _ in self.input_components
            ]
        else:
            raise ValueError("Invalid value for parameter: interpretation")

        self.api_mode = _api_mode
        self.fn = fn
        self.fn_durations = [0, 0]
        self.__name__ = getattr(fn, "__name__", "fn")
        self.live = live
        self.title = title

        md = utils.get_markdown_parser()
        simple_description: str | None = None
        if description is not None:
            description = md.render(description)
            simple_description = utils.remove_html_tags(description)
        self.simple_description = simple_description
        self.description = description
        if article is not None:
            article = utils.readme_to_html(article)
            article = md.render(article)
        self.article = article

        self.thumbnail = thumbnail

        self.examples = examples
        self.num_shap = num_shap
        self.examples_per_page = examples_per_page

        self.simple_server = None

        # For allow_flagging: (1) first check for parameter,
        # (2) check for env variable, (3) default to True/"manual"
        if allow_flagging is None:
            allow_flagging = os.getenv("GRADIO_ALLOW_FLAGGING", "manual")
        if allow_flagging is True:
            warnings.warn(
                "The `allow_flagging` parameter in `Interface` now"
                "takes a string value ('auto', 'manual', or 'never')"
                ", not a boolean. Setting parameter to: 'manual'."
            )
            self.allow_flagging = "manual"
        elif allow_flagging == "manual":
            self.allow_flagging = "manual"
        elif allow_flagging is False:
            warnings.warn(
                "The `allow_flagging` parameter in `Interface` now"
                "takes a string value ('auto', 'manual', or 'never')"
                ", not a boolean. Setting parameter to: 'never'."
            )
            self.allow_flagging = "never"
        elif allow_flagging == "never":
            self.allow_flagging = "never"
        elif allow_flagging == "auto":
            self.allow_flagging = "auto"
        else:
            raise ValueError(
                "Invalid value for `allow_flagging` parameter."
                "Must be: 'auto', 'manual', or 'never'."
            )

        if flagging_options is None:
            self.flagging_options = [("Flag", "")]
        elif not (isinstance(flagging_options, list)):
            raise ValueError(
                "flagging_options must be a list of strings or list of (string, string) tuples."
            )
        elif all(isinstance(x, str) for x in flagging_options):
            self.flagging_options = [(f"Flag as {x}", x) for x in flagging_options]
        elif all(isinstance(x, tuple) for x in flagging_options):
            self.flagging_options = flagging_options
        else:
            raise ValueError(
                "flagging_options must be a list of strings or list of (string, string) tuples."
            )

        self.flagging_callback = flagging_callback
        self.flagging_dir = flagging_dir
        self.batch = batch
        self.max_batch_size = max_batch_size

        self.save_to = None  # Used for selenium tests
        self.share = None
        self.share_url = None
        self.local_url = None

        self.favicon_path = None
        analytics.version_check()
        Interface.instances.add(self)

        param_names = inspect.getfullargspec(self.fn)[0]
        if len(param_names) > 0 and inspect.ismethod(self.fn):
            param_names = param_names[1:]
        for component, param_name in zip(self.input_components, param_names):
            assert isinstance(component, IOComponent)
            if component.label is None:
                component.label = param_name
        for i, component in enumerate(self.output_components):
            assert isinstance(component, IOComponent)
            if component.label is None:
                if len(self.output_components) == 1:
                    component.label = "output"
                else:
                    component.label = f"output {i}"

        if self.allow_flagging != "never":
            if (
                self.interface_type == InterfaceTypes.UNIFIED
                or self.allow_flagging == "auto"
            ):
                self.flagging_callback.setup(self.input_components, self.flagging_dir)  # type: ignore
            elif self.interface_type == InterfaceTypes.INPUT_ONLY:
                pass
            else:
                self.flagging_callback.setup(
                    self.input_components + self.output_components, self.flagging_dir  # type: ignore
                )

        # Render the Gradio UI
        with self:
            self.render_title_description()

            submit_btn, clear_btn, stop_btn, flag_btns = None, None, None, None
            interpretation_btn, interpretation_set = None, None
            input_component_column, interpret_component_column = None, None

            with Row().style(equal_height=False):
                if self.interface_type in [
                    InterfaceTypes.STANDARD,
                    InterfaceTypes.INPUT_ONLY,
                    InterfaceTypes.UNIFIED,
                ]:
                    (
                        submit_btn,
                        clear_btn,
                        stop_btn,
                        flag_btns,
                        input_component_column,
                        interpret_component_column,
                        interpretation_set,
                    ) = self.render_input_column()
                if self.interface_type in [
                    InterfaceTypes.STANDARD,
                    InterfaceTypes.OUTPUT_ONLY,
                ]:
                    (
                        submit_btn_out,
                        clear_btn_2_out,
                        stop_btn_2_out,
                        flag_btns_out,
                        interpretation_btn,
                    ) = self.render_output_column(submit_btn)
                    submit_btn = submit_btn or submit_btn_out
                    clear_btn = clear_btn or clear_btn_2_out
                    stop_btn = stop_btn or stop_btn_2_out
                    flag_btns = flag_btns or flag_btns_out

            assert clear_btn is not None, "Clear button not rendered"

            self.attach_submit_events(submit_btn, stop_btn)
            self.attach_clear_events(
                clear_btn, input_component_column, interpret_component_column
            )
            self.attach_interpretation_events(
                interpretation_btn,
                interpretation_set,
                input_component_column,
                interpret_component_column,
            )

            self.attach_flagging_events(flag_btns, clear_btn)
            self.render_examples()
            self.render_article()

        self.config = self.get_config_file()

    def render_title_description(self) -> None:
        if self.title:
            Markdown(
                f"<h1 style='text-align: center; margin-bottom: 1rem'>{self.title}</h1>"
            )
        if self.description:
            Markdown(self.description)

    def render_flag_btns(self) -> list[Button]:
        return [Button(label) for label, _ in self.flagging_options]

    def render_input_column(
        self,
    ) -> tuple[
        Button | None,
        Button | None,
        Button | None,
        list[Button] | None,
        Column,
        Column | None,
        list[Interpretation] | None,
    ]:
        submit_btn, clear_btn, stop_btn, flag_btns = None, None, None, None
        interpret_component_column, interpretation_set = None, None

        with Column(variant="panel"):
            input_component_column = Column()
            with input_component_column:
                for component in self.input_components:
                    component.render()
            if self.interpretation:
                interpret_component_column = Column(visible=False)
                interpretation_set = []
                with interpret_component_column:
                    for component in self.input_components:
                        interpretation_set.append(Interpretation(component))
            with Row():
                if self.interface_type in [
                    InterfaceTypes.STANDARD,
                    InterfaceTypes.INPUT_ONLY,
                ]:
                    clear_btn = Button("Clear")
                    if not self.live:
                        submit_btn = Button("Submit", variant="primary")
                        # Stopping jobs only works if the queue is enabled
                        # We don't know if the queue is enabled when the interface
                        # is created. We use whether a generator function is provided
                        # as a proxy of whether the queue will be enabled.
                        # Using a generator function without the queue will raise an error.
                        if inspect.isgeneratorfunction(
                            self.fn
                        ) or inspect.isasyncgenfunction(self.fn):
                            stop_btn = Button("Stop", variant="stop", visible=False)
                elif self.interface_type == InterfaceTypes.UNIFIED:
                    clear_btn = Button("Clear")
                    submit_btn = Button("Submit", variant="primary")
                    if (
                        inspect.isgeneratorfunction(self.fn)
                        or inspect.isasyncgenfunction(self.fn)
                    ) and not self.live:
                        stop_btn = Button("Stop", variant="stop")
                    if self.allow_flagging == "manual":
                        flag_btns = self.render_flag_btns()
                    elif self.allow_flagging == "auto":
                        flag_btns = [submit_btn]
        return (
            submit_btn,
            clear_btn,
            stop_btn,
            flag_btns,
            input_component_column,
            interpret_component_column,
            interpretation_set,
        )

    def render_output_column(
        self,
        submit_btn_in: Button | None,
    ) -> tuple[Button | None, Button | None, Button | None, list | None, Button | None]:
        submit_btn = submit_btn_in
        interpretation_btn, clear_btn, flag_btns, stop_btn = None, None, None, None

        with Column(variant="panel"):
            for component in self.output_components:
                if not (isinstance(component, State)):
                    component.render()
            with Row():
                if self.interface_type == InterfaceTypes.OUTPUT_ONLY:
                    clear_btn = Button("Clear")
                    submit_btn = Button("Generate", variant="primary")
                    if (
                        inspect.isgeneratorfunction(self.fn)
                        or inspect.isasyncgenfunction(self.fn)
                    ) and not self.live:
                        # Stopping jobs only works if the queue is enabled
                        # We don't know if the queue is enabled when the interface
                        # is created. We use whether a generator function is provided
                        # as a proxy of whether the queue will be enabled.
                        # Using a generator function without the queue will raise an error.
                        stop_btn = Button("Stop", variant="stop", visible=False)
                if self.allow_flagging == "manual":
                    flag_btns = self.render_flag_btns()
                elif self.allow_flagging == "auto":
                    assert submit_btn is not None, "Submit button not rendered"
                    flag_btns = [submit_btn]
                if self.interpretation:
                    interpretation_btn = Button("Interpret")

        return submit_btn, clear_btn, stop_btn, flag_btns, interpretation_btn

    def render_article(self):
        if self.article:
            Markdown(self.article)

    def attach_submit_events(self, submit_btn: Button | None, stop_btn: Button | None):
        if self.live:
            if self.interface_type == InterfaceTypes.OUTPUT_ONLY:
                assert submit_btn is not None, "Submit button not rendered"
                super().load(self.fn, None, self.output_components)
                # For output-only interfaces, the user probably still want a "generate"
                # button even if the Interface is live
                submit_btn.click(
                    self.fn,
                    None,
                    self.output_components,
                    api_name="predict",
                    preprocess=not (self.api_mode),
                    postprocess=not (self.api_mode),
                    batch=self.batch,
                    max_batch_size=self.max_batch_size,
                )
            else:
                for component in self.input_components:
                    if isinstance(component, Streamable) and component.streaming:
                        component.stream(
                            self.fn,
                            self.input_components,
                            self.output_components,
                            api_name="predict",
                            preprocess=not (self.api_mode),
                            postprocess=not (self.api_mode),
                        )
                        continue
                    if isinstance(component, Changeable):
                        component.change(
                            self.fn,
                            self.input_components,
                            self.output_components,
                            api_name="predict",
                            preprocess=not (self.api_mode),
                            postprocess=not (self.api_mode),
                        )
        else:
            assert submit_btn is not None, "Submit button not rendered"
            fn = self.fn
            extra_output = []

            triggers = [submit_btn.click] + [
                component.submit
                for component in self.input_components
                if isinstance(component, Submittable)
            ]
            predict_events = []

            if stop_btn:
                # Wrap the original function to show/hide the "Stop" button
                async def fn(*args):
                    # The main idea here is to call the original function
                    # and append some updates to keep the "Submit" button
                    # hidden and the "Stop" button visible

                    if inspect.isasyncgenfunction(self.fn):
                        iterator = self.fn(*args)
                    else:
                        iterator = utils.SyncToAsyncIterator(
                            self.fn(*args), limiter=self.limiter
                        )
                    async for output in iterator:
                        yield output

                extra_output = [submit_btn, stop_btn]

                def cleanup():
                    return [Button.update(visible=True), Button.update(visible=False)]

                for i, trigger in enumerate(triggers):
                    predict_event = trigger(
                        lambda: (
                            submit_btn.update(visible=False),
                            stop_btn.update(visible=True),
                        ),
                        inputs=None,
                        outputs=[submit_btn, stop_btn],
                        queue=False,
                    ).then(
                        fn,
                        self.input_components,
                        self.output_components,
                        api_name="predict" if i == 0 else None,
                        scroll_to_output=True,
                        preprocess=not (self.api_mode),
                        postprocess=not (self.api_mode),
                        batch=self.batch,
                        max_batch_size=self.max_batch_size,
                    )
                    predict_events.append(predict_event)

                    predict_event.then(
                        cleanup,
                        inputs=None,
                        outputs=extra_output,  # type: ignore
                        queue=False,
                    )

                stop_btn.click(
                    cleanup,
                    inputs=None,
                    outputs=[submit_btn, stop_btn],
                    cancels=predict_events,
                    queue=False,
                )
            else:
                for i, trigger in enumerate(triggers):
                    predict_events.append(
                        trigger(
                            fn,
                            self.input_components,
                            self.output_components,
                            api_name="predict" if i == 0 else None,
                            scroll_to_output=True,
                            preprocess=not (self.api_mode),
                            postprocess=not (self.api_mode),
                            batch=self.batch,
                            max_batch_size=self.max_batch_size,
                        )
                    )

    def attach_clear_events(
        self,
        clear_btn: Button,
        input_component_column: Column | None,
        interpret_component_column: Column | None,
    ):
        clear_btn.click(
            None,
            [],
            (
                self.input_components
                + self.output_components
                + ([input_component_column] if input_component_column else [])
                + ([interpret_component_column] if self.interpretation else [])
            ),  # type: ignore
            _js=f"""() => {json.dumps(
                [getattr(component, "cleared_value", None)
                    for component in self.input_components + self.output_components] + (
                    [Column.update(visible=True)]
                    if self.interface_type
                        in [
                            InterfaceTypes.STANDARD,
                            InterfaceTypes.INPUT_ONLY,
                            InterfaceTypes.UNIFIED,
                        ]
                    else []
                )
                + ([Column.update(visible=False)] if self.interpretation else [])
            )}
            """,
        )

    def attach_interpretation_events(
        self,
        interpretation_btn: Button | None,
        interpretation_set: list[Interpretation] | None,
        input_component_column: Column | None,
        interpret_component_column: Column | None,
    ):
        if interpretation_btn:
            interpretation_btn.click(
                self.interpret_func,
                inputs=self.input_components + self.output_components,
                outputs=(interpretation_set or []) + [input_component_column, interpret_component_column],  # type: ignore
                preprocess=False,
            )

    def attach_flagging_events(self, flag_btns: list[Button] | None, clear_btn: Button):
        if not (
            flag_btns
            and self.interface_type
            in (
                InterfaceTypes.STANDARD,
                InterfaceTypes.OUTPUT_ONLY,
                InterfaceTypes.UNIFIED,
            )
        ):
            return

        if self.allow_flagging == "auto":
            flag_method = FlagMethod(
                self.flagging_callback, "", "", visual_feedback=False
            )
            flag_btns[0].click(  # flag_btns[0] is just the "Submit" button
                flag_method,
                inputs=self.input_components,
                outputs=None,
                preprocess=False,
                queue=False,
            )
            return

        if self.interface_type == InterfaceTypes.UNIFIED:
            flag_components = self.input_components
        else:
            flag_components = self.input_components + self.output_components

        for flag_btn, (label, value) in zip(flag_btns, self.flagging_options):
            assert isinstance(value, str)
            flag_method = FlagMethod(self.flagging_callback, label, value)
            flag_btn.click(
                lambda: Button.update(value="Saving...", interactive=False),
                None,
                flag_btn,
                queue=False,
            )
            flag_btn.click(
                flag_method,
                inputs=flag_components,
                outputs=flag_btn,
                preprocess=False,
                queue=False,
            )
            clear_btn.click(
                flag_method.reset,
                None,
                flag_btn,
                queue=False,
            )

    def render_examples(self):
        if self.examples:
            non_state_inputs = [
                c for c in self.input_components if not isinstance(c, State)
            ]
            non_state_outputs = [
                c for c in self.output_components if not isinstance(c, State)
            ]
            self.examples_handler = Examples(
                examples=self.examples,
                inputs=non_state_inputs,  # type: ignore
                outputs=non_state_outputs,  # type: ignore
                fn=self.fn,
                cache_examples=self.cache_examples,
                examples_per_page=self.examples_per_page,
                _api_mode=self.api_mode,
                batch=self.batch,
            )

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        repr = f"Gradio Interface for: {self.__name__}"
        repr += f"\n{'-' * len(repr)}"
        repr += "\ninputs:"
        for component in self.input_components:
            repr += f"\n|-{component}"
        repr += "\noutputs:"
        for component in self.output_components:
            repr += f"\n|-{component}"
        return repr

    async def interpret_func(self, *args):
        return await self.interpret(list(args)) + [
            Column.update(visible=False),
            Column.update(visible=True),
        ]

    async def interpret(self, raw_input: list[Any]) -> list[Any]:
        return [
            {"original": raw_value, "interpretation": interpretation}
            for interpretation, raw_value in zip(
                (await interpretation.run_interpret(self, raw_input))[0], raw_input
            )
        ]

    def test_launch(self) -> None:
        """
        Deprecated.
        """
        warnings.warn("The Interface.test_launch() function is deprecated.")


@document()
class TabbedInterface(Blocks):
    """
    A TabbedInterface is created by providing a list of Interfaces, each of which gets
    rendered in a separate tab.
    Demos: stt_or_tts
    """

    def __init__(
        self,
        interface_list: list[Interface],
        tab_names: list[str] | None = None,
        title: str | None = None,
        theme: Theme | None = None,
        analytics_enabled: bool | None = None,
        css: str | None = None,
    ):
        """
        Parameters:
            interface_list: a list of interfaces to be rendered in tabs.
            tab_names: a list of tab names. If None, the tab names will be "Tab 1", "Tab 2", etc.
            title: a title for the interface; if provided, appears above the input and output components in large font. Also used as the tab title when opened in a browser window.
            analytics_enabled: whether to allow basic telemetry. If None, will use GRADIO_ANALYTICS_ENABLED environment variable or default to True.
            css: custom css or path to custom css file to apply to entire Blocks
        Returns:
            a Gradio Tabbed Interface for the given interfaces
        """
        super().__init__(
            title=title or "Gradio",
            theme=theme,
            analytics_enabled=analytics_enabled,
            mode="tabbed_interface",
            css=css,
        )
        if tab_names is None:
            tab_names = [f"Tab {i}" for i in range(len(interface_list))]
        with self:
            if title:
                Markdown(
                    f"<h1 style='text-align: center; margin-bottom: 1rem'>{title}</h1>"
                )
            with Tabs():
                for interface, tab_name in zip(interface_list, tab_names):
                    with Tab(label=tab_name):
                        interface.render()


def close_all(verbose: bool = True) -> None:
    for io in Interface.get_instances():
        io.close(verbose)
