"""Contains all of the events that can be triggered in a gr.Blocks() app, with the exception
of the on-page-load event, which is defined in gr.Blocks().load()."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Callable

from gradio_client.documentation import document, set_documentation_group

from gradio.blocks import Block
from gradio.helpers import EventData
from gradio.utils import get_cancel_function

if TYPE_CHECKING:  # Only import for type checking (is False at runtime).
    from gradio.components import Component, StatusTracker

set_documentation_group("events")


def set_cancel_events(
    block: Block, event_name: str, cancels: None | dict[str, Any] | list[dict[str, Any]]
):
    if cancels:
        if not isinstance(cancels, list):
            cancels = [cancels]
        cancel_fn, fn_indices_to_cancel = get_cancel_function(cancels)
        block.set_event_trigger(
            event_name,
            cancel_fn,
            inputs=None,
            outputs=None,
            queue=False,
            preprocess=False,
            cancels=fn_indices_to_cancel,
        )


class EventListener(Block):
    def __init__(self: Any):
        for event_listener_class in EventListener.__subclasses__():
            if isinstance(self, event_listener_class):
                event_listener_class.__init__(self)


class Dependency(dict):
    def __init__(self, trigger, key_vals, dep_index):
        super().__init__(key_vals)
        self.trigger = trigger
        self.then = EventListenerMethod(
            self.trigger,
            "then",
            trigger_after=dep_index,
            trigger_only_on_success=False,
        )
        """
        Triggered after directly preceding event is completed, regardless of success or failure.
        """
        self.success = EventListenerMethod(
            self.trigger,
            "success",
            trigger_after=dep_index,
            trigger_only_on_success=True,
        )
        """
        Triggered after directly preceding event is completed, if it was successful.
        """


class EventListenerMethod:
    """
    Triggered on an event deployment.
    """

    def __init__(
        self,
        trigger: Block,
        event_name: str,
        show_progress: bool = True,
        callback: Callable | None = None,
        trigger_after: int | None = None,
        trigger_only_on_success: bool = False,
    ):
        self.trigger = trigger
        self.event_name = event_name
        self.show_progress = show_progress
        self.callback = callback
        self.trigger_after = trigger_after
        self.trigger_only_on_success = trigger_only_on_success

    def __call__(
        self,
        fn: Callable | None,
        inputs: Component | list[Component] | set[Component] | None = None,
        outputs: Component | list[Component] | None = None,
        api_name: str | None = None,
        status_tracker: StatusTracker | None = None,
        scroll_to_output: bool = False,
        show_progress: bool | None = None,
        queue: bool | None = None,
        batch: bool = False,
        max_batch_size: int = 4,
        preprocess: bool = True,
        postprocess: bool = True,
        cancels: dict[str, Any] | list[dict[str, Any]] | None = None,
        every: float | None = None,
        _js: str | None = None,
    ) -> Dependency:
        """
        Parameters:
            fn: the function to wrap an interface around. Often a machine learning model's prediction function. Each parameter of the function corresponds to one input component, and the function should return a single value or a tuple of values, with each element in the tuple corresponding to one output component.
            inputs: List of gradio.components to use as inputs. If the function takes no inputs, this should be an empty list.
            outputs: List of gradio.components to use as outputs. If the function returns no outputs, this should be an empty list.
            api_name: Defining this parameter exposes the endpoint in the api docs
            scroll_to_output: If True, will scroll to output component on completion
            show_progress: If True, will show progress animation while pending
            queue: If True, will place the request on the queue, if the queue has been enabled. If False, will not put this event on the queue, even if the queue has been enabled. If None, will use the queue setting of the gradio app.
            batch: If True, then the function should process a batch of inputs, meaning that it should accept a list of input values for each parameter. The lists should be of equal length (and be up to length `max_batch_size`). The function is then *required* to return a tuple of lists (even if there is only 1 output component), with each list in the tuple corresponding to one output component.
            max_batch_size: Maximum number of inputs to batch together if this is called from the queue (only relevant if batch=True)
            preprocess: If False, will not run preprocessing of component data before running 'fn' (e.g. leaving it as a base64 string if this method is called with the `Image` component).
            postprocess: If False, will not run postprocessing of component data before returning 'fn' output to the browser.
            cancels: A list of other events to cancel when This listener is triggered. For example, setting cancels=[click_event] will cancel the click_event, where click_event is the return value of another components .click method. Functions that have not yet run (or generators that are iterating) will be cancelled, but functions that are currently running will be allowed to finish.
            every: Run this event 'every' number of seconds while the client connection is open. Interpreted in seconds. Queue must be enabled.
        """
        if status_tracker:
            warnings.warn(
                "The 'status_tracker' parameter has been deprecated and has no effect."
            )
        if isinstance(self, Streamable):
            self.check_streamable()

        dep, dep_index = self.trigger.set_event_trigger(
            self.event_name,
            fn,
            inputs,
            outputs,
            preprocess=preprocess,
            postprocess=postprocess,
            scroll_to_output=scroll_to_output,
            show_progress=show_progress
            if show_progress is not None
            else self.show_progress,
            api_name=api_name,
            js=_js,
            queue=queue,
            batch=batch,
            max_batch_size=max_batch_size,
            every=every,
            trigger_after=self.trigger_after,
            trigger_only_on_success=self.trigger_only_on_success,
        )
        set_cancel_events(self.trigger, self.event_name, cancels)
        if self.callback:
            self.callback()
        return Dependency(self.trigger, dep, dep_index)


@document("*change", inherit=True)
class Changeable(EventListener):
    def __init__(self):
        self.change = EventListenerMethod(self, "change")
        """
        This listener is triggered when the component's value changes either because of user input (e.g. a user types in a textbox) OR because of a function update (e.g. an image receives a value from the output of an event trigger).
        See `.input()` for a listener that is only triggered by user input.
        This method can be used when this component is in a Gradio Blocks.
        """


@document("*input", inherit=True)
class Inputable(EventListener):
    def __init__(self):
        self.input = EventListenerMethod(self, "input")
        """
        This listener is triggered when the user changes the value of the component.
        This method can be used when this component is in a Gradio Blocks.
        """


@document("*click", inherit=True)
class Clickable(EventListener):
    def __init__(self):
        self.click = EventListenerMethod(self, "click")
        """
        This listener is triggered when the component (e.g. a button) is clicked.
        This method can be used when this component is in a Gradio Blocks.
        """


@document("*submit", inherit=True)
class Submittable(EventListener):
    def __init__(self):
        self.submit = EventListenerMethod(self, "submit")
        """
        This listener is triggered when the user presses the Enter key while the component (e.g. a textbox) is focused.
        This method can be used when this component is in a Gradio Blocks.
        """


@document("*edit", inherit=True)
class Editable(EventListener):
    def __init__(self):
        self.edit = EventListenerMethod(self, "edit")
        """
        This listener is triggered when the user edits the component (e.g. image) using the
        built-in editor. This method can be used when this component is in a Gradio Blocks.
        """


@document("*clear", inherit=True)
class Clearable(EventListener):
    def __init__(self):
        self.clear = EventListenerMethod(self, "clear")
        """
        This listener is triggered when the user clears the component (e.g. image or audio)
        using the X button for the component. This method can be used when this component is in a Gradio Blocks.
        """


@document("*play", "*pause", "*stop", inherit=True)
class Playable(EventListener):
    def __init__(self):
        self.play = EventListenerMethod(self, "play")
        """
        This listener is triggered when the user plays the component (e.g. audio or video).
        This method can be used when this component is in a Gradio Blocks.
        """

        self.pause = EventListenerMethod(self, "pause")
        """
        This listener is triggered when the user pauses the component (e.g. audio or video).
        This method can be used when this component is in a Gradio Blocks.
        """

        self.stop = EventListenerMethod(self, "stop")
        """
        This listener is triggered when the user stops the component (e.g. audio or video).
        This method can be used when this component is in a Gradio Blocks.
        """


@document("*stream", inherit=True)
class Streamable(EventListener):
    def __init__(self):
        self.streaming: bool
        self.stream = EventListenerMethod(
            self,
            "stream",
            show_progress=False,
            callback=lambda: setattr(self, "streaming", True),
        )
        """
        This listener is triggered when the user streams the component (e.g. a live webcam
        component). This method can be used when this component is in a Gradio Blocks.
        """

    def check_streamable(self):
        pass


@document("*blur", inherit=True)
class Blurrable(EventListener):
    def __init__(self):
        self.blur = EventListenerMethod(self, "blur")
        """
        This listener is triggered when the component's is unfocused/blurred (e.g. when the user clicks outside of a textbox). 
        This method can be used when this component is in a Gradio Blocks.
        """


@document("*upload", inherit=True)
class Uploadable(EventListener):
    def __init__(self):
        self.upload = EventListenerMethod(self, "upload")
        """
        This listener is triggered when the user uploads a file into the component (e.g. when the user uploads a video into a video component).
        This method can be used when this component is in a Gradio Blocks.
        """


@document("*release", inherit=True)
class Releaseable(EventListener):
    def __init__(self):
        self.release = EventListenerMethod(self, "release")
        """
        This listener is triggered when the user releases the mouse on this component (e.g. when the user releases the slider).
        This method can be used when this component is in a Gradio Blocks.
        """


@document("*select", inherit=True)
class Selectable(EventListener):
    def __init__(self):
        self.selectable: bool = False
        self.select = EventListenerMethod(
            self, "select", callback=lambda: setattr(self, "selectable", True)
        )
        """
        This listener is triggered when the user selects from within the Component.
        This event has EventData of type gradio.SelectData that carries information, accessible through SelectData.index and SelectData.value.
        See EventData documentation on how to use this event data.
        """


class SelectData(EventData):
    def __init__(self, target: Block | None, data: Any):
        super().__init__(target, data)
        self.index: int | tuple[int, int] = data["index"]
        """
        The index of the selected item. Is a tuple if the component is two dimensional or selection is a range.
        """
        self.value: Any = data["value"]
        """
        The value of the selected item.
        """
        self.selected: bool = data.get("selected", True)
        """
        True if the item was selected, False if deselected.
        """
