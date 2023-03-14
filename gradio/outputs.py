# type: ignore
"""
This module defines various classes that can serve as the `output` to an interface. Each class must inherit from
`OutputComponent`, and each class must define a path to its template. All of the subclasses of `OutputComponent` are
automatically added to a registry, which allows them to be easily referenced in other parts of the code.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

from gradio import components


class Textbox(components.Textbox):
    def __init__(
        self,
        type: str = "text",
        label: Optional[str] = None,
    ):
        warnings.warn(
            "Usage of gradio.outputs is deprecated, and will not be supported in the future, please import your components from gradio.components",
        )
        super().__init__(label=label, type=type)


class Image(components.Image):
    """
    Component displays an output image.
    Output type: Union[numpy.array, PIL.Image, str, matplotlib.pyplot, Tuple[Union[numpy.array, PIL.Image, str], List[Tuple[str, float, float, float, float]]]]
    """

    def __init__(
        self, type: str = "auto", plot: bool = False, label: Optional[str] = None
    ):
        """
        Parameters:
        type (str): Type of value to be passed to component. "numpy" expects a numpy array with shape (width, height, 3), "pil" expects a PIL image object, "file" expects a file path to the saved image or a remote URL, "plot" expects a matplotlib.pyplot object, "auto" detects return type.
        plot (bool): DEPRECATED. Whether to expect a plot to be returned by the function.
        label (str): component name in interface.
        """
        warnings.warn(
            "Usage of gradio.outputs is deprecated, and will not be supported in the future, please import your components from gradio.components",
        )
        if plot:
            type = "plot"
        super().__init__(type=type, label=label)


class Video(components.Video):
    """
    Used for video output.
    Output type: filepath
    """

    def __init__(self, type: Optional[str] = None, label: Optional[str] = None):
        """
        Parameters:
        type (str): Type of video format to be passed to component, such as 'avi' or 'mp4'. Use 'mp4' to ensure browser playability. If set to None, video will keep returned format.
        label (str): component name in interface.
        """
        warnings.warn(
            "Usage of gradio.outputs is deprecated, and will not be supported in the future, please import your components from gradio.components",
        )
        super().__init__(format=type, label=label)


class Audio(components.Audio):
    """
    Creates an audio player that plays the output audio.
    Output type: Union[Tuple[int, numpy.array], str]
    """

    def __init__(self, type: str = "auto", label: Optional[str] = None):
        """
        Parameters:
        type (str): Type of value to be passed to component. "numpy" returns a 2-set tuple with an integer sample_rate and the data as 16-bit int numpy.array of shape (samples, 2), "file" returns a temporary file path to the saved wav audio file, "auto" detects return type.
        label (str): component name in interface.
        """
        warnings.warn(
            "Usage of gradio.outputs is deprecated, and will not be supported in the future, please import your components from gradio.components",
        )
        super().__init__(type=type, label=label)


class File(components.File):
    """
    Used for file output.
    Output type: Union[file-like, str]
    """

    def __init__(self, label: Optional[str] = None):
        """
        Parameters:
        label (str): component name in interface.
        """
        warnings.warn(
            "Usage of gradio.outputs is deprecated, and will not be supported in the future, please import your components from gradio.components",
        )
        super().__init__(label=label)


class Dataframe(components.Dataframe):
    """
    Component displays 2D output through a spreadsheet interface.
    Output type: Union[pandas.DataFrame, numpy.array, List[Union[str, float]], List[List[Union[str, float]]]]
    """

    def __init__(
        self,
        headers: Optional[List[str]] = None,
        max_rows: Optional[int] = 20,
        max_cols: Optional[int] = None,
        overflow_row_behaviour: str = "paginate",
        type: str = "auto",
        label: Optional[str] = None,
    ):
        """
        Parameters:
        headers (List[str]): Header names to dataframe. Only applicable if type is "numpy" or "array".
        max_rows (int): Maximum number of rows to display at once. Set to None for infinite.
        max_cols (int): Maximum number of columns to display at once. Set to None for infinite.
        overflow_row_behaviour (str): If set to "paginate", will create pages for overflow rows. If set to "show_ends", will show initial and final rows and truncate middle rows.
        type (str): Type of value to be passed to component. "pandas" for pandas dataframe, "numpy" for numpy array, or "array" for Python array, "auto" detects return type.
        label (str): component name in interface.
        """
        warnings.warn(
            "Usage of gradio.outputs is deprecated, and will not be supported in the future, please import your components from gradio.components",
        )
        super().__init__(
            headers=headers,
            type=type,
            label=label,
            max_rows=max_rows,
            max_cols=max_cols,
            overflow_row_behaviour=overflow_row_behaviour,
        )


class Timeseries(components.Timeseries):
    """
    Component accepts pandas.DataFrame.
    Output type: pandas.DataFrame
    """

    def __init__(
        self, x: str = None, y: str | List[str] = None, label: Optional[str] = None
    ):
        """
        Parameters:
        x (str): Column name of x (time) series. None if csv has no headers, in which case first column is x series.
        y (Union[str, List[str]]): Column name of y series, or list of column names if multiple series. None if csv has no headers, in which case every column after first is a y series.
        label (str): component name in interface.
        """
        warnings.warn(
            "Usage of gradio.outputs is deprecated, and will not be supported in the future, please import your components from gradio.components",
        )
        super().__init__(x=x, y=y, label=label)


class State(components.State):
    """
    Special hidden component that stores state across runs of the interface.
    Output type: Any
    """

    def __init__(self, label: Optional[str] = None):
        """
        Parameters:
        label (str): component name in interface (not used).
        """
        warnings.warn(
            "Usage of gradio.outputs is deprecated, and will not be supported in the future, please import this component as gr.State() from gradio.components",
        )
        super().__init__(label=label)


class Label(components.Label):
    """
    Component outputs a classification label, along with confidence scores of top categories if provided. Confidence scores are represented as a dictionary mapping labels to scores between 0 and 1.
    Output type: Union[Dict[str, float], str, int, float]
    """

    def __init__(
        self,
        num_top_classes: Optional[int] = None,
        type: str = "auto",
        label: Optional[str] = None,
    ):
        """
        Parameters:
        num_top_classes (int): number of most confident classes to show.
        type (str): Type of value to be passed to component. "value" expects a single out label, "confidences" expects a dictionary mapping labels to confidence scores, "auto" detects return type.
        label (str): component name in interface.
        """
        warnings.warn(
            "Usage of gradio.outputs is deprecated, and will not be supported in the future, please import your components from gradio.components",
        )
        super().__init__(num_top_classes=num_top_classes, type=type, label=label)


class KeyValues:
    """
    Component displays a table representing values for multiple fields.
    Output type: Union[Dict, List[Tuple[str, Union[str, int, float]]]]
    """

    def __init__(self, value: str = " ", *, label: Optional[str] = None, **kwargs):
        """
        Parameters:
        value (str): IGNORED
        label (str): component name in interface.
        """
        raise DeprecationWarning(
            "The KeyValues component is deprecated. Please use the DataFrame or JSON "
            "components instead."
        )


class HighlightedText(components.HighlightedText):
    """
    Component creates text that contains spans that are highlighted by category or numerical value.
    Output is represent as a list of Tuple pairs, where the first element represents the span of text represented by the tuple, and the second element represents the category or value of the text.
    Output type: List[Tuple[str, Union[float, str]]]
    """

    def __init__(
        self,
        color_map: Dict[str, str] = None,
        label: Optional[str] = None,
        show_legend: bool = False,
    ):
        """
        Parameters:
        color_map (Dict[str, str]): Map between category and respective colors
        label (str): component name in interface.
        show_legend (bool): whether to show span categories in a separate legend or inline.
        """
        warnings.warn(
            "Usage of gradio.outputs is deprecated, and will not be supported in the future, please import your components from gradio.components",
        )
        super().__init__(color_map=color_map, label=label, show_legend=show_legend)


class JSON(components.JSON):
    """
    Used for JSON output. Expects a JSON string or a Python object that is JSON serializable.
    Output type: Union[str, Any]
    """

    def __init__(self, label: Optional[str] = None):
        """
        Parameters:
        label (str): component name in interface.
        """
        warnings.warn(
            "Usage of gradio.outputs is deprecated, and will not be supported in the future, please import your components from gradio.components",
        )
        super().__init__(label=label)


class HTML(components.HTML):
    """
    Used for HTML output. Expects an HTML valid string.
    Output type: str
    """

    def __init__(self, label: Optional[str] = None):
        """
        Parameters:
        label (str): component name in interface.
        """
        super().__init__(label=label)


class Carousel(components.Carousel):
    """
    Component displays a set of output components that can be scrolled through.
    """

    def __init__(
        self,
        components: components.Component | List[components.Component],
        label: Optional[str] = None,
    ):
        """
        Parameters:
        components (Union[List[Component], Component]): Classes of component(s) that will be scrolled through.
        label (str): component name in interface.
        """
        warnings.warn(
            "Usage of gradio.outputs is deprecated, and will not be supported in the future, please import your components from gradio.components",
        )
        super().__init__(components=components, label=label)


class Chatbot(components.Chatbot):
    """
    Component displays a chatbot output showing both user submitted messages and responses
    Output type: List[Tuple[str, str]]
    """

    def __init__(self, label: Optional[str] = None):
        """
        Parameters:
        label (str): component name in interface (not used).
        """
        warnings.warn(
            "Usage of gradio.outputs is deprecated, and will not be supported in the future, please import your components from gradio.components",
        )
        super().__init__(label=label)


class Image3D(components.Model3D):
    """
    Used for 3D image model output.
    Input type: File object of type (.obj, glb, or .gltf)
    """

    def __init__(
        self,
        clear_color=None,
        label: Optional[str] = None,
    ):
        """
        Parameters:
        label (str): component name in interface.
        optional (bool): If True, the interface can be submitted with no uploaded image, in which case the input value is None.
        """
        warnings.warn(
            "Usage of gradio.outputs is deprecated, and will not be supported in the future, please import your components from gradio.components",
        )
        super().__init__(clear_color=clear_color, label=label)
