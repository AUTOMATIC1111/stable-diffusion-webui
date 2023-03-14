# type: ignore
"""
This module defines various classes that can serve as the `input` to an interface. Each class must inherit from
`InputComponent`, and each class must define a path to its template. All of the subclasses of `InputComponent` are
automatically added to a registry, which allows them to be easily referenced in other parts of the code.
"""

from __future__ import annotations

import warnings
from typing import Any, List, Optional, Tuple

from gradio import components


class Textbox(components.Textbox):
    def __init__(
        self,
        lines: int = 1,
        placeholder: Optional[str] = None,
        default: str = "",
        numeric: Optional[bool] = False,
        type: Optional[str] = "text",
        label: Optional[str] = None,
        optional: bool = False,
    ):
        warnings.warn(
            "Usage of gradio.inputs is deprecated, and will not be supported in the future, please import your component from gradio.components",
        )
        super().__init__(
            value=default,
            lines=lines,
            placeholder=placeholder,
            label=label,
            numeric=numeric,
            type=type,
            optional=optional,
        )


class Number(components.Number):
    """
    Component creates a field for user to enter numeric input. Provides a number as an argument to the wrapped function.
    Input type: float
    """

    def __init__(
        self,
        default: Optional[float] = None,
        label: Optional[str] = None,
        optional: bool = False,
    ):
        """
        Parameters:
        default (float): default value.
        label (str): component name in interface.
        optional (bool): If True, the interface can be submitted with no value for this component.
        """
        warnings.warn(
            "Usage of gradio.inputs is deprecated, and will not be supported in the future, please import your component from gradio.components",
        )
        super().__init__(value=default, label=label, optional=optional)


class Slider(components.Slider):
    """
    Component creates a slider that ranges from `minimum` to `maximum`. Provides number as an argument to the wrapped function.
    Input type: float
    """

    def __init__(
        self,
        minimum: float = 0,
        maximum: float = 100,
        step: Optional[float] = None,
        default: Optional[float] = None,
        label: Optional[str] = None,
        optional: bool = False,
    ):
        """
        Parameters:
        minimum (float): minimum value for slider.
        maximum (float): maximum value for slider.
        step (float): increment between slider values.
        default (float): default value.
        label (str): component name in interface.
        optional (bool): this parameter is ignored.
        """
        warnings.warn(
            "Usage of gradio.inputs is deprecated, and will not be supported in the future, please import your component from gradio.components",
        )

        super().__init__(
            value=default,
            minimum=minimum,
            maximum=maximum,
            step=step,
            label=label,
            optional=optional,
        )


class Checkbox(components.Checkbox):
    """
    Component creates a checkbox that can be set to `True` or `False`. Provides a boolean as an argument to the wrapped function.
    Input type: bool
    """

    def __init__(
        self,
        default: bool = False,
        label: Optional[str] = None,
        optional: bool = False,
    ):
        """
        Parameters:
        label (str): component name in interface.
        default (bool): if True, checked by default.
        optional (bool): this parameter is ignored.
        """
        warnings.warn(
            "Usage of gradio.inputs is deprecated, and will not be supported in the future, please import your component from gradio.components",
        )
        super().__init__(value=default, label=label, optional=optional)


class CheckboxGroup(components.CheckboxGroup):
    """
    Component creates a set of checkboxes of which a subset can be selected. Provides a list of strings representing the selected choices as an argument to the wrapped function.
    Input type: Union[List[str], List[int]]
    """

    def __init__(
        self,
        choices: List[str],
        default: List[str] = [],
        type: str = "value",
        label: Optional[str] = None,
        optional: bool = False,
    ):
        """
        Parameters:
        choices (List[str]): list of options to select from.
        default (List[str]): default selected list of options.
        type (str): Type of value to be returned by component. "value" returns the list of strings of the choices selected, "index" returns the list of indicies of the choices selected.
        label (str): component name in interface.
        optional (bool): this parameter is ignored.
        """
        warnings.warn(
            "Usage of gradio.inputs is deprecated, and will not be supported in the future, please import your component from gradio.components",
        )
        super().__init__(
            value=default,
            choices=choices,
            type=type,
            label=label,
            optional=optional,
        )


class Radio(components.Radio):
    """
    Component creates a set of radio buttons of which only one can be selected. Provides string representing selected choice as an argument to the wrapped function.
    Input type: Union[str, int]
    """

    def __init__(
        self,
        choices: List[str],
        type: str = "value",
        default: Optional[str] = None,
        label: Optional[str] = None,
        optional: bool = False,
    ):
        """
        Parameters:
        choices (List[str]): list of options to select from.
        type (str): Type of value to be returned by component. "value" returns the string of the choice selected, "index" returns the index of the choice selected.
        default (str): the button selected by default. If None, no button is selected by default.
        label (str): component name in interface.
        optional (bool): this parameter is ignored.
        """
        warnings.warn(
            "Usage of gradio.inputs is deprecated, and will not be supported in the future, please import your component from gradio.components",
        )
        super().__init__(
            choices=choices,
            type=type,
            value=default,
            label=label,
            optional=optional,
        )


class Dropdown(components.Dropdown):
    """
    Component creates a dropdown of which only one can be selected. Provides string representing selected choice as an argument to the wrapped function.
    Input type: Union[str, int]
    """

    def __init__(
        self,
        choices: List[str],
        type: str = "value",
        default: Optional[str] = None,
        label: Optional[str] = None,
        optional: bool = False,
    ):
        """
        Parameters:
        choices (List[str]): list of options to select from.
        type (str): Type of value to be returned by component. "value" returns the string of the choice selected, "index" returns the index of the choice selected.
        default (str): default value selected in dropdown. If None, no value is selected by default.
        label (str): component name in interface.
        optional (bool): this parameter is ignored.
        """
        warnings.warn(
            "Usage of gradio.inputs is deprecated, and will not be supported in the future, please import your component from gradio.components",
        )
        super().__init__(
            choices=choices,
            type=type,
            value=default,
            label=label,
            optional=optional,
        )


class Image(components.Image):
    """
    Component creates an image upload box with editing capabilities.
    Input type: Union[numpy.array, PIL.Image, file-object]
    """

    def __init__(
        self,
        shape: Tuple[int, int] = None,
        image_mode: str = "RGB",
        invert_colors: bool = False,
        source: str = "upload",
        tool: str = "editor",
        type: str = "numpy",
        label: str = None,
        optional: bool = False,
    ):
        """
        Parameters:
        shape (Tuple[int, int]): (width, height) shape to crop and resize image to; if None, matches input image size.
        image_mode (str): How to process the uploaded image. Accepts any of the PIL image modes, e.g. "RGB" for color images, "RGBA" to include the transparency mask, "L" for black-and-white images.
        invert_colors (bool): whether to invert the image as a preprocessing step.
        source (str): Source of image. "upload" creates a box where user can drop an image file, "webcam" allows user to take snapshot from their webcam, "canvas" defaults to a white image that can be edited and drawn upon with tools.
        tool (str): Tools used for editing. "editor" allows a full screen editor, "select" provides a cropping and zoom tool.
        type (str): Type of value to be returned by component. "numpy" returns a numpy array with shape (width, height, 3) and values from 0 to 255, "pil" returns a PIL image object, "file" returns a temporary file object whose path can be retrieved by file_obj.name, "filepath" returns the path directly.
        label (str): component name in interface.
        optional (bool): If True, the interface can be submitted with no uploaded image, in which case the input value is None.
        """
        warnings.warn(
            "Usage of gradio.inputs is deprecated, and will not be supported in the future, please import your component from gradio.components",
        )
        super().__init__(
            shape=shape,
            image_mode=image_mode,
            invert_colors=invert_colors,
            source=source,
            tool=tool,
            type=type,
            label=label,
            optional=optional,
        )


class Video(components.Video):
    """
    Component creates a video file upload that is converted to a file path.

    Input type: filepath
    """

    def __init__(
        self,
        type: Optional[str] = None,
        source: str = "upload",
        label: Optional[str] = None,
        optional: bool = False,
    ):
        """
        Parameters:
        type (str): Type of video format to be returned by component, such as 'avi' or 'mp4'. If set to None, video will keep uploaded format.
        source (str): Source of video. "upload" creates a box where user can drop an video file, "webcam" allows user to record a video from their webcam.
        label (str): component name in interface.
        optional (bool): If True, the interface can be submitted with no uploaded video, in which case the input value is None.
        """
        warnings.warn(
            "Usage of gradio.inputs is deprecated, and will not be supported in the future, please import your components from gradio.components",
        )
        super().__init__(format=type, source=source, label=label, optional=optional)


class Audio(components.Audio):
    """
    Component accepts audio input files.
    Input type: Union[Tuple[int, numpy.array], file-object, numpy.array]
    """

    def __init__(
        self,
        source: str = "upload",
        type: str = "numpy",
        label: str = None,
        optional: bool = False,
    ):
        """
        Parameters:
        source (str): Source of audio. "upload" creates a box where user can drop an audio file, "microphone" creates a microphone input.
        type (str): Type of value to be returned by component. "numpy" returns a 2-set tuple with an integer sample_rate and the data numpy.array of shape (samples, 2), "file" returns a temporary file object whose path can be retrieved by file_obj.name, "filepath" returns the path directly.
        label (str): component name in interface.
        optional (bool): If True, the interface can be submitted with no uploaded audio, in which case the input value is None.
        """
        warnings.warn(
            "Usage of gradio.inputs is deprecated, and will not be supported in the future, please import your components from gradio.components",
        )
        super().__init__(source=source, type=type, label=label, optional=optional)


class File(components.File):
    """
    Component accepts generic file uploads.
    Input type: Union[file-object, bytes, List[Union[file-object, bytes]]]
    """

    def __init__(
        self,
        file_count: str = "single",
        type: str = "file",
        label: Optional[str] = None,
        keep_filename: bool = True,
        optional: bool = False,
    ):
        """
        Parameters:
        file_count (str): if single, allows user to upload one file. If "multiple", user uploads multiple files. If "directory", user uploads all files in selected directory. Return type will be list for each file in case of "multiple" or "directory".
        type (str): Type of value to be returned by component. "file" returns a temporary file object whose path can be retrieved by file_obj.name, "binary" returns an bytes object.
        label (str): component name in interface.
        keep_filename (bool): DEPRECATED. Original filename always kept.
        optional (bool): If True, the interface can be submitted with no uploaded image, in which case the input value is None.
        """
        warnings.warn(
            "Usage of gradio.inputs is deprecated, and will not be supported in the future, please import your components from gradio.components",
        )
        super().__init__(
            file_count=file_count,
            type=type,
            label=label,
            keep_filename=keep_filename,
            optional=optional,
        )


class Dataframe(components.Dataframe):
    """
    Component accepts 2D input through a spreadsheet interface.
    Input type: Union[pandas.DataFrame, numpy.array, List[Union[str, float]], List[List[Union[str, float]]]]
    """

    def __init__(
        self,
        headers: Optional[List[str]] = None,
        row_count: int = 3,
        col_count: Optional[int] = 3,
        datatype: str | List[str] = "str",
        col_width: int | List[int] = None,
        default: Optional[List[List[Any]]] = None,
        type: str = "pandas",
        label: Optional[str] = None,
        optional: bool = False,
    ):
        """
        Parameters:
        headers (List[str]): Header names to dataframe. If None, no headers are shown.
        row_count (int): Limit number of rows for input.
        col_count (int): Limit number of columns for input. If equal to 1, return data will be one-dimensional. Ignored if `headers` is provided.
        datatype (Union[str, List[str]]): Datatype of values in sheet. Can be provided per column as a list of strings, or for the entire sheet as a single string. Valid datatypes are "str", "number", "bool", and "date".
        col_width (Union[int, List[int]]): Width of columns in pixels. Can be provided as single value or list of values per column.
        default (List[List[Any]]): Default value
        type (str): Type of value to be returned by component. "pandas" for pandas dataframe, "numpy" for numpy array, or "array" for a Python array.
        label (str): component name in interface.
        optional (bool): this parameter is ignored.
        """
        warnings.warn(
            "Usage of gradio.inputs is deprecated, and will not be supported in the future, please import your components from gradio.components",
        )
        super().__init__(
            value=default,
            headers=headers,
            row_count=row_count,
            col_count=col_count,
            datatype=datatype,
            col_width=col_width,
            type=type,
            label=label,
            optional=optional,
        )


class Timeseries(components.Timeseries):
    """
    Component accepts pandas.DataFrame uploaded as a timeseries csv file.
    Input type: pandas.DataFrame
    """

    def __init__(
        self,
        x: Optional[str] = None,
        y: str | List[str] = None,
        label: Optional[str] = None,
        optional: bool = False,
    ):
        """
        Parameters:
        x (str): Column name of x (time) series. None if csv has no headers, in which case first column is x series.
        y (Union[str, List[str]]): Column name of y series, or list of column names if multiple series. None if csv has no headers, in which case every column after first is a y series.
        label (str): component name in interface.
        optional (bool): If True, the interface can be submitted with no uploaded csv file, in which case the input value is None.
        """
        warnings.warn(
            "Usage of gradio.inputs is deprecated, and will not be supported in the future, please import your components from gradio.components",
        )
        super().__init__(x=x, y=y, label=label, optional=optional)


class State(components.State):
    """
    Special hidden component that stores state across runs of the interface.
    Input type: Any
    """

    def __init__(
        self,
        label: str = None,
        default: Any = None,
    ):
        """
        Parameters:
        label (str): component name in interface (not used).
        default (Any): the initial value of the state.
        optional (bool): this parameter is ignored.
        """
        warnings.warn(
            "Usage of gradio.inputs is deprecated, and will not be supported in the future, please import this component as gr.State() from gradio.components",
        )
        super().__init__(value=default, label=label)


class Image3D(components.Model3D):
    """
    Used for 3D image model output.
    Input type: File object of type (.obj, glb, or .gltf)
    """

    def __init__(
        self,
        label: Optional[str] = None,
        optional: bool = False,
    ):
        """
        Parameters:
        label (str): component name in interface.
        optional (bool): If True, the interface can be submitted with no uploaded image, in which case the input value is None.
        """
        warnings.warn(
            "Usage of gradio.outputs is deprecated, and will not be supported in the future, please import your components from gradio.components",
        )
        super().__init__(label=label, optional=optional)
