from __future__ import annotations

import typing
from typing import Any, Callable, Tuple

import numpy as np
from PIL.Image import Image

from gradio import components


class TextArea(components.Textbox):
    """
    Sets: lines=7
    """

    is_template = True

    def __init__(
        self,
        value: str | Callable | None = "",
        *,
        lines: int = 7,
        max_lines: int = 20,
        placeholder: str | None = None,
        label: str | None = None,
        show_label: bool = True,
        interactive: bool | None = None,
        visible: bool = True,
        elem_id: str | None = None,
        **kwargs,
    ):
        super().__init__(
            value=value,
            lines=lines,
            max_lines=max_lines,
            placeholder=placeholder,
            label=label,
            show_label=show_label,
            interactive=interactive,
            visible=visible,
            elem_id=elem_id,
            **kwargs,
        )


class Webcam(components.Image):
    """
    Sets: source="webcam", interactive=True
    """

    is_template = True

    def __init__(
        self,
        value: str | Image | np.ndarray | None = None,
        *,
        shape: Tuple[int, int] | None = None,
        image_mode: str = "RGB",
        invert_colors: bool = False,
        source: str = "webcam",
        tool: str | None = None,
        type: str = "numpy",
        label: str | None = None,
        show_label: bool = True,
        interactive: bool | None = True,
        visible: bool = True,
        streaming: bool = False,
        elem_id: str | None = None,
        mirror_webcam: bool = True,
        **kwargs,
    ):
        super().__init__(
            value=value,
            shape=shape,
            image_mode=image_mode,
            invert_colors=invert_colors,
            source=source,
            tool=tool,
            type=type,
            label=label,
            show_label=show_label,
            interactive=interactive,
            visible=visible,
            streaming=streaming,
            elem_id=elem_id,
            mirror_webcam=mirror_webcam,
            **kwargs,
        )


class Sketchpad(components.Image):
    """
    Sets: image_mode="L", source="canvas", shape=(28, 28), invert_colors=True, interactive=True
    """

    is_template = True

    def __init__(
        self,
        value: str | Image | np.ndarray | None = None,
        *,
        shape: Tuple[int, int] = (28, 28),
        image_mode: str = "L",
        invert_colors: bool = True,
        source: str = "canvas",
        tool: str | None = None,
        type: str = "numpy",
        label: str | None = None,
        show_label: bool = True,
        interactive: bool | None = True,
        visible: bool = True,
        streaming: bool = False,
        elem_id: str | None = None,
        mirror_webcam: bool = True,
        **kwargs,
    ):
        super().__init__(
            value=value,
            shape=shape,
            image_mode=image_mode,
            invert_colors=invert_colors,
            source=source,
            tool=tool,
            type=type,
            label=label,
            show_label=show_label,
            interactive=interactive,
            visible=visible,
            streaming=streaming,
            elem_id=elem_id,
            mirror_webcam=mirror_webcam,
            **kwargs,
        )


class Paint(components.Image):
    """
    Sets: source="canvas", tool="color-sketch", interactive=True
    """

    is_template = True

    def __init__(
        self,
        value: str | Image | np.ndarray | None = None,
        *,
        shape: Tuple[int, int] | None = None,
        image_mode: str = "RGB",
        invert_colors: bool = False,
        source: str = "canvas",
        tool: str = "color-sketch",
        type: str = "numpy",
        label: str | None = None,
        show_label: bool = True,
        interactive: bool | None = True,
        visible: bool = True,
        streaming: bool = False,
        elem_id: str | None = None,
        mirror_webcam: bool = True,
        **kwargs,
    ):
        super().__init__(
            value=value,
            shape=shape,
            image_mode=image_mode,
            invert_colors=invert_colors,
            source=source,
            tool=tool,
            type=type,
            label=label,
            show_label=show_label,
            interactive=interactive,
            visible=visible,
            streaming=streaming,
            elem_id=elem_id,
            mirror_webcam=mirror_webcam,
            **kwargs,
        )


class ImageMask(components.Image):
    """
    Sets: source="upload", tool="sketch", interactive=True
    """

    is_template = True

    def __init__(
        self,
        value: str | Image | np.ndarray | None = None,
        *,
        shape: Tuple[int, int] | None = None,
        image_mode: str = "RGB",
        invert_colors: bool = False,
        source: str = "upload",
        tool: str = "sketch",
        type: str = "numpy",
        label: str | None = None,
        show_label: bool = True,
        interactive: bool | None = True,
        visible: bool = True,
        streaming: bool = False,
        elem_id: str | None = None,
        mirror_webcam: bool = True,
        **kwargs,
    ):
        super().__init__(
            value=value,
            shape=shape,
            image_mode=image_mode,
            invert_colors=invert_colors,
            source=source,
            tool=tool,
            type=type,
            label=label,
            show_label=show_label,
            interactive=interactive,
            visible=visible,
            streaming=streaming,
            elem_id=elem_id,
            mirror_webcam=mirror_webcam,
            **kwargs,
        )


class ImagePaint(components.Image):
    """
    Sets: source="upload", tool="color-sketch", interactive=True
    """

    is_template = True

    def __init__(
        self,
        value: str | Image | np.ndarray | None = None,
        *,
        shape: Tuple[int, int] | None = None,
        image_mode: str = "RGB",
        invert_colors: bool = False,
        source: str = "upload",
        tool: str = "color-sketch",
        type: str = "numpy",
        label: str | None = None,
        show_label: bool = True,
        interactive: bool | None = True,
        visible: bool = True,
        streaming: bool = False,
        elem_id: str | None = None,
        mirror_webcam: bool = True,
        **kwargs,
    ):
        super().__init__(
            value=value,
            shape=shape,
            image_mode=image_mode,
            invert_colors=invert_colors,
            source=source,
            tool=tool,
            type=type,
            label=label,
            show_label=show_label,
            interactive=interactive,
            visible=visible,
            streaming=streaming,
            elem_id=elem_id,
            mirror_webcam=mirror_webcam,
            **kwargs,
        )


class Pil(components.Image):
    """
    Sets: type="pil"
    """

    is_template = True

    def __init__(
        self,
        value: str | Image | np.ndarray | None = None,
        *,
        shape: Tuple[int, int] | None = None,
        image_mode: str = "RGB",
        invert_colors: bool = False,
        source: str = "upload",
        tool: str | None = None,
        type: str = "pil",
        label: str | None = None,
        show_label: bool = True,
        interactive: bool | None = None,
        visible: bool = True,
        streaming: bool = False,
        elem_id: str | None = None,
        mirror_webcam: bool = True,
        **kwargs,
    ):
        super().__init__(
            value=value,
            shape=shape,
            image_mode=image_mode,
            invert_colors=invert_colors,
            source=source,
            tool=tool,
            type=type,
            label=label,
            show_label=show_label,
            interactive=interactive,
            visible=visible,
            streaming=streaming,
            elem_id=elem_id,
            mirror_webcam=mirror_webcam,
            **kwargs,
        )


class PlayableVideo(components.Video):
    """
    Sets: format="mp4"
    """

    is_template = True

    def __init__(
        self,
        value: str | Callable | None = None,
        *,
        format: str | None = "mp4",
        source: str = "upload",
        label: str | None = None,
        show_label: bool = True,
        interactive: bool | None = None,
        visible: bool = True,
        elem_id: str | None = None,
        mirror_webcam: bool = True,
        include_audio: bool | None = None,
        **kwargs,
    ):
        super().__init__(
            value=value,
            format=format,
            source=source,
            label=label,
            show_label=show_label,
            interactive=interactive,
            visible=visible,
            elem_id=elem_id,
            mirror_webcam=mirror_webcam,
            include_audio=include_audio,
            **kwargs,
        )


class Microphone(components.Audio):
    """
    Sets: source="microphone"
    """

    is_template = True

    def __init__(
        self,
        value: str | Tuple[int, np.ndarray] | Callable | None = None,
        *,
        source: str = "microphone",
        type: str = "numpy",
        label: str | None = None,
        show_label: bool = True,
        interactive: bool | None = None,
        visible: bool = True,
        streaming: bool = False,
        elem_id: str | None = None,
        **kwargs,
    ):
        super().__init__(
            value=value,
            source=source,
            type=type,
            label=label,
            show_label=show_label,
            interactive=interactive,
            visible=visible,
            streaming=streaming,
            elem_id=elem_id,
            **kwargs,
        )


class Files(components.File):
    """
    Sets: file_count="multiple"
    """

    is_template = True

    def __init__(
        self,
        value: str | typing.List[str] | Callable | None = None,
        *,
        file_count: str = "multiple",
        type: str = "file",
        label: str | None = None,
        show_label: bool = True,
        interactive: bool | None = None,
        visible: bool = True,
        elem_id: str | None = None,
        **kwargs,
    ):
        super().__init__(
            value=value,
            file_count=file_count,
            type=type,
            label=label,
            show_label=show_label,
            interactive=interactive,
            visible=visible,
            elem_id=elem_id,
            **kwargs,
        )


class Numpy(components.Dataframe):
    """
    Sets: type="numpy"
    """

    is_template = True

    def __init__(
        self,
        value: typing.List[typing.List[Any]] | Callable | None = None,
        *,
        headers: typing.List[str] | None = None,
        row_count: int | Tuple[int, str] = (1, "dynamic"),
        col_count: int | Tuple[int, str] | None = None,
        datatype: str | typing.List[str] = "str",
        type: str = "numpy",
        max_rows: int | None = 20,
        max_cols: int | None = None,
        overflow_row_behaviour: str = "paginate",
        label: str | None = None,
        show_label: bool = True,
        interactive: bool | None = None,
        visible: bool = True,
        elem_id: str | None = None,
        wrap: bool = False,
        **kwargs,
    ):
        super().__init__(
            value=value,
            headers=headers,
            row_count=row_count,
            col_count=col_count,
            datatype=datatype,
            type=type,
            max_rows=max_rows,
            max_cols=max_cols,
            overflow_row_behaviour=overflow_row_behaviour,
            label=label,
            show_label=show_label,
            interactive=interactive,
            visible=visible,
            elem_id=elem_id,
            wrap=wrap,
            **kwargs,
        )


class Matrix(components.Dataframe):
    """
    Sets: type="array"
    """

    is_template = True

    def __init__(
        self,
        value: typing.List[typing.List[Any]] | Callable | None = None,
        *,
        headers: typing.List[str] | None = None,
        row_count: int | Tuple[int, str] = (1, "dynamic"),
        col_count: int | Tuple[int, str] | None = None,
        datatype: str | typing.List[str] = "str",
        type: str = "array",
        max_rows: int | None = 20,
        max_cols: int | None = None,
        overflow_row_behaviour: str = "paginate",
        label: str | None = None,
        show_label: bool = True,
        interactive: bool | None = None,
        visible: bool = True,
        elem_id: str | None = None,
        wrap: bool = False,
        **kwargs,
    ):
        super().__init__(
            value=value,
            headers=headers,
            row_count=row_count,
            col_count=col_count,
            datatype=datatype,
            type=type,
            max_rows=max_rows,
            max_cols=max_cols,
            overflow_row_behaviour=overflow_row_behaviour,
            label=label,
            show_label=show_label,
            interactive=interactive,
            visible=visible,
            elem_id=elem_id,
            wrap=wrap,
            **kwargs,
        )


class List(components.Dataframe):
    """
    Sets: type="array", col_count=1
    """

    is_template = True

    def __init__(
        self,
        value: typing.List[typing.List[Any]] | Callable | None = None,
        *,
        headers: typing.List[str] | None = None,
        row_count: int | Tuple[int, str] = (1, "dynamic"),
        col_count: int | Tuple[int, str] = 1,
        datatype: str | typing.List[str] = "str",
        type: str = "array",
        max_rows: int | None = 20,
        max_cols: int | None = None,
        overflow_row_behaviour: str = "paginate",
        label: str | None = None,
        show_label: bool = True,
        interactive: bool | None = None,
        visible: bool = True,
        elem_id: str | None = None,
        wrap: bool = False,
        **kwargs,
    ):
        super().__init__(
            value=value,
            headers=headers,
            row_count=row_count,
            col_count=col_count,
            datatype=datatype,
            type=type,
            max_rows=max_rows,
            max_cols=max_cols,
            overflow_row_behaviour=overflow_row_behaviour,
            label=label,
            show_label=show_label,
            interactive=interactive,
            visible=visible,
            elem_id=elem_id,
            wrap=wrap,
            **kwargs,
        )


Mic = Microphone
