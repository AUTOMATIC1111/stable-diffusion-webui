import pkgutil

import gradio.components as components
import gradio.inputs as inputs
import gradio.outputs as outputs
import gradio.processing_utils
import gradio.templates
from gradio.blocks import Blocks
from gradio.components import (
    HTML,
    JSON,
    Audio,
    Button,
    Carousel,
    Chatbot,
    Checkbox,
    Checkboxgroup,
    CheckboxGroup,
    ColorPicker,
    DataFrame,
    Dataframe,
    Dataset,
    Dropdown,
    File,
    Gallery,
    Highlight,
    Highlightedtext,
    HighlightedText,
    Image,
    Interpretation,
    Json,
    Label,
    LinePlot,
    Markdown,
    Model3D,
    Number,
    Plot,
    Radio,
    ScatterPlot,
    Slider,
    State,
    StatusTracker,
    Text,
    Textbox,
    TimeSeries,
    Timeseries,
    UploadButton,
    Variable,
    Video,
    component,
)
from gradio.exceptions import Error
from gradio.flagging import (
    CSVLogger,
    FlaggingCallback,
    HuggingFaceDatasetJSONSaver,
    HuggingFaceDatasetSaver,
    SimpleCSVLogger,
)
from gradio.helpers import Progress
from gradio.helpers import create_examples as Examples
from gradio.helpers import make_waveform, skip, update
from gradio.interface import Interface, TabbedInterface, close_all
from gradio.ipython_ext import load_ipython_extension
from gradio.layouts import Accordion, Box, Column, Group, Row, Tab, TabItem, Tabs
from gradio.mix import Parallel, Series
from gradio.routes import Request, mount_gradio_app
from gradio.templates import (
    Files,
    ImageMask,
    ImagePaint,
    List,
    Matrix,
    Mic,
    Microphone,
    Numpy,
    Paint,
    Pil,
    PlayableVideo,
    Sketchpad,
    TextArea,
    Webcam,
)

current_pkg_version = (
    (pkgutil.get_data(__name__, "version.txt") or b"").decode("ascii").strip()
)
__version__ = current_pkg_version
