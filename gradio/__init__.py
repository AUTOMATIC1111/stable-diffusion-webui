import pkgutil

import gradio.components as components
import gradio.inputs as inputs
import gradio.outputs as outputs
import gradio.processing_utils
import gradio.templates
import gradio.themes as themes
from gradio.blocks import Blocks
from gradio.components import (
    HTML,
    JSON,
    AnnotatedImage,
    Annotatedimage,
    Audio,
    BarPlot,
    Button,
    Carousel,
    Chatbot,
    Checkbox,
    CheckboxGroup,
    Checkboxgroup,
    Code,
    ColorPicker,
    DataFrame,
    Dataframe,
    Dataset,
    Dropdown,
    File,
    Gallery,
    Highlight,
    HighlightedText,
    Highlightedtext,
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
from gradio.deploy_space import deploy
from gradio.events import SelectData
from gradio.exceptions import Error
from gradio.external import load
from gradio.flagging import (
    CSVLogger,
    FlaggingCallback,
    HuggingFaceDatasetJSONSaver,
    HuggingFaceDatasetSaver,
    SimpleCSVLogger,
)
from gradio.helpers import EventData, Progress, make_waveform, skip, update
from gradio.helpers import create_examples as Examples  # noqa: N812
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
from gradio.themes import Base as Theme

current_pkg_version = (
    (pkgutil.get_data(__name__, "version.txt") or b"").decode("ascii").strip()
)
__version__ = current_pkg_version
