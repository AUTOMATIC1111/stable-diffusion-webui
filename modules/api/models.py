import inspect
from typing import Any, Optional, Dict, List
from pydantic import BaseModel, Field, create_model # pylint: disable=no-name-in-module
from inflection import underscore
from modules.processing import StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img
import modules.shared as shared

API_NOT_ALLOWED = [
    "self",
    "kwargs",
    "sd_model",
    "outpath_samples",
    "outpath_grids",
    "sampler_index",
    "extra_generation_params",
    "overlay_images",
    "do_not_reload_embeddings",
    "seed_enable_extras",
    "prompt_for_display",
    "sampler_noise_scheduler_override",
    "ddim_discretize"
]

class ModelDef(BaseModel):
    """Assistance Class for Pydantic Dynamic Model Generation"""

    field: str
    field_alias: str
    field_type: Any
    field_value: Any
    field_exclude: bool = False


class PydanticModelGenerator:
    """
    Takes in created classes and stubs them out in a way FastAPI/Pydantic is happy about:
    source_data is a snapshot of the default values produced by the class
    params are the names of the actual keys required by __init__
    """

    def __init__(
        self,
        model_name: str = None,
        class_instance = None,
        additional_fields = None,
    ):
        def field_type_generator(_k, v):
            # field_type = str if not overrides.get(k) else overrides[k]["type"]
            # print(k, v.annotation, v.default)
            field_type = v.annotation

            return Optional[field_type]

        def merge_class_params(class_):
            all_classes = list(filter(lambda x: x is not object, inspect.getmro(class_)))
            parameters = {}
            for classes in all_classes:
                parameters = {**parameters, **inspect.signature(classes.__init__).parameters}
            return parameters


        self._model_name = model_name
        self._class_data = merge_class_params(class_instance)

        self._model_def = [
            ModelDef(
                field=underscore(k),
                field_alias=k,
                field_type=field_type_generator(k, v),
                field_value=v.default
            )
            for (k,v) in self._class_data.items() if k not in API_NOT_ALLOWED
        ]

        for fld in additional_fields:
            self._model_def.append(ModelDef(
                field=underscore(fld["key"]),
                field_alias=fld["key"],
                field_type=fld["type"],
                field_value=fld["default"],
                field_exclude=fld["exclude"] if "exclude" in fld else False))

    def generate_model(self):
        """
        Creates a pydantic BaseModel
        from the json and overrides provided at initialization
        """
        model_fields = { d.field: (d.field_type, Field(default=d.field_value, alias=d.field_alias, exclude=d.field_exclude)) for d in self._model_def }
        DynamicModel = create_model(self._model_name, **model_fields)
        DynamicModel.__config__.allow_population_by_field_name = True
        DynamicModel.__config__.allow_mutation = True
        return DynamicModel


class IPAdapterItem(BaseModel):
    adapter: str = Field(title="Adapter", default="Base", description="Adapter to use")
    image: str = Field(title="Image", default="", description="Adapter image, must be a base64 string containing the image's data.")
    scale: float = Field(title="Scale", default=0.5, gt=0, le=1, description="Scale of the adapter image, must be between 0 and 1.")


class FaceIDItem(BaseModel):
    model: str = Field(title="Model", default="FaceID Base",description="The FaceID model to use.")
    image: str = Field(title="Image", default="", description="Source face image, must be a base64 string containing the image's data.")
    scale: float = Field(title="Scale", default=1, gt=0, le=1, description="Scale of the source face, must be between 0 and 1.")
    structure: float = Field(title="Structure", default=1, gt=0, le=1, description="Structure to use, must be between 0 and 1.")
    rank: float = Field(title="Rank", default=128, ge=4, le=256, description="Rank to use, must be between 4 and 256.")
    override_sampler: bool = Field(title="Override Sampler", default=True, description="Should the sampler be overriden?")
    tokens: int = Field(title="Tokens", default=4, ge=1, le=16, description="Amount of tokens to use, must be between 1 and 16.")
    cache_model: bool = Field(title="Cache", default=True, description="Should the model be cached?")


StableDiffusionTxt2ImgProcessingAPI = PydanticModelGenerator(
    "StableDiffusionProcessingTxt2Img",
    StableDiffusionProcessingTxt2Img,
    [
        {"key": "sampler_index", "type": str, "default": "Euler"},
        {"key": "script_name", "type": str, "default": None},
        {"key": "script_args", "type": list, "default": []},
        {"key": "send_images", "type": bool, "default": True},
        {"key": "save_images", "type": bool, "default": False},
        {"key": "alwayson_scripts", "type": dict, "default": {}},
        {"key": "ip_adapter", "type": Optional[IPAdapterItem], "default": None, "exclude": True},
        {"key": "face_id", "type": Optional[FaceIDItem], "default": None, "exclude": True},
    ]
).generate_model()

StableDiffusionImg2ImgProcessingAPI = PydanticModelGenerator(
    "StableDiffusionProcessingImg2Img",
    StableDiffusionProcessingImg2Img,
    [
        {"key": "sampler_index", "type": str, "default": "Euler"},
        {"key": "init_images", "type": list, "default": None},
        {"key": "denoising_strength", "type": float, "default": 0.75},
        {"key": "mask", "type": str, "default": None},
        {"key": "include_init_images", "type": bool, "default": False, "exclude": True},
        {"key": "script_name", "type": str, "default": None},
        {"key": "script_args", "type": list, "default": []},
        {"key": "send_images", "type": bool, "default": True},
        {"key": "save_images", "type": bool, "default": False},
        {"key": "alwayson_scripts", "type": dict, "default": {}},
        {"key": "ip_adapter", "type": Optional[IPAdapterItem], "default": None, "exclude": True},
        {"key": "face_id", "type": Optional[FaceIDItem], "default": None, "exclude": True},
    ]
).generate_model()

class TextToImageResponse(BaseModel):
    images: List[str] = Field(default=None, title="Image", description="The generated image in base64 format.")
    parameters: dict
    info: str

class ImageToImageResponse(BaseModel):
    images: List[str] = Field(default=None, title="Image", description="The generated image in base64 format.")
    parameters: dict
    info: str

class ExtrasBaseRequest(BaseModel):
    resize_mode: float = Field(default=0, title="Resize Mode", description="Sets the resize mode: 0 to upscale by upscaling_resize amount, 1 to upscale up to upscaling_resize_h x upscaling_resize_w.")
    show_extras_results: bool = Field(default=True, title="Show results", description="Should the backend return the generated image?")
    gfpgan_visibility: float = Field(default=0, title="GFPGAN Visibility", ge=0, le=1, allow_inf_nan=False, description="Sets the visibility of GFPGAN, values should be between 0 and 1.")
    codeformer_visibility: float = Field(default=0, title="CodeFormer Visibility", ge=0, le=1, allow_inf_nan=False, description="Sets the visibility of CodeFormer, values should be between 0 and 1.")
    codeformer_weight: float = Field(default=0, title="CodeFormer Weight", ge=0, le=1, allow_inf_nan=False, description="Sets the weight of CodeFormer, values should be between 0 and 1.")
    upscaling_resize: float = Field(default=2, title="Upscaling Factor", ge=1, le=8, description="By how much to upscale the image, only used when resize_mode=0.")
    upscaling_resize_w: int = Field(default=512, title="Target Width", ge=1, description="Target width for the upscaler to hit. Only used when resize_mode=1.")
    upscaling_resize_h: int = Field(default=512, title="Target Height", ge=1, description="Target height for the upscaler to hit. Only used when resize_mode=1.")
    upscaling_crop: bool = Field(default=True, title="Crop to fit", description="Should the upscaler crop the image to fit in the chosen size?")
    upscaler_1: str = Field(default="None", title="Main upscaler", description=f"The name of the main upscaler to use, it has to be one of this list: {' , '.join([x.name for x in shared.sd_upscalers])}")
    upscaler_2: str = Field(default="None", title="Secondary upscaler", description=f"The name of the secondary upscaler to use, it has to be one of this list: {' , '.join([x.name for x in shared.sd_upscalers])}")
    extras_upscaler_2_visibility: float = Field(default=0, title="Secondary upscaler visibility", ge=0, le=1, allow_inf_nan=False, description="Sets the visibility of secondary upscaler, values should be between 0 and 1.")
    upscale_first: bool = Field(default=False, title="Upscale first", description="Should the upscaler run before restoring faces?")

class ExtraBaseResponse(BaseModel):
    html_info: str = Field(title="HTML info", description="A series of HTML tags containing the process info.")

class ExtrasSingleImageRequest(ExtrasBaseRequest):
    image: str = Field(default="", title="Image", description="Image to work on, must be a Base64 string containing the image's data.")

class ExtrasSingleImageResponse(ExtraBaseResponse):
    image: str = Field(default=None, title="Image", description="The generated image in base64 format.")

class FileData(BaseModel):
    data: str = Field(title="File data", description="Base64 representation of the file")
    name: str = Field(title="File name")

class ExtrasBatchImagesRequest(ExtrasBaseRequest):
    imageList: List[FileData] = Field(title="Images", description="List of images to work on. Must be Base64 strings")

class ExtrasBatchImagesResponse(ExtraBaseResponse):
    images: List[str] = Field(title="Images", description="The generated images in base64 format.")

class PNGInfoRequest(BaseModel):
    image: str = Field(title="Image", description="The base64 encoded PNG image")

class PNGInfoResponse(BaseModel):
    info: str = Field(title="Image info", description="A string with the parameters used to generate the image")
    items: dict = Field(title="Items", description="A dictionary containing all the other fields the image had")
    parameters: dict = Field(title="Parameters", description="A dictionary with parsed generation info fields")

class LogRequest(BaseModel):
    lines: int = Field(default=100, title="Lines", description="How many lines to return")
    clear: bool = Field(default=False, title="Clear", description="Should the log be cleared after returning the lines?")

class ProgressRequest(BaseModel):
    skip_current_image: bool = Field(default=False, title="Skip current image", description="Skip current image serialization")

class ProgressResponse(BaseModel):
    progress: float = Field(title="Progress", description="The progress with a range of 0 to 1")
    eta_relative: float = Field(title="ETA in secs")
    state: dict = Field(title="State", description="The current state snapshot")
    current_image: str = Field(default=None, title="Current image", description="The current image in base64 format. opts.show_progress_every_n_steps is required for this to work.")
    textinfo: str = Field(default=None, title="Info text", description="Info text used by WebUI.")

class InterrogateRequest(BaseModel):
    image: str = Field(default="", title="Image", description="Image to work on, must be a Base64 string containing the image's data.")
    model: str = Field(default="clip", title="Model", description="The interrogate model used.")

class InterrogateResponse(BaseModel):
    caption: str = Field(default=None, title="Caption", description="The generated caption for the image.")

class TrainResponse(BaseModel):
    info: str = Field(title="Train info", description="Response string from train embedding or hypernetwork task.")

class CreateResponse(BaseModel):
    info: str = Field(title="Create info", description="Response string from create embedding or hypernetwork task.")

class PreprocessResponse(BaseModel):
    info: str = Field(title="Preprocess info", description="Response string from preprocessing task.")

fields = {}
for key, metadata in shared.opts.data_labels.items():
    value = shared.opts.data.get(key) or shared.opts.data_labels[key].default
    optType = shared.opts.typemap.get(type(metadata.default), type(value))

    if metadata is not None:
        fields.update({key: (Optional[optType], Field(
            default=metadata.default, description=metadata.label))})
    else:
        fields.update({key: (Optional[optType], Field())})

OptionsModel = create_model("Options", **fields)

flags = {}
_options = vars(shared.parser)['_option_string_actions']
for key in _options:
    if _options[key].dest != 'help':
        flag = _options[key]
        _type = str
        if _options[key].default is not None:
            _type = type(_options[key].default)
        flags.update({flag.dest: (_type, Field(default=flag.default, description=flag.help))})

FlagsModel = create_model("Flags", **flags)

class SamplerItem(BaseModel):
    name: str = Field(title="Name")
    aliases: List[str] = Field(title="Aliases")
    options: Dict[str, str] = Field(title="Options")

class SDVaeItem(BaseModel):
    model_name: str = Field(title="Model Name")
    filename: str = Field(title="Filename")

class UpscalerItem(BaseModel):
    name: str = Field(title="Name")
    model_name: Optional[str] = Field(title="Model Name")
    model_path: Optional[str] = Field(title="Path")
    model_url: Optional[str] = Field(title="URL")
    scale: Optional[float] = Field(title="Scale")

class SDModelItem(BaseModel):
    title: str = Field(title="Title")
    model_name: str = Field(title="Model Name")
    filename: str = Field(title="Filename")
    type: str = Field(title="Model type")
    sha256: Optional[str] = Field(title="SHA256 hash")
    hash: Optional[str] = Field(title="Short hash")
    config: Optional[str] = Field(title="Config file")

class HypernetworkItem(BaseModel):
    name: str = Field(title="Name")
    path: Optional[str] = Field(title="Path")

class FaceRestorerItem(BaseModel):
    name: str = Field(title="Name")
    cmd_dir: Optional[str] = Field(title="Path")

class RealesrganItem(BaseModel):
    name: str = Field(title="Name")
    path: Optional[str] = Field(title="Path")
    scale: Optional[int] = Field(title="Scale")

class StyleItem(BaseModel):
    name: str = Field(title="Name")
    prompt: Optional[str] = Field(title="Prompt")
    negative_prompt: Optional[str] = Field(title="Negative Prompt")
    extra: Optional[str] = Field(title="Extra")
    filename: Optional[str] = Field(title="Filename")
    preview: Optional[str] = Field(title="Preview")

class ExtraNetworkItem(BaseModel):
    name: str = Field(title="Name")
    type: str = Field(title="Type")
    title: Optional[str] = Field(title="Title")
    fullname: Optional[str] = Field(title="Fullname")
    filename: Optional[str] = Field(title="Filename")
    hash: Optional[str] = Field(title="Hash")
    preview: Optional[str] = Field(title="Preview image URL")
    # description: Optional[str] = Field(title="Description")
    # info: Optional[str] = Field(title="Information")
    # metadata: Optional[Any] = Field(title="Metadata")
    # local: Optional[str] = Field(title="Local")

class ArtistItem(BaseModel):
    name: str = Field(title="Name")
    score: float = Field(title="Score")
    category: str = Field(title="Category")

class EmbeddingItem(BaseModel):
    step: Optional[int] = Field(title="Step", description="The number of steps that were used to train this embedding, if available")
    sd_checkpoint: Optional[str] = Field(title="SD Checkpoint", description="The hash of the checkpoint this embedding was trained on, if available")
    sd_checkpoint_name: Optional[str] = Field(title="SD Checkpoint Name", description="The name of the checkpoint this embedding was trained on, if available. Note that this is the name that was used by the trainer; for a stable identifier, use `sd_checkpoint` instead")
    shape: int = Field(title="Shape", description="The length of each individual vector in the embedding")
    vectors: int = Field(title="Vectors", description="The number of vectors in the embedding")

class EmbeddingsResponse(BaseModel):
    loaded: Dict[str, EmbeddingItem] = Field(title="Loaded", description="Embeddings loaded for the current model")
    skipped: Dict[str, EmbeddingItem] = Field(title="Skipped", description="Embeddings skipped for the current model (likely due to architecture incompatibility)")

class MemoryResponse(BaseModel):
    ram: dict = Field(title="RAM", description="System memory stats")
    cuda: dict = Field(title="CUDA", description="nVidia CUDA memory stats")

class ScriptsList(BaseModel):
    txt2img: list = Field(default=None, title="Txt2img", description="Titles of scripts (txt2img)")
    img2img: list = Field(default=None, title="Img2img", description="Titles of scripts (img2img)")
    control: list = Field(default=None, title="Control", description="Titles of scripts (control)")

class ScriptArg(BaseModel):
    label: str = Field(default=None, title="Label", description="Name of the argument in UI")
    value: Optional[Any] = Field(default=None, title="Value", description="Default value of the argument")
    minimum: Optional[Any] = Field(default=None, title="Minimum", description="Minimum allowed value for the argumentin UI")
    maximum: Optional[Any] = Field(default=None, title="Minimum", description="Maximum allowed value for the argumentin UI")
    step: Optional[Any] = Field(default=None, title="Minimum", description="Step for changing value of the argumentin UI")
    choices: Optional[Any] = Field(default=None, title="Choices", description="Possible values for the argument")

class ScriptInfo(BaseModel):
    name: str = Field(default=None, title="Name", description="Script name")
    is_alwayson: bool = Field(default=None, title="IsAlwayson", description="Flag specifying whether this script is an alwayson script")
    is_img2img: bool = Field(default=None, title="IsImg2img", description="Flag specifying whether this script is an img2img script")
    args: List[ScriptArg] = Field(title="Arguments", description="List of script's arguments")

class ExtensionItem(BaseModel):
    name: str = Field(title="Name", description="Extension name")
    remote: str = Field(title="Remote", description="Extension Repository URL")
    branch: str = Field(title="Branch", description="Extension Repository Branch")
    commit_hash: str = Field(title="Commit Hash", description="Extension Repository Commit Hash")
    version: str = Field(title="Version", description="Extension Version")
    commit_date: str = Field(title="Commit Date", description="Extension Repository Commit Date")
    enabled: bool = Field(title="Enabled", description="Flag specifying whether this extension is enabled")
