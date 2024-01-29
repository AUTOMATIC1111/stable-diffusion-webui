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

### item classes

class ItemSampler(BaseModel):
    name: str = Field(title="Name")
    aliases: List[str] = Field(title="Aliases")
    options: Dict[str, str] = Field(title="Options")

class ItemVae(BaseModel):
    model_name: str = Field(title="Model Name")
    filename: str = Field(title="Filename")

class ItemUpscaler(BaseModel):
    name: str = Field(title="Name")
    model_name: Optional[str] = Field(title="Model Name")
    model_path: Optional[str] = Field(title="Path")
    model_url: Optional[str] = Field(title="URL")
    scale: Optional[float] = Field(title="Scale")

class ItemModel(BaseModel):
    title: str = Field(title="Title")
    model_name: str = Field(title="Model Name")
    filename: str = Field(title="Filename")
    type: str = Field(title="Model type")
    sha256: Optional[str] = Field(title="SHA256 hash")
    hash: Optional[str] = Field(title="Short hash")
    config: Optional[str] = Field(title="Config file")

class ItemHypernetwork(BaseModel):
    name: str = Field(title="Name")
    path: Optional[str] = Field(title="Path")

class ItemFaceRestorer(BaseModel):
    name: str = Field(title="Name")
    cmd_dir: Optional[str] = Field(title="Path")

class ItemGAN(BaseModel):
    name: str = Field(title="Name")
    path: Optional[str] = Field(title="Path")
    scale: Optional[int] = Field(title="Scale")

class ItemStyle(BaseModel):
    name: str = Field(title="Name")
    prompt: Optional[str] = Field(title="Prompt")
    negative_prompt: Optional[str] = Field(title="Negative Prompt")
    extra: Optional[str] = Field(title="Extra")
    filename: Optional[str] = Field(title="Filename")
    preview: Optional[str] = Field(title="Preview")

class ItemExtraNetwork(BaseModel):
    name: str = Field(title="Name")
    type: str = Field(title="Type")
    title: Optional[str] = Field(title="Title")
    fullname: Optional[str] = Field(title="Fullname")
    filename: Optional[str] = Field(title="Filename")
    hash: Optional[str] = Field(title="Hash")
    preview: Optional[str] = Field(title="Preview image URL")

class ItemArtist(BaseModel):
    name: str = Field(title="Name")
    score: float = Field(title="Score")
    category: str = Field(title="Category")

class ItemEmbedding(BaseModel):
    step: Optional[int] = Field(title="Step", description="The number of steps that were used to train this embedding, if available")
    sd_checkpoint: Optional[str] = Field(title="SD Checkpoint", description="The hash of the checkpoint this embedding was trained on, if available")
    sd_checkpoint_name: Optional[str] = Field(title="SD Checkpoint Name", description="The name of the checkpoint this embedding was trained on, if available. Note that this is the name that was used by the trainer; for a stable identifier, use `sd_checkpoint` instead")
    shape: int = Field(title="Shape", description="The length of each individual vector in the embedding")
    vectors: int = Field(title="Vectors", description="The number of vectors in the embedding")

class ItemIPAdapter(BaseModel):
    adapter: str = Field(title="Adapter", default="Base", description="Adapter to use")
    image: str = Field(title="Image", default="", description="Adapter image, must be a base64 string containing the image's data.")
    scale: float = Field(title="Scale", default=0.5, gt=0, le=1, description="Scale of the adapter image, must be between 0 and 1.")

class ItemFace(BaseModel):
    mode: str = Field(title="Mode", default=["FaceID"], description="The mode to use (available values: FaceID, FaceSwap, PhotoMaker, InstantID).")
    source_images: list[str] = Field(title="Source Images", description="Source face images, must be base64 encoded containing the image's data.")
    ip_model: str = Field(title="IPAdapter Model", default="FaceID Base", description="The IPAdapter model to use.")
    ip_override_sampler: bool = Field(title="IPAdapter Override Sampler", default=True, description="Should the sampler be overriden?")
    ip_cache_model: bool = Field(title="IPAdapter Cache", default=True, description="Should the IPAdapter model be cached?")
    ip_strength: float = Field(title="IPAdapter Strength", default=1, ge=0, le=2, description="IPAdapter strength of the source images, must be between 0.0 and 2.0.")
    ip_structure: float = Field(title="IPAdapter Structure", default=1, ge=0, le=1, description="IPAdapter structure to use, must be between 0.0 and 1.0.")
    id_strength: float = Field(title="InstantID Strength", default=1, ge=0, le=2, description="InstantID Strength of the source images, must be between 0.0 and 2.0.")
    id_conditioning: float = Field(title="InstantID Condition", default=0.5, ge=0, le=2, description="InstantID control amount, must be between 0.0 and 2.0.")
    id_cache: bool = Field(title="InstantID Cache", default=True, description="Should the InstantID model be cached?")
    pm_trigger: str = Field(title="PhotoMaker Trigger", default="person", description="PhotoMaker trigger word to use.")
    pm_strength: float = Field(title="PhotoMaker Strength", default=1, ge=0, le=2, description="PhotoMaker strength to use, must be between 0.0 and 2.0.")
    pm_start: float = Field(title="PhotoMaker Start", default=0.5, ge=0, le=1, description="PhotoMaker start value, must be between 0.0 and 1.0.")
    fs_cache: bool = Field(title="FaceSwap Cache", default=True, description="Should the FaceSwap model be cached?")

class ScriptArg(BaseModel):
    label: str = Field(default=None, title="Label", description="Name of the argument in UI")
    value: Optional[Any] = Field(default=None, title="Value", description="Default value of the argument")
    minimum: Optional[Any] = Field(default=None, title="Minimum", description="Minimum allowed value for the argumentin UI")
    maximum: Optional[Any] = Field(default=None, title="Minimum", description="Maximum allowed value for the argumentin UI")
    step: Optional[Any] = Field(default=None, title="Minimum", description="Step for changing value of the argumentin UI")
    choices: Optional[Any] = Field(default=None, title="Choices", description="Possible values for the argument")

class ItemScript(BaseModel):
    name: str = Field(default=None, title="Name", description="Script name")
    is_alwayson: bool = Field(default=None, title="IsAlwayson", description="Flag specifying whether this script is an alwayson script")
    is_img2img: bool = Field(default=None, title="IsImg2img", description="Flag specifying whether this script is an img2img script")
    args: List[ScriptArg] = Field(title="Arguments", description="List of script's arguments")

class ItemExtension(BaseModel):
    name: str = Field(title="Name", description="Extension name")
    remote: str = Field(title="Remote", description="Extension Repository URL")
    branch: str = Field(title="Branch", description="Extension Repository Branch")
    commit_hash: str = Field(title="Commit Hash", description="Extension Repository Commit Hash")
    version: str = Field(title="Version", description="Extension Version")
    commit_date: str = Field(title="Commit Date", description="Extension Repository Commit Date")
    enabled: bool = Field(title="Enabled", description="Flag specifying whether this extension is enabled")

### request/response classes

ReqTxt2Img = PydanticModelGenerator(
    "StableDiffusionProcessingTxt2Img",
    StableDiffusionProcessingTxt2Img,
    [
        {"key": "sampler_index", "type": str, "default": "Euler"},
        {"key": "script_name", "type": str, "default": None},
        {"key": "script_args", "type": list, "default": []},
        {"key": "send_images", "type": bool, "default": True},
        {"key": "save_images", "type": bool, "default": False},
        {"key": "alwayson_scripts", "type": dict, "default": {}},
        {"key": "ip_adapter", "type": Optional[ItemIPAdapter], "default": None, "exclude": True},
        {"key": "face", "type": Optional[ItemFace], "default": None, "exclude": True},
    ]
).generate_model()
StableDiffusionTxt2ImgProcessingAPI = ReqTxt2Img

class ResTxt2Img(BaseModel):
    images: List[str] = Field(default=None, title="Image", description="The generated image in base64 format.")
    parameters: dict
    info: str

ReqImg2Img = PydanticModelGenerator(
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
        {"key": "ip_adapter", "type": Optional[ItemIPAdapter], "default": None, "exclude": True},
        {"key": "face_id", "type": Optional[ItemFace], "default": None, "exclude": True},
    ]
).generate_model()
StableDiffusionImg2ImgProcessingAPI = ReqImg2Img

class ResImg2Img(BaseModel):
    images: List[str] = Field(default=None, title="Image", description="The generated image in base64 format.")
    parameters: dict
    info: str

class FileData(BaseModel):
    data: str = Field(title="File data", description="Base64 representation of the file")
    name: str = Field(title="File name")

class ReqProcess(BaseModel):
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

class ResProcess(BaseModel):
    html_info: str = Field(title="HTML info", description="A series of HTML tags containing the process info.")

class ReqProcessImage(ReqProcess):
    image: str = Field(default="", title="Image", description="Image to work on, must be a Base64 string containing the image's data.")

class ResProcessImage(ResProcess):
    image: str = Field(default=None, title="Image", description="The generated image in base64 format.")

class ReqProcessBatch(ReqProcess):
    imageList: List[FileData] = Field(title="Images", description="List of images to work on. Must be Base64 strings")

class ResProcessBatch(ResProcess):
    images: List[str] = Field(title="Images", description="The generated images in base64 format.")

class ReqImageInfo(BaseModel):
    image: str = Field(title="Image", description="The base64 encoded PNG image")

class ResImageInfo(BaseModel):
    info: str = Field(title="Image info", description="A string with the parameters used to generate the image")
    items: dict = Field(title="Items", description="A dictionary containing all the other fields the image had")
    parameters: dict = Field(title="Parameters", description="A dictionary with parsed generation info fields")

class ReqLog(BaseModel):
    lines: int = Field(default=100, title="Lines", description="How many lines to return")
    clear: bool = Field(default=False, title="Clear", description="Should the log be cleared after returning the lines?")

class ReqProgress(BaseModel):
    skip_current_image: bool = Field(default=False, title="Skip current image", description="Skip current image serialization")

class ResProgress(BaseModel):
    progress: float = Field(title="Progress", description="The progress with a range of 0 to 1")
    eta_relative: float = Field(title="ETA in secs")
    state: dict = Field(title="State", description="The current state snapshot")
    current_image: str = Field(default=None, title="Current image", description="The current image in base64 format. opts.show_progress_every_n_steps is required for this to work.")
    textinfo: str = Field(default=None, title="Info text", description="Info text used by WebUI.")

class ReqInterrogate(BaseModel):
    image: str = Field(default="", title="Image", description="Image to work on, must be a Base64 string containing the image's data.")
    model: str = Field(default="clip", title="Model", description="The interrogate model used.")

class ResInterrogate(BaseModel):
    caption: Optional[str] = Field(default=None, title="Caption", description="The generated caption for the image.")
    medium: Optional[str] = Field(default=None, title="Medium", description="Image medium.")
    artist: Optional[str] = Field(default=None, title="Medium", description="Image artist.")
    movement: Optional[str] = Field(default=None, title="Medium", description="Image movement.")
    trending: Optional[str] = Field(default=None, title="Medium", description="Image trending.")
    flavor: Optional[str] = Field(default=None, title="Medium", description="Image flavor.")

class ResTrain(BaseModel):
    info: str = Field(title="Train info", description="Response string from train embedding or hypernetwork task.")

class ResCreate(BaseModel):
    info: str = Field(title="Create info", description="Response string from create embedding or hypernetwork task.")

class ResPreprocess(BaseModel):
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

class ResEmbeddings(BaseModel):
    loaded: Dict[str, ItemEmbedding] = Field(title="Loaded", description="Embeddings loaded for the current model")
    skipped: Dict[str, ItemEmbedding] = Field(title="Skipped", description="Embeddings skipped for the current model (likely due to architecture incompatibility)")

class ResMemory(BaseModel):
    ram: dict = Field(title="RAM", description="System memory stats")
    cuda: dict = Field(title="CUDA", description="nVidia CUDA memory stats")

class ResScripts(BaseModel):
    txt2img: list = Field(default=None, title="Txt2img", description="Titles of scripts (txt2img)")
    img2img: list = Field(default=None, title="Img2img", description="Titles of scripts (img2img)")
    control: list = Field(default=None, title="Control", description="Titles of scripts (control)")

class ResNVML(BaseModel): # definition of http response
    name: str = Field(title="Name")
    version: dict = Field(title="Version")
    pci: dict = Field(title="Version")
    memory: dict = Field(title="Version")
    clock: dict = Field(title="Version")
    load: dict = Field(title="Version")
    power: list = []
    state: str = Field(title="State")
