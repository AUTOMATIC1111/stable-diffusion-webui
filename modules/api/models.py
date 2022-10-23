from pydantic import BaseModel, Field, Json
from typing_extensions import Literal
from modules.shared import sd_upscalers

class TextToImageResponse(BaseModel):
    images: list[str] = Field(default=None, title="Image", description="The generated image in base64 format.")
    parameters: Json
    info: Json

class ExtrasBaseRequest(BaseModel):
    resize_mode: Literal[0, 1] = Field(default=0, title="Resize Mode", description="Sets the resize mode: 0 to upscale by upscaling_resize amount, 1 to upscale up to upscaling_resize_h x upscaling_resize_w.")
    show_extras_results: bool = Field(default=True, title="Show results", description="Should the backend return the generated image?")
    gfpgan_visibility: float = Field(default=0, title="GFPGAN Visibility", ge=0, le=1, allow_inf_nan=False, description="Sets the visibility of GFPGAN, values should be between 0 and 1.")
    codeformer_visibility: float = Field(default=0, title="CodeFormer Visibility", ge=0, le=1, allow_inf_nan=False, description="Sets the visibility of CodeFormer, values should be between 0 and 1.")
    codeformer_weight: float = Field(default=0, title="CodeFormer Weight", ge=0, le=1, allow_inf_nan=False, description="Sets the weight of CodeFormer, values should be between 0 and 1.")
    upscaling_resize: float = Field(default=2, title="Upscaling Factor", ge=1, le=4, description="By how much to upscale the image, only used when resize_mode=0.")
    upscaling_resize_w: int = Field(default=512, title="Target Width", ge=1, description="Target width for the upscaler to hit. Only used when resize_mode=1.")
    upscaling_resize_h: int = Field(default=512, title="Target Height", ge=1, description="Target height for the upscaler to hit. Only used when resize_mode=1.")
    upscaling_crop: bool = Field(default=True, title="Crop to fit", description="Should the upscaler crop the image to fit in the choosen size?")
    upscaler_1: str = Field(default="None", title="Main upscaler", description=f"The name of the main upscaler to use, it has to be one of this list: {' , '.join([x.name for x in sd_upscalers])}")
    upscaler_2: str = Field(default="None", title="Secondary upscaler", description=f"The name of the secondary upscaler to use, it has to be one of this list: {' , '.join([x.name for x in sd_upscalers])}")
    extras_upscaler_2_visibility: float = Field(default=0, title="Secondary upscaler visibility", ge=0, le=1, allow_inf_nan=False, description="Sets the visibility of secondary upscaler, values should be between 0 and 1.")

class ExtraBaseResponse(BaseModel):
    html_info_x: str
    html_info: str

class ExtrasSingleImageRequest(ExtrasBaseRequest):
    image: str = Field(default="", title="Image", description="Image to work on, must be a Base64 string containing the image's data.")

class ExtrasSingleImageResponse(ExtraBaseResponse):
    image: str = Field(default=None, title="Image", description="The generated image in base64 format.")

class SerializableImage(BaseModel):
  path: str = Field(title="Path", description="The image's path ()")

class ImageItem(BaseModel):
    data: str = Field(title="image data")
    name: str = Field(title="filename")

class ExtrasBatchImagesRequest(ExtrasBaseRequest):
    imageList: list[str] = Field(title="Images", description="List of images to work on. Must be Base64 strings")

class ExtrasBatchImagesResponse(ExtraBaseResponse):
    images: list[str] = Field(title="Images", description="The generated images in base64 format.")