import uvicorn
from fastapi import FastAPI, Body, APIRouter
from pydantic import BaseModel, Field
import json
import io
import base64


class TextToImage(BaseModel):
    prompt: str = Field(..., title="Prompt Text", description="The text to generate an image from.")
    negative_prompt: str = Field(default="", title="Negative Prompt Text")
    prompt_style: str = Field(default="None", title="Prompt Style")
    prompt_style2: str = Field(default="None", title="Prompt Style 2")
    steps: int = Field(default=20, title="Steps")
    sampler_index: int = Field(0, title="Sampler Index")
    restore_faces: bool = Field(default=False, title="Restore Faces")
    tiling: bool = Field(default=False, title="Tiling")
    n_iter: int = Field(default=1, title="N Iter")
    batch_size: int = Field(default=1, title="Batch Size")
    cfg_scale: float = Field(default=7, title="Config Scale")
    seed: int = Field(default=-1.0, title="Seed")
    subseed: int = Field(default=-1.0, title="Subseed")
    subseed_strength: float = Field(default=0, title="Subseed Strength")
    seed_resize_from_h: int = Field(default=0, title="Seed Resize From Height")
    seed_resize_from_w: int = Field(default=0, title="Seed Resize From Width")
    height: int = Field(default=512, title="Height")
    width: int = Field(default=512, title="Width")
    enable_hr: bool = Field(default=False, title="Enable HR")
    scale_latent: bool = Field(default=True, title="Scale Latent")
    denoising_strength: float = Field(default=0.7, title="Denoising Strength")


class TextToImageResponse(BaseModel):
    images: list[str] = Field(default=None, title="Image", description="The generated image in base64 format.")
    all_prompts: list[str] = Field(default=None, title="All Prompts", description="The prompt text.")
    negative_prompt: str = Field(default=None, title="Negative Prompt Text")
    seed: int = Field(default=None, title="Seed")
    all_seeds: list[int] = Field(default=None, title="All Seeds")
    subseed: int = Field(default=None, title="Subseed")
    all_subseeds: list[int] = Field(default=None, title="All Subseeds")
    subseed_strength: float = Field(default=None, title="Subseed Strength")
    width: int = Field(default=None, title="Width")
    height: int = Field(default=None, title="Height")
    sampler_index: int = Field(default=None, title="Sampler Index")
    sampler: str = Field(default=None, title="Sampler")
    cfg_scale: float = Field(default=None, title="Config Scale")
    steps: int = Field(default=None, title="Steps")
    batch_size: int = Field(default=None, title="Batch Size")
    restore_faces: bool = Field(default=None, title="Restore Faces")
    face_restoration_model: str = Field(default=None, title="Face Restoration Model")
    sd_model_hash: str = Field(default=None, title="SD Model Hash")
    seed_resize_from_w: int = Field(default=None, title="Seed Resize From Width")
    seed_resize_from_h: int = Field(default=None, title="Seed Resize From Height")
    denoising_strength: float = Field(default=None, title="Denoising Strength")
    extra_generation_params: dict = Field(default={}, title="Extra Generation Params")
    index_of_first_image: int = Field(default=None, title="Index of First Image")
    html: str = Field(default=None, title="HTML")


app = FastAPI()


class Api:
    def __init__(self, txt2img, img2img, run_extras, run_pnginfo):
        self.txt2img = txt2img
        self.img2img = img2img
        self.run_extras = run_extras
        self.run_pnginfo = run_pnginfo

        self.router = APIRouter()
        app.add_api_route("/v1/txt2img", self.txt2imgendoint, response_model=TextToImageResponse)
        # app.add_api_route("/v1/img2img", self.img2imgendoint)
        # app.add_api_route("/v1/extras", self.extrasendoint)
        # app.add_api_route("/v1/pnginfo", self.pnginfoendoint)

    def txt2imgendoint(self, txt2imgreq: TextToImage = Body(embed=True)):
        images, params, html = self.txt2img(*[v for v in txt2imgreq.dict().values()], 0, False, None, '', False, 1, '', 4, '', True)
        b64images = []
        for i in images:
            buffer = io.BytesIO()
            i.save(buffer, format="png")
            b64images.append(base64.b64encode(buffer.getvalue()))
        resp_params = json.loads(params)

        return TextToImageResponse(images=b64images, **resp_params, html=html)

    def img2imgendoint(self):
        raise NotImplementedError

    def extrasendoint(self):
        raise NotImplementedError

    def pnginfoendoint(self):
        raise NotImplementedError

    def launch(self, server_name, port):
        app.include_router(self.router)
        uvicorn.run(app, host=server_name, port=port)
