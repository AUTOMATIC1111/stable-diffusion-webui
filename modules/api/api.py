from modules.api.processing import StableDiffusionProcessingAPI
from modules.processing import StableDiffusionProcessingTxt2Img, process_images
from modules.sd_samplers import all_samplers
from modules.extras import run_pnginfo
import modules.shared as shared
import modules.sd_models as sd_models
from modules.sd_models import create_checkpoint_info, save_checkpoint_file_from_url
import uvicorn
from fastapi import Body, APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, Json
import json
import io
import base64


def sampler_to_index(name): return next(filter(
    lambda row: name.lower() == row[1].name.lower(), enumerate(all_samplers)), None)


class TextToImageResponse(BaseModel):
    images: list[str] = Field(default=None, title="Image",
                              description="The generated image in base64 format.")
    parameters: Json
    info: Json


class SaveCheckpointRequest(BaseModel):
    url: str = Field(default=None, title="Checkpoint",
                     description="The checkpoint to save.")


class Api:
    def __init__(self, app, queue_lock):
        self.router = APIRouter()
        self.app = app
        self.queue_lock = queue_lock

        self.app.add_api_route("/sdapi/v1/txt2img", self.text2imgapi, methods=["POST"])
        self.app.add_api_route("/sdapi/v1/save_checkpoint", self.save_checkpoint, methods=["POST"])

    def text2imgapi(self, txt2imgreq: StableDiffusionProcessingAPI):
        sampler_index = sampler_to_index(txt2imgreq.sampler_index)[0]
        checkpoint_filename = txt2imgreq.model_checkpoint
        checkpoint_info = None

        if sampler_index is None:
            raise HTTPException(status_code=404, detail="Sampler not found")

        if checkpoint_filename is not None:
            if checkpoint_filename.startswith("http"):
                save_checkpoint_file_from_url(
                    url=checkpoint_filename
                )
            if checkpoint_info is None:
                checkpoint_info = create_checkpoint_info(
                    url=checkpoint_filename
                )
            print("loaded new model", checkpoint_info)
            sd_models.load_model(checkpoint_info=checkpoint_info)

        populate = txt2imgreq.copy(update={  # Override __init__ params
            "sd_model": shared.sd_model,
            "sampler_index": sampler_index,
            "do_not_save_samples": True,
            "do_not_save_grid": True
        })

        p = StableDiffusionProcessingTxt2Img(**vars(populate))
        with self.queue_lock:
            processed = process_images(p)

        b64images = []
        for i in processed.images:
            buffer = io.BytesIO()
            i.save(buffer, format="png")
            b64images.append(base64.b64encode(buffer.getvalue()))

        return TextToImageResponse(images=b64images, parameters=json.dumps(vars(txt2imgreq)), info=json.dumps(processed.info))

    def save_checkpoint(self, req: SaveCheckpointRequest):
        save_checkpoint_file_from_url(url=req.url)
        return JSONResponse(status_code=200, content={"message": "OK"})

    def img2imgapi(self):
        raise NotImplementedError

    def extrasapi(self):
        raise NotImplementedError

    def pnginfoapi(self):
        raise NotImplementedError

    def launch(self, server_name, port):
        self.app.include_router(self.router)
        uvicorn.run(self.app, host=server_name, port=port)
