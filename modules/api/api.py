from modules.api.processing import StableDiffusionProcessingAPI
from modules.processing import StableDiffusionProcessingTxt2Img, process_images
from modules.sd_samplers import all_samplers
from modules.extras import run_pnginfo
import modules.shared as shared
import modules.sd_models as sd_models
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


class Api:
    def __init__(self, app, queue_lock):
        self.router = APIRouter()
        self.app = app
        self.queue_lock = queue_lock
        self.app.add_api_route("/sdapi/v1/txt2img",
                               self.text2imgapi, methods=["POST"])

    def text2imgapi(self, txt2imgreq: StableDiffusionProcessingAPI):
        sampler_index = sampler_to_index(txt2imgreq.sampler_index)[0]

        if sampler_index is None:
            raise HTTPException(status_code=404, detail="Sampler not found")

        if txt2imgreq.model_checkpoint is not None:
            if txt2imgreq.model_checkpoint.startswith("http"):
                print("downloading model", txt2imgreq.model_checkpoint)
                sd_models.save_checkpoint_file_from_url(
                    url=txt2imgreq.model_checkpoint
                )
            checkpoint_info = sd_models.create_checkpoint_info(
                url=txt2imgreq.model_checkpoint
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
        # Override object param
        with self.queue_lock:
            processed = process_images(p)

        b64images = []
        for i in processed.images:
            buffer = io.BytesIO()
            i.save(buffer, format="png")
            b64images.append(base64.b64encode(buffer.getvalue()))

        return TextToImageResponse(images=b64images, parameters=json.dumps(vars(txt2imgreq)), info=json.dumps(processed.info))

    def img2imgapi(self):
        raise NotImplementedError

    def extrasapi(self):
        raise NotImplementedError

    def pnginfoapi(self):
        raise NotImplementedError

    def launch(self, server_name, port):
        self.app.include_router(self.router)
        uvicorn.run(self.app, host=server_name, port=port)
