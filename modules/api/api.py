from modules.api.processing import StableDiffusionProcessingAPI
from modules.processing import StableDiffusionProcessingTxt2Img, process_images
import modules.shared as shared
import uvicorn
from fastapi import Body, APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, Json
import json
import io
import base64

class TextToImageResponse(BaseModel):
    images: list[str] = Field(default=None, title="Image", description="The generated image in base64 format.")
    parameters: Json
    info: Json


class Api:
    def __init__(self, app):
        self.router = APIRouter()
        app.add_api_route("/sdapi/v1/txt2img", self.text2imgapi, methods=["POST"])

    def text2imgapi(self, txt2imgreq: StableDiffusionProcessingAPI ):
        populate = txt2imgreq.copy(update={ # Override __init__ params
            "sd_model": shared.sd_model, 
            "sampler_index": 0,
            "do_not_save_samples": True,
            "do_not_save_grid": True
            }
        )
        p = StableDiffusionProcessingTxt2Img(**vars(populate))
        # Override object param
        processed = process_images(p)
        
        b64images = []
        for i in processed.images:
            buffer = io.BytesIO()
            i.save(buffer, format="png")
            b64images.append(base64.b64encode(buffer.getvalue()))

        return TextToImageResponse(images=b64images, parameters=json.dumps(vars(txt2imgreq)), info=json.dumps(processed.info))
        
        

    def img2imgendoint(self):
        raise NotImplementedError

    def extrasendoint(self):
        raise NotImplementedError

    def pnginfoendoint(self):
        raise NotImplementedError

    def launch(self, server_name, port):
        app.include_router(self.router)
        uvicorn.run(app, host=server_name, port=port)
