from modules.api.processing import StableDiffusionProcessingAPI
from modules.processing import StableDiffusionProcessingTxt2Img, process_images
from modules.sd_samplers import all_samplers
import modules.shared as shared
import uvicorn
from fastapi import APIRouter, HTTPException
import json
import io
import base64
from modules.api.models import *

sampler_to_index = lambda name: next(filter(lambda row: name.lower() == row[1].name.lower(), enumerate(all_samplers)), None)

def img_to_base64(img):
    buffer = io.BytesIO()
    img.save(buffer, format="png")
    return base64.b64encode(buffer.getvalue())

class Api:
    def __init__(self, app, queue_lock):
        self.router = APIRouter()
        self.app = app
        self.queue_lock = queue_lock
        self.app.add_api_route("/sdapi/v1/txt2img", self.text2imgapi, methods=["POST"])

    def text2imgapi(self, txt2imgreq: StableDiffusionProcessingAPI ):
        sampler_index = sampler_to_index(txt2imgreq.sampler_index)
        
        if sampler_index is None:
            raise HTTPException(status_code=404, detail="Sampler not found") 
        
        populate = txt2imgreq.copy(update={ # Override __init__ params
            "sd_model": shared.sd_model, 
            "sampler_index": sampler_index[0],
            "do_not_save_samples": True,
            "do_not_save_grid": True
            }
        )
        p = StableDiffusionProcessingTxt2Img(**vars(populate))
        # Override object param
        with self.queue_lock:
            processed = process_images(p)
        
        b64images = list(map(img_to_base64, processed.images))

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
