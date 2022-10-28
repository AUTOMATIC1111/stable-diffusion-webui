from modules.api.models import StableDiffusionTxt2ImgProcessingAPI, StableDiffusionImg2ImgProcessingAPI
from modules.processing import StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img, process_images
from modules.sd_samplers import all_samplers
from modules.extras import run_pnginfo
import modules.shared as shared
import uvicorn
from fastapi import Body, APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, Json
from typing import List
import json
import io
import base64
from PIL import Image

sampler_to_index = lambda name: next(filter(lambda row: name.lower() == row[1].name.lower(), enumerate(all_samplers)), None)

class TextToImageResponse(BaseModel):
    images: List[str] = Field(default=None, title="Image", description="The generated image in base64 format.")
    parameters: Json
    info: Json

class ImageToImageResponse(BaseModel):
    images: List[str] = Field(default=None, title="Image", description="The generated image in base64 format.")
    parameters: Json
    info: Json


class Api:
    def __init__(self, app, queue_lock):
        self.router = APIRouter()
        self.app = app
        self.queue_lock = queue_lock
        self.app.add_api_route("/sdapi/v1/txt2img", self.text2imgapi, methods=["POST"])
        self.app.add_api_route("/sdapi/v1/img2img", self.img2imgapi, methods=["POST"])

    def __base64_to_image(self, base64_string):
        # if has a comma, deal with prefix
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
        imgdata = base64.b64decode(base64_string)
        # convert base64 to PIL image
        return Image.open(io.BytesIO(imgdata))

    def text2imgapi(self, txt2imgreq: StableDiffusionTxt2ImgProcessingAPI):
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
        
        b64images = []
        for i in processed.images:
            buffer = io.BytesIO()
            i.save(buffer, format="png")
            b64images.append(base64.b64encode(buffer.getvalue()))

        return TextToImageResponse(images=b64images, parameters=json.dumps(vars(txt2imgreq)), info=processed.js())
        
        

    def img2imgapi(self, img2imgreq: StableDiffusionImg2ImgProcessingAPI):
        sampler_index = sampler_to_index(img2imgreq.sampler_index)
        
        if sampler_index is None:
            raise HTTPException(status_code=404, detail="Sampler not found") 


        init_images = img2imgreq.init_images
        if init_images is None:
            raise HTTPException(status_code=404, detail="Init image not found") 

        mask = img2imgreq.mask
        if mask:
            mask = self.__base64_to_image(mask)

        
        populate = img2imgreq.copy(update={ # Override __init__ params
            "sd_model": shared.sd_model, 
            "sampler_index": sampler_index[0],
            "do_not_save_samples": True,
            "do_not_save_grid": True, 
            "mask": mask
            }
        )
        p = StableDiffusionProcessingImg2Img(**vars(populate))

        imgs = []
        for img in init_images:
            img = self.__base64_to_image(img)
            imgs = [img] * p.batch_size

        p.init_images = imgs
        # Override object param
        with self.queue_lock:
            processed = process_images(p)
        
        b64images = []
        for i in processed.images:
            buffer = io.BytesIO()
            i.save(buffer, format="png")
            b64images.append(base64.b64encode(buffer.getvalue()))

        if (not img2imgreq.include_init_images):
            img2imgreq.init_images = None
            img2imgreq.mask = None

        return ImageToImageResponse(images=b64images, parameters=json.dumps(vars(img2imgreq)), info=processed.js())

    def extrasapi(self):
        raise NotImplementedError

    def pnginfoapi(self):
        raise NotImplementedError

    def launch(self, server_name, port):
        self.app.include_router(self.router)
        uvicorn.run(self.app, host=server_name, port=port)
