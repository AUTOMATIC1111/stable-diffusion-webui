# main.py
# Created on: 2024-01-30 08:47:41
# Author: VuDLe@Tu Nong's server
# Last updated: 2024-01-31 09:52:42
# Last modified by: VuDLe@Tu Nong's server

from celery.result import AsyncResult
from fastapi import Body, FastAPI, Form, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, File, UploadFile, Response, Form
from typing import Optional
from pydantic import BaseModel

from workers import celery_workers
from workers import api
import os


app = FastAPI()


class Diffusion(BaseModel):
    prompt: str
    negative_prompt: str


@app.get("/get_info/")
async def get_info():
    result = {"task_id": 12, "task_result": "LeDuc Vu says yeah"}
    return JSONResponse(result)


@app.post("/txt2image/")
async def txt2image(
    diffusion: Diffusion = Depends(), image_file: UploadFile = File(...)
):
    uploaded_folder = "./uploaded"
    os.makedirs(uploaded_folder, exist_ok=True)
    print(f"file name uploaded is {image_file.filename}")

    uploaded_image_path = os.path.join(
        uploaded_folder, os.path.basename(image_file.filename)
    )
    print(f"file path is {uploaded_image_path}")
    with open(uploaded_image_path, "wb") as f:
        f.write(image_file.file.read())
        message = "received %s succesfully" % (image_file.filename,)
        response = {"message": message}

    data_db = diffusion.dict()

    payload = api.payloader(
        image_file=uploaded_image_path,
        prompt=data_db["prompt"],
        negative_prompt=data_db["negative_prompt"],
    )

    task = celery_workers.txt2image.delay(
        payload, webui_server_url="http://127.0.0.1:7861"
    )
    response = {
        "task_id": task.id,
        "message": "Image generation is running. Please wait",
    }
    return JSONResponse(response)


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    print(f"file name uploaded is {file.filename}")
    return {"filename": file.filename}
