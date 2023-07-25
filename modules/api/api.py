import base64
import io
import os
import time
import datetime
import uvicorn
import gradio as gr
from threading import Lock
from io import BytesIO
from fastapi import APIRouter, Depends, FastAPI, Request, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from secrets import compare_digest

import modules.shared as shared
from modules import sd_samplers, deepbooru, sd_hijack, images, scripts, ui, postprocessing, errors, restart
from modules.api import models
from modules.shared import opts
from modules.processing import StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img, process_images
from modules.textual_inversion.textual_inversion import create_embedding, train_embedding
from modules.textual_inversion.preprocess import preprocess
from modules.hypernetworks.hypernetwork import create_hypernetwork, train_hypernetwork
from PIL import PngImagePlugin,Image
from modules.sd_models import checkpoints_list, unload_model_weights, reload_model_weights, checkpoint_aliases
from modules.sd_vae import vae_dict
from modules.sd_models_config import find_checkpoint_config_near_filename
from modules.realesrgan_model import get_realesrgan_models
from modules import devices
from typing import Dict, List, Any
import piexif
import piexif.helper
from contextlib import closing


def script_name_to_index(name, scripts):
    try:
        return [script.title().lower() for script in scripts].index(name.lower())
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Script '{name}' not found") from e


def validate_sampler_name(name):
    config = sd_samplers.all_samplers_map.get(name, None)
    if config is None:
        raise HTTPException(status_code=404, detail="Sampler not found")

    return name


def setUpscalers(req: dict):
    reqDict = vars(req)
    reqDict['extras_upscaler_1'] = reqDict.pop('upscaler_1', None)
    reqDict['extras_upscaler_2'] = reqDict.pop('upscaler_2', None)
    return reqDict


def decode_base64_to_image(encoding):
    if encoding.startswith("data:image/"):
        encoding = encoding.split(";")[1].split(",")[1]
    try:
        image = Image.open(BytesIO(base64.b64decode(encoding)))
        return image
    except Exception as e:
        raise HTTPException(status_code=500, detail="Invalid encoded image") from e


def encode_pil_to_base64(image):
    with io.BytesIO() as output_bytes:

        if opts.samples_format.lower() == 'png':
            use_metadata = False
            metadata = PngImagePlugin.PngInfo()
            for key, value in image.info.items():
                if isinstance(key, str) and isinstance(value, str):
                    metadata.add_text(key, value)
                    use_metadata = True
            image.save(output_bytes, format="PNG", pnginfo=(metadata if use_metadata else None), quality=opts.jpeg_quality)

        elif opts.samples_format.lower() in ("jpg", "jpeg", "webp"):
            if image.mode == "RGBA":
                image = image.convert("RGB")
            parameters = image.info.get('parameters', None)
            exif_bytes = piexif.dump({
                "Exif": { piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(parameters or "", encoding="unicode") }
            })
            if opts.samples_format.lower() in ("jpg", "jpeg"):
                image.save(output_bytes, format="JPEG", exif = exif_bytes, quality=opts.jpeg_quality)
            else:
                image.save(output_bytes, format="WEBP", exif = exif_bytes, quality=opts.jpeg_quality)

        else:
            raise HTTPException(status_code=500, detail="Invalid image format")

        bytes_data = output_bytes.getvalue()

    return base64.b64encode(bytes_data)


def api_middleware(app: FastAPI):
    rich_available = False
    try:
        if os.environ.get('WEBUI_RICH_EXCEPTIONS', None) is not None:
            import anyio  # importing just so it can be placed on silent list
            import starlette  # importing just so it can be placed on silent list
            from rich.console import Console
            console = Console()
            rich_available = True
    except Exception:
        pass

    @app.middleware("http")
    async def log_and_time(req: Request, call_next):
        ts = time.time()
        res: Response = await call_next(req)
        duration = str(round(time.time() - ts, 4))
        res.headers["X-Process-Time"] = duration
        endpoint = req.scope.get('path', 'err')
        if shared.cmd_opts.api_log and endpoint.startswith('/sdapi'):
            print('API {t} {code} {prot}/{ver} {method} {endpoint} {cli} {duration}'.format(
                t=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                code=res.status_code,
                ver=req.scope.get('http_version', '0.0'),
                cli=req.scope.get('client', ('0:0.0.0', 0))[0],
                prot=req.scope.get('scheme', 'err'),
                method=req.scope.get('method', 'err'),
                endpoint=endpoint,
                duration=duration,
            ))
        return res

    def handle_exception(request: Request, e: Exception):
        err = {
            "error": type(e).__name__,
            "detail": vars(e).get('detail', ''),
            "body": vars(e).get('body', ''),
            "errors": str(e),
        }
        if not isinstance(e, HTTPException):  # do not print backtrace on known httpexceptions
            message = f"API error: {request.method}: {request.url} {err}"
            if rich_available:
                print(message)
                console.print_exception(show_locals=True, max_frames=2, extra_lines=1, suppress=[anyio, starlette], word_wrap=False, width=min([console.width, 200]))
            else:
                errors.report(message, exc_info=True)
        return JSONResponse(status_code=vars(e).get('status_code', 500), content=jsonable_encoder(err))

    @app.middleware("http")
    async def exception_handling(request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as e:
            return handle_exception(request, e)

    @app.exception_handler(Exception)
    async def fastapi_exception_handler(request: Request, e: Exception):
        return handle_exception(request, e)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, e: HTTPException):
        return handle_exception(request, e)


class Api:
    def __init__(self, app: FastAPI, queue_lock: Lock):
        if shared.cmd_opts.api_auth:
            self.credentials = {}
            for auth in shared.cmd_opts.api_auth.split(","):
                user, password = auth.split(":")
                self.credentials[user] = password

        self.router = APIRouter()
        self.app = app
        self.queue_lock = queue_lock
        api_middleware(self.app)
        self.add_api_route("/sdapi/v1/txt2img", self.text2imgapi, methods=["POST"], response_model=models.TextToImageResponse)
        self.add_api_route("/sdapi/v1/img2img", self.img2imgapi, methods=["POST"], response_model=models.ImageToImageResponse)
        self.add_api_route("/sdapi/v1/extra-single-image", self.extras_single_image_api, methods=["POST"], response_model=models.ExtrasSingleImageResponse)
        self.add_api_route("/sdapi/v1/extra-batch-images", self.extras_batch_images_api, methods=["POST"], response_model=models.ExtrasBatchImagesResponse)
        self.add_api_route("/sdapi/v1/png-info", self.pnginfoapi, methods=["POST"], response_model=models.PNGInfoResponse)
        self.add_api_route("/sdapi/v1/progress", self.progressapi, methods=["GET"], response_model=models.ProgressResponse)
        self.add_api_route("/sdapi/v1/interrogate", self.interrogateapi, methods=["POST"])
        self.add_api_route("/sdapi/v1/interrupt", self.interruptapi, methods=["POST"])
        self.add_api_route("/sdapi/v1/skip", self.skip, methods=["POST"])
        self.add_api_route("/sdapi/v1/options", self.get_config, methods=["GET"], response_model=models.OptionsModel)
        self.add_api_route("/sdapi/v1/options", self.set_config, methods=["POST"])
        self.add_api_route("/sdapi/v1/cmd-flags", self.get_cmd_flags, methods=["GET"], response_model=models.FlagsModel)
        self.add_api_route("/sdapi/v1/samplers", self.get_samplers, methods=["GET"], response_model=List[models.SamplerItem])
        self.add_api_route("/sdapi/v1/upscalers", self.get_upscalers, methods=["GET"], response_model=List[models.UpscalerItem])
        self.add_api_route("/sdapi/v1/latent-upscale-modes", self.get_latent_upscale_modes, methods=["GET"], response_model=List[models.LatentUpscalerModeItem])
        self.add_api_route("/sdapi/v1/sd-models", self.get_sd_models, methods=["GET"], response_model=List[models.SDModelItem])
        self.add_api_route("/sdapi/v1/sd-vae", self.get_sd_vaes, methods=["GET"], response_model=List[models.SDVaeItem])
        self.add_api_route("/sdapi/v1/hypernetworks", self.get_hypernetworks, methods=["GET"], response_model=List[models.HypernetworkItem])
        self.add_api_route("/sdapi/v1/face-restorers", self.get_face_restorers, methods=["GET"], response_model=List[models.FaceRestorerItem])
        self.add_api_route("/sdapi/v1/realesrgan-models", self.get_realesrgan_models, methods=["GET"], response_model=List[models.RealesrganItem])
        self.add_api_route("/sdapi/v1/prompt-styles", self.get_prompt_styles, methods=["GET"], response_model=List[models.PromptStyleItem])
        self.add_api_route("/sdapi/v1/embeddings", self.get_embeddings, methods=["GET"], response_model=models.EmbeddingsResponse)
        self.add_api_route("/sdapi/v1/refresh-checkpoints", self.refresh_checkpoints, methods=["POST"])
        self.add_api_route("/sdapi/v1/create/embedding", self.create_embedding, methods=["POST"], response_model=models.CreateResponse)
        self.add_api_route("/sdapi/v1/create/hypernetwork", self.create_hypernetwork, methods=["POST"], response_model=models.CreateResponse)
        self.add_api_route("/sdapi/v1/preprocess", self.preprocess, methods=["POST"], response_model=models.PreprocessResponse)
        self.add_api_route("/sdapi/v1/train/embedding", self.train_embedding, methods=["POST"], response_model=models.TrainResponse)
        self.add_api_route("/sdapi/v1/train/hypernetwork", self.train_hypernetwork, methods=["POST"], response_model=models.TrainResponse)
        self.add_api_route("/sdapi/v1/memory", self.get_memory, methods=["GET"], response_model=models.MemoryResponse)
        self.add_api_route("/sdapi/v1/unload-checkpoint", self.unloadapi, methods=["POST"])
        self.add_api_route("/sdapi/v1/reload-checkpoint", self.reloadapi, methods=["POST"])
        self.add_api_route("/sdapi/v1/scripts", self.get_scripts_list, methods=["GET"], response_model=models.ScriptsList)
        self.add_api_route("/sdapi/v1/script-info", self.get_script_info, methods=["GET"], response_model=List[models.ScriptInfo])

        if shared.cmd_opts.api_server_stop:
            self.add_api_route("/sdapi/v1/server-kill", self.kill_webui, methods=["POST"])
            self.add_api_route("/sdapi/v1/server-restart", self.restart_webui, methods=["POST"])
            self.add_api_route("/sdapi/v1/server-stop", self.stop_webui, methods=["POST"])

        self.default_script_arg_txt2img = []
        self.default_script_arg_img2img = []

    def add_api_route(self, path: str, endpoint, **kwargs):
        if shared.cmd_opts.api_auth:
            return self.app.add_api_route(path, endpoint, dependencies=[Depends(self.auth)], **kwargs)
        return self.app.add_api_route(path, endpoint, **kwargs)

    def auth(self, credentials: HTTPBasicCredentials = Depends(HTTPBasic())):
        if credentials.username in self.credentials:
            if compare_digest(credentials.password, self.credentials[credentials.username]):
                return True

        raise HTTPException(status_code=401, detail="Incorrect username or password", headers={"WWW-Authenticate": "Basic"})

    def get_selectable_script(self, script_name, script_runner):
        if script_name is None or script_name == "":
            return None, None

        script_idx = script_name_to_index(script_name, script_runner.selectable_scripts)
        script = script_runner.selectable_scripts[script_idx]
        return script, script_idx

    def get_scripts_list(self):
        t2ilist = [script.name for script in scripts.scripts_txt2img.scripts if script.name is not None]
        i2ilist = [script.name for script in scripts.scripts_img2img.scripts if script.name is not None]

        return models.ScriptsList(txt2img=t2ilist, img2img=i2ilist)

    def get_script_info(self):
        res = []

        for script_list in [scripts.scripts_txt2img.scripts, scripts.scripts_img2img.scripts]:
            res += [script.api_info for script in script_list if script.api_info is not None]

        return res

    def get_script(self, script_name, script_runner):
        if script_name is None or script_name == "":
            return None, None

        script_idx = script_name_to_index(script_name, script_runner.scripts)
        return script_runner.scripts[script_idx]

    def init_default_script_args(self, script_runner):
        #find max idx from the scripts in runner and generate a none array to init script_args
        last_arg_index = 1
        for script in script_runner.scripts:
            if last_arg_index < script.args_to:
                last_arg_index = script.args_to
        # None everywhere except position 0 to initialize script args
        script_args = [None]*last_arg_index
        script_args[0] = 0

        # get default values
        with gr.Blocks(): # will throw errors calling ui function without this
            for script in script_runner.scripts:
                if script.ui(script.is_img2img):
                    ui_default_values = []
                    for elem in script.ui(script.is_img2img):
                        ui_default_values.append(elem.value)
                    script_args[script.args_from:script.args_to] = ui_default_values
        return script_args

    def init_script_args(self, request, default_script_args, selectable_scripts, selectable_idx, script_runner):
        script_args = default_script_args.copy()
        # position 0 in script_arg is the idx+1 of the selectable script that is going to be run when using scripts.scripts_*2img.run()
        if selectable_scripts:
            script_args[selectable_scripts.args_from:selectable_scripts.args_to] = request.script_args
            script_args[0] = selectable_idx + 1

        # Now check for always on scripts
        if request.alwayson_scripts:
            for alwayson_script_name in request.alwayson_scripts.keys():
                alwayson_script = self.get_script(alwayson_script_name, script_runner)
                if alwayson_script is None:
                    raise HTTPException(status_code=422, detail=f"always on script {alwayson_script_name} not found")
                # Selectable script in always on script param check
                if alwayson_script.alwayson is False:
                    raise HTTPException(status_code=422, detail="Cannot have a selectable script in the always on scripts params")
                # always on script with no arg should always run so you don't really need to add them to the requests
                if "args" in request.alwayson_scripts[alwayson_script_name]:
                    # min between arg length in scriptrunner and arg length in the request
                    for idx in range(0, min((alwayson_script.args_to - alwayson_script.args_from), len(request.alwayson_scripts[alwayson_script_name]["args"]))):
                        script_args[alwayson_script.args_from + idx] = request.alwayson_scripts[alwayson_script_name]["args"][idx]
        return script_args

    def text2imgapi(self, txt2imgreq: models.StableDiffusionTxt2ImgProcessingAPI):
        script_runner = scripts.scripts_txt2img
        if not script_runner.scripts:
            script_runner.initialize_scripts(False)
            ui.create_ui()
        if not self.default_script_arg_txt2img:
            self.default_script_arg_txt2img = self.init_default_script_args(script_runner)
        selectable_scripts, selectable_script_idx = self.get_selectable_script(txt2imgreq.script_name, script_runner)

        populate = txt2imgreq.copy(update={  # Override __init__ params
            "sampler_name": validate_sampler_name(txt2imgreq.sampler_name or txt2imgreq.sampler_index),
            "do_not_save_samples": not txt2imgreq.save_images,
            "do_not_save_grid": not txt2imgreq.save_images,
        })
        if populate.sampler_name:
            populate.sampler_index = None  # prevent a warning later on

        args = vars(populate)
        args.pop('script_name', None)
        args.pop('script_args', None) # will refeed them to the pipeline directly after initializing them
        args.pop('alwayson_scripts', None)

        script_args = self.init_script_args(txt2imgreq, self.default_script_arg_txt2img, selectable_scripts, selectable_script_idx, script_runner)

        send_images = args.pop('send_images', True)
        args.pop('save_images', None)

        with self.queue_lock:
            with closing(StableDiffusionProcessingTxt2Img(sd_model=shared.sd_model, **args)) as p:
                p.scripts = script_runner
                p.outpath_grids = opts.outdir_txt2img_grids
                p.outpath_samples = opts.outdir_txt2img_samples

                try:
                    shared.state.begin(job="scripts_txt2img")
                    if selectable_scripts is not None:
                        p.script_args = script_args
                        processed = scripts.scripts_txt2img.run(p, *p.script_args) # Need to pass args as list here
                    else:
                        p.script_args = tuple(script_args) # Need to pass args as tuple here
                        processed = process_images(p)
                finally:
                    shared.state.end()

        b64images = list(map(encode_pil_to_base64, processed.images)) if send_images else []

        return models.TextToImageResponse(images=b64images, parameters=vars(txt2imgreq), info=processed.js())

    def img2imgapi(self, img2imgreq: models.StableDiffusionImg2ImgProcessingAPI):
        init_images = img2imgreq.init_images
        if init_images is None:
            raise HTTPException(status_code=404, detail="Init image not found")

        mask = img2imgreq.mask
        if mask:
            mask = decode_base64_to_image(mask)

        script_runner = scripts.scripts_img2img
        if not script_runner.scripts:
            script_runner.initialize_scripts(True)
            ui.create_ui()
        if not self.default_script_arg_img2img:
            self.default_script_arg_img2img = self.init_default_script_args(script_runner)
        selectable_scripts, selectable_script_idx = self.get_selectable_script(img2imgreq.script_name, script_runner)

        populate = img2imgreq.copy(update={  # Override __init__ params
            "sampler_name": validate_sampler_name(img2imgreq.sampler_name or img2imgreq.sampler_index),
            "do_not_save_samples": not img2imgreq.save_images,
            "do_not_save_grid": not img2imgreq.save_images,
            "mask": mask,
        })
        if populate.sampler_name:
            populate.sampler_index = None  # prevent a warning later on

        args = vars(populate)
        args.pop('include_init_images', None)  # this is meant to be done by "exclude": True in model, but it's for a reason that I cannot determine.
        args.pop('script_name', None)
        args.pop('script_args', None)  # will refeed them to the pipeline directly after initializing them
        args.pop('alwayson_scripts', None)

        script_args = self.init_script_args(img2imgreq, self.default_script_arg_img2img, selectable_scripts, selectable_script_idx, script_runner)

        send_images = args.pop('send_images', True)
        args.pop('save_images', None)

        with self.queue_lock:
            with closing(StableDiffusionProcessingImg2Img(sd_model=shared.sd_model, **args)) as p:
                p.init_images = [decode_base64_to_image(x) for x in init_images]
                p.scripts = script_runner
                p.outpath_grids = opts.outdir_img2img_grids
                p.outpath_samples = opts.outdir_img2img_samples

                try:
                    shared.state.begin(job="scripts_img2img")
                    if selectable_scripts is not None:
                        p.script_args = script_args
                        processed = scripts.scripts_img2img.run(p, *p.script_args) # Need to pass args as list here
                    else:
                        p.script_args = tuple(script_args) # Need to pass args as tuple here
                        processed = process_images(p)
                finally:
                    shared.state.end()

        b64images = list(map(encode_pil_to_base64, processed.images)) if send_images else []

        if not img2imgreq.include_init_images:
            img2imgreq.init_images = None
            img2imgreq.mask = None

        return models.ImageToImageResponse(images=b64images, parameters=vars(img2imgreq), info=processed.js())

    def extras_single_image_api(self, req: models.ExtrasSingleImageRequest):
        reqDict = setUpscalers(req)

        reqDict['image'] = decode_base64_to_image(reqDict['image'])

        with self.queue_lock:
            result = postprocessing.run_extras(extras_mode=0, image_folder="", input_dir="", output_dir="", save_output=False, **reqDict)

        return models.ExtrasSingleImageResponse(image=encode_pil_to_base64(result[0][0]), html_info=result[1])

    def extras_batch_images_api(self, req: models.ExtrasBatchImagesRequest):
        reqDict = setUpscalers(req)

        image_list = reqDict.pop('imageList', [])
        image_folder = [decode_base64_to_image(x.data) for x in image_list]

        with self.queue_lock:
            result = postprocessing.run_extras(extras_mode=1, image_folder=image_folder, image="", input_dir="", output_dir="", save_output=False, **reqDict)

        return models.ExtrasBatchImagesResponse(images=list(map(encode_pil_to_base64, result[0])), html_info=result[1])

    def pnginfoapi(self, req: models.PNGInfoRequest):
        if(not req.image.strip()):
            return models.PNGInfoResponse(info="")

        image = decode_base64_to_image(req.image.strip())
        if image is None:
            return models.PNGInfoResponse(info="")

        geninfo, items = images.read_info_from_image(image)
        if geninfo is None:
            geninfo = ""

        items = {**{'parameters': geninfo}, **items}

        return models.PNGInfoResponse(info=geninfo, items=items)

    def progressapi(self, req: models.ProgressRequest = Depends()):
        # copy from check_progress_call of ui.py

        if shared.state.job_count == 0:
            return models.ProgressResponse(progress=0, eta_relative=0, state=shared.state.dict(), textinfo=shared.state.textinfo)

        # avoid dividing zero
        progress = 0.01

        if shared.state.job_count > 0:
            progress += shared.state.job_no / shared.state.job_count
        if shared.state.sampling_steps > 0:
            progress += 1 / shared.state.job_count * shared.state.sampling_step / shared.state.sampling_steps

        time_since_start = time.time() - shared.state.time_start
        eta = (time_since_start/progress)
        eta_relative = eta-time_since_start

        progress = min(progress, 1)

        shared.state.set_current_image()

        current_image = None
        if shared.state.current_image and not req.skip_current_image:
            current_image = encode_pil_to_base64(shared.state.current_image)

        return models.ProgressResponse(progress=progress, eta_relative=eta_relative, state=shared.state.dict(), current_image=current_image, textinfo=shared.state.textinfo)

    def interrogateapi(self, interrogatereq: models.InterrogateRequest):
        image_b64 = interrogatereq.image
        if image_b64 is None:
            raise HTTPException(status_code=404, detail="Image not found")

        img = decode_base64_to_image(image_b64)
        img = img.convert('RGB')

        # Override object param
        with self.queue_lock:
            if interrogatereq.model == "clip":
                processed = shared.interrogator.interrogate(img)
            elif interrogatereq.model == "deepdanbooru":
                processed = deepbooru.model.tag(img)
            else:
                raise HTTPException(status_code=404, detail="Model not found")

        return models.InterrogateResponse(caption=processed)

    def interruptapi(self):
        shared.state.interrupt()

        return {}

    def unloadapi(self):
        unload_model_weights()

        return {}

    def reloadapi(self):
        reload_model_weights()

        return {}

    def skip(self):
        shared.state.skip()

    def get_config(self):
        options = {}
        for key in shared.opts.data.keys():
            metadata = shared.opts.data_labels.get(key)
            if(metadata is not None):
                options.update({key: shared.opts.data.get(key, shared.opts.data_labels.get(key).default)})
            else:
                options.update({key: shared.opts.data.get(key, None)})

        return options

    def set_config(self, req: Dict[str, Any]):
        checkpoint_name = req.get("sd_model_checkpoint", None)
        if checkpoint_name is not None and checkpoint_name not in checkpoint_aliases:
            raise RuntimeError(f"model {checkpoint_name!r} not found")

        for k, v in req.items():
            shared.opts.set(k, v)

        shared.opts.save(shared.config_filename)
        return

    def get_cmd_flags(self):
        return vars(shared.cmd_opts)

    def get_samplers(self):
        return [{"name": sampler[0], "aliases":sampler[2], "options":sampler[3]} for sampler in sd_samplers.all_samplers]

    def get_upscalers(self):
        return [
            {
                "name": upscaler.name,
                "model_name": upscaler.scaler.model_name,
                "model_path": upscaler.data_path,
                "model_url": None,
                "scale": upscaler.scale,
            }
            for upscaler in shared.sd_upscalers
        ]

    def get_latent_upscale_modes(self):
        return [
            {
                "name": upscale_mode,
            }
            for upscale_mode in [*(shared.latent_upscale_modes or {})]
        ]

    def get_sd_models(self):
        return [{"title": x.title, "model_name": x.model_name, "hash": x.shorthash, "sha256": x.sha256, "filename": x.filename, "config": find_checkpoint_config_near_filename(x)} for x in checkpoints_list.values()]

    def get_sd_vaes(self):
        return [{"model_name": x, "filename": vae_dict[x]} for x in vae_dict.keys()]

    def get_hypernetworks(self):
        return [{"name": name, "path": shared.hypernetworks[name]} for name in shared.hypernetworks]

    def get_face_restorers(self):
        return [{"name":x.name(), "cmd_dir": getattr(x, "cmd_dir", None)} for x in shared.face_restorers]

    def get_realesrgan_models(self):
        return [{"name":x.name,"path":x.data_path, "scale":x.scale} for x in get_realesrgan_models(None)]

    def get_prompt_styles(self):
        styleList = []
        for k in shared.prompt_styles.styles:
            style = shared.prompt_styles.styles[k]
            styleList.append({"name":style[0], "prompt": style[1], "negative_prompt": style[2]})

        return styleList

    def get_embeddings(self):
        db = sd_hijack.model_hijack.embedding_db

        def convert_embedding(embedding):
            return {
                "step": embedding.step,
                "sd_checkpoint": embedding.sd_checkpoint,
                "sd_checkpoint_name": embedding.sd_checkpoint_name,
                "shape": embedding.shape,
                "vectors": embedding.vectors,
            }

        def convert_embeddings(embeddings):
            return {embedding.name: convert_embedding(embedding) for embedding in embeddings.values()}

        return {
            "loaded": convert_embeddings(db.word_embeddings),
            "skipped": convert_embeddings(db.skipped_embeddings),
        }

    def refresh_checkpoints(self):
        with self.queue_lock:
            shared.refresh_checkpoints()

    def create_embedding(self, args: dict):
        try:
            shared.state.begin(job="create_embedding")
            filename = create_embedding(**args) # create empty embedding
            sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings() # reload embeddings so new one can be immediately used
            return models.CreateResponse(info=f"create embedding filename: {filename}")
        except AssertionError as e:
            return models.TrainResponse(info=f"create embedding error: {e}")
        finally:
            shared.state.end()


    def create_hypernetwork(self, args: dict):
        try:
            shared.state.begin(job="create_hypernetwork")
            filename = create_hypernetwork(**args) # create empty embedding
            return models.CreateResponse(info=f"create hypernetwork filename: {filename}")
        except AssertionError as e:
            return models.TrainResponse(info=f"create hypernetwork error: {e}")
        finally:
            shared.state.end()

    def preprocess(self, args: dict):
        try:
            shared.state.begin(job="preprocess")
            preprocess(**args) # quick operation unless blip/booru interrogation is enabled
            shared.state.end()
            return models.PreprocessResponse(info='preprocess complete')
        except KeyError as e:
            return models.PreprocessResponse(info=f"preprocess error: invalid token: {e}")
        except Exception as e:
            return models.PreprocessResponse(info=f"preprocess error: {e}")
        finally:
            shared.state.end()

    def train_embedding(self, args: dict):
        try:
            shared.state.begin(job="train_embedding")
            apply_optimizations = shared.opts.training_xattention_optimizations
            error = None
            filename = ''
            if not apply_optimizations:
                sd_hijack.undo_optimizations()
            try:
                embedding, filename = train_embedding(**args) # can take a long time to complete
            except Exception as e:
                error = e
            finally:
                if not apply_optimizations:
                    sd_hijack.apply_optimizations()
            return models.TrainResponse(info=f"train embedding complete: filename: {filename} error: {error}")
        except Exception as msg:
            return models.TrainResponse(info=f"train embedding error: {msg}")
        finally:
            shared.state.end()

    def train_hypernetwork(self, args: dict):
        try:
            shared.state.begin(job="train_hypernetwork")
            shared.loaded_hypernetworks = []
            apply_optimizations = shared.opts.training_xattention_optimizations
            error = None
            filename = ''
            if not apply_optimizations:
                sd_hijack.undo_optimizations()
            try:
                hypernetwork, filename = train_hypernetwork(**args)
            except Exception as e:
                error = e
            finally:
                shared.sd_model.cond_stage_model.to(devices.device)
                shared.sd_model.first_stage_model.to(devices.device)
                if not apply_optimizations:
                    sd_hijack.apply_optimizations()
                shared.state.end()
            return models.TrainResponse(info=f"train embedding complete: filename: {filename} error: {error}")
        except Exception as exc:
            return models.TrainResponse(info=f"train embedding error: {exc}")
        finally:
            shared.state.end()

    def get_memory(self):
        try:
            import os
            import psutil
            process = psutil.Process(os.getpid())
            res = process.memory_info() # only rss is cross-platform guaranteed so we dont rely on other values
            ram_total = 100 * res.rss / process.memory_percent() # and total memory is calculated as actual value is not cross-platform safe
            ram = { 'free': ram_total - res.rss, 'used': res.rss, 'total': ram_total }
        except Exception as err:
            ram = { 'error': f'{err}' }
        try:
            import torch
            if torch.cuda.is_available():
                s = torch.cuda.mem_get_info()
                system = { 'free': s[0], 'used': s[1] - s[0], 'total': s[1] }
                s = dict(torch.cuda.memory_stats(shared.device))
                allocated = { 'current': s['allocated_bytes.all.current'], 'peak': s['allocated_bytes.all.peak'] }
                reserved = { 'current': s['reserved_bytes.all.current'], 'peak': s['reserved_bytes.all.peak'] }
                active = { 'current': s['active_bytes.all.current'], 'peak': s['active_bytes.all.peak'] }
                inactive = { 'current': s['inactive_split_bytes.all.current'], 'peak': s['inactive_split_bytes.all.peak'] }
                warnings = { 'retries': s['num_alloc_retries'], 'oom': s['num_ooms'] }
                cuda = {
                    'system': system,
                    'active': active,
                    'allocated': allocated,
                    'reserved': reserved,
                    'inactive': inactive,
                    'events': warnings,
                }
            else:
                cuda = {'error': 'unavailable'}
        except Exception as err:
            cuda = {'error': f'{err}'}
        return models.MemoryResponse(ram=ram, cuda=cuda)

    def launch(self, server_name, port, root_path):
        self.app.include_router(self.router)
        uvicorn.run(self.app, host=server_name, port=port, timeout_keep_alive=shared.cmd_opts.timeout_keep_alive, root_path=root_path)

    def kill_webui(self):
        restart.stop_program()

    def restart_webui(self):
        if restart.is_restartable():
            restart.restart_program()
        return Response(status_code=501)

    def stop_webui(request):
        shared.state.server_command = "stop"
        return Response("Stopping.")

