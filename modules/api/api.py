import io
import time
import base64
from io import BytesIO
from typing import List, Dict, Any, Optional
from threading import Lock
from secrets import compare_digest
from fastapi import FastAPI, APIRouter, Depends, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.exceptions import HTTPException
from PIL import PngImagePlugin,Image
import requests
import piexif
import piexif.helper
import gradio as gr
from modules import errors, shared, sd_samplers, deepbooru, sd_hijack, images, scripts, ui, postprocessing, script_callbacks, generation_parameters_copypaste
from modules.sd_vae import vae_dict
from modules.api import models
from modules.processing import StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img, process_images
from modules.textual_inversion.textual_inversion import create_embedding, train_embedding
from modules.textual_inversion.preprocess import preprocess
from modules.hypernetworks.hypernetwork import create_hypernetwork, train_hypernetwork
from modules.sd_models import checkpoints_list, unload_model_weights, reload_model_weights
from modules.sd_models_config import find_checkpoint_config_near_filename
from modules import devices

errors.install()


def upscaler_to_index(name: str):
    try:
        return [x.name.lower() for x in shared.sd_upscalers].index(name.lower())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid upscaler, needs to be one of these: {' , '.join([x.name for x in shared.sd_upscalers])}") from e

def script_name_to_index(name, scripts_list):
    try:
        return [script.title().lower() for script in scripts_list].index(name.lower())
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
        shared.log.warning(f'API cannot decode image: {e}')
        raise HTTPException(status_code=500, detail="Invalid encoded image") from e


def save_image(image, fn, ext):
    # actual save
    parameters = image.info.get('parameters', None)
    image_format = Image.registered_extensions()[f'.{ext}']
    if image_format == 'PNG':
        pnginfo_data = PngImagePlugin.PngInfo()
        for k, v in image.info.items():
            pnginfo_data.add_text(k, str(v))
        image.save(fn, format=image_format, quality=shared.opts.jpeg_quality, pnginfo=pnginfo_data)
    elif image_format == 'JPEG':
        if image.mode == 'RGBA':
            shared.log.warning('Saving RGBA image as JPEG: Alpha channel will be lost')
            image = image.convert("RGB")
        elif image.mode == 'I;16':
            image = image.point(lambda p: p * 0.0038910505836576).convert("L")
        exif_bytes = piexif.dump({ "Exif": { piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(parameters or "", encoding="unicode") } })
        image.save(fn, format=image_format, quality=shared.opts.jpeg_quality, exif=exif_bytes)
    elif image_format == 'WEBP':
        if image.mode == 'I;16':
            image = image.point(lambda p: p * 0.0038910505836576).convert("RGB")
        exif_bytes = piexif.dump({ "Exif": { piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(parameters or "", encoding="unicode") } })
        image.save(fn, format=image_format, quality=shared.opts.jpeg_quality, lossless=shared.opts.webp_lossless, exif=exif_bytes)
    else:
        # shared.log.warning(f'Unrecognized image format: {extension} attempting save as {image_format}')
        image.save(fn, format=image_format, quality=shared.opts.jpeg_quality)


def encode_pil_to_base64(image):
    with io.BytesIO() as output_bytes:
        save_image(image, output_bytes, shared.opts.samples_format)
        bytes_data = output_bytes.getvalue()
    return base64.b64encode(bytes_data)


class Api:
    def __init__(self, app: FastAPI, queue_lock: Lock):
        self.credentials = {}
        if shared.cmd_opts.auth:
            for auth in shared.cmd_opts.auth.split(","):
                user, password = auth.split(":")
                self.credentials[user.replace('"', '').strip()] = password.replace('"', '').strip()
        if shared.cmd_opts.auth_file:
            with open(shared.cmd_opts.auth_file, 'r', encoding="utf8") as file:
                for line in file.readlines():
                    user, password = line.split(":")
                    self.credentials[user.replace('"', '').strip()] = password.replace('"', '').strip()

        self.router = APIRouter()
        self.app = app
        self.queue_lock = queue_lock
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
        self.add_api_route("/sdapi/v1/sd-models", self.get_sd_models, methods=["GET"], response_model=List[models.SDModelItem])
        self.add_api_route("/sdapi/v1/hypernetworks", self.get_hypernetworks, methods=["GET"], response_model=List[models.HypernetworkItem])
        self.add_api_route("/sdapi/v1/face-restorers", self.get_face_restorers, methods=["GET"], response_model=List[models.FaceRestorerItem])
        self.add_api_route("/sdapi/v1/prompt-styles", self.get_prompt_styles, methods=["GET"], response_model=List[models.StyleItem])
        self.add_api_route("/sdapi/v1/embeddings", self.get_embeddings, methods=["GET"], response_model=models.EmbeddingsResponse)
        self.add_api_route("/sdapi/v1/refresh-checkpoints", self.refresh_checkpoints, methods=["POST"])
        self.add_api_route("/sdapi/v1/sd-vae", self.get_sd_vaes, methods=["GET"], response_model=List[models.SDVaeItem])
        self.add_api_route("/sdapi/v1/refresh-vae", self.refresh_vaes, methods=["POST"])
        self.add_api_route("/sdapi/v1/create/embedding", self.create_embedding, methods=["POST"], response_model=models.CreateResponse)
        self.add_api_route("/sdapi/v1/create/hypernetwork", self.create_hypernetwork, methods=["POST"], response_model=models.CreateResponse)
        self.add_api_route("/sdapi/v1/preprocess", self.preprocess, methods=["POST"], response_model=models.PreprocessResponse)
        self.add_api_route("/sdapi/v1/train/embedding", self.train_embedding, methods=["POST"], response_model=models.TrainResponse)
        self.add_api_route("/sdapi/v1/train/hypernetwork", self.train_hypernetwork, methods=["POST"], response_model=models.TrainResponse)
        self.add_api_route("/sdapi/v1/shutdown", self.shutdown, methods=["POST"])
        self.add_api_route("/sdapi/v1/memory", self.get_memory, methods=["GET"], response_model=models.MemoryResponse)
        self.add_api_route("/sdapi/v1/unload-checkpoint", self.unloadapi, methods=["POST"])
        self.add_api_route("/sdapi/v1/reload-checkpoint", self.reloadapi, methods=["POST"])
        self.add_api_route("/sdapi/v1/scripts", self.get_scripts_list, methods=["GET"], response_model=models.ScriptsList)
        self.add_api_route("/sdapi/v1/script-info", self.get_script_info, methods=["GET"], response_model=List[models.ScriptInfo])
        self.add_api_route("/sdapi/v1/extensions", self.get_extensions_list, methods=["GET"], response_model=List[models.ExtensionItem])
        self.add_api_route("/sdapi/v1/log", self.get_log_buffer, methods=["GET"], response_model=List)
        self.add_api_route("/sdapi/v1/start", self.session_start, methods=["GET"])
        self.add_api_route("/sdapi/v1/motd", self.get_motd, methods=["GET"], response_model=str)
        self.add_api_route("/sdapi/v1/extra-networks", self.get_extra_networks, methods=["GET"], response_model=List[models.ExtraNetworkItem])
        self.default_script_arg_txt2img = []
        self.default_script_arg_img2img = []

    def add_api_route(self, path: str, endpoint, **kwargs):
        if (shared.cmd_opts.auth or shared.cmd_opts.auth_file) and shared.cmd_opts.api_only:
            return self.app.add_api_route(path, endpoint, dependencies=[Depends(self.auth)], **kwargs)
        return self.app.add_api_route(path, endpoint, **kwargs)

    def auth(self, credentials: HTTPBasicCredentials = Depends(HTTPBasic())):
        # this is only needed for api-only since otherwise auth is handled in gradio/routes.py
        if credentials.username in self.credentials:
            if compare_digest(credentials.password, self.credentials[credentials.username]):
                return True
        raise HTTPException(status_code=401, detail="Unauthorized", headers={"WWW-Authenticate": "Basic"})

    def get_log_buffer(self, req: models.LogRequest = Depends()):
        lines = shared.log.buffer[:req.lines] if req.lines > 0 else shared.log.buffer.copy()
        if req.clear:
            shared.log.buffer.clear()
        return lines

    def session_start(self, req: Request, agent: Optional[str] = None):
        token = req.cookies.get("access-token") or req.cookies.get("access-token-unsecure")
        user = self.app.tokens.get(token)
        shared.log.info(f'Browser session: user={user} client={req.client.host} agent={agent}')
        return {}

    def get_motd(self):
        from installer import get_version
        motd = ''
        ver = get_version()
        if ver.get('updated', None) is not None:
            motd = f"version <b>{ver['hash']} {ver['updated']}</b> <span style='color: var(--primary-500)'>{ver['url'].split('/')[-1]}</span><br>"
        if shared.opts.motd:
            res = requests.get('https://vladmandic.github.io/automatic/motd', timeout=10)
            if res.status_code == 200:
                msg = (res.text or '').strip()
                shared.log.info(f'MOTD: {msg if len(msg) > 0 else "N/A"}')
                motd += res.text
        return motd

    def get_selectable_script(self, script_name, script_runner):
        if script_name is None or script_name == "":
            return None, None
        script_idx = script_name_to_index(script_name, script_runner.selectable_scripts)
        script = script_runner.selectable_scripts[script_idx]
        return script, script_idx

    def get_scripts_list(self):
        t2ilist = [script.name for script in scripts.scripts_txt2img.scripts if script.name is not None]
        i2ilist = [script.name for script in scripts.scripts_img2img.scripts if script.name is not None]
        control = [script.name for script in scripts.scripts_control.scripts if script.name is not None]
        return models.ScriptsList(txt2img = t2ilist, img2img = i2ilist, control = control)

    def get_script_info(self, script_name: Optional[str] = None):
        res = []
        for script_list in [scripts.scripts_txt2img.scripts, scripts.scripts_img2img.scripts, scripts.scripts_control.scripts]:
            for script in script_list:
                if script.api_info is not None and (script_name is None or script_name == script.api_info.name):
                    res.append(script.api_info)
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
        if gr is None:
            return script_args
        with gr.Blocks(): # will throw errors calling ui function without this
            for script in script_runner.scripts:
                if script.ui(script.is_img2img):
                    ui_default_values = []
                    for elem in script.ui(script.is_img2img):
                        ui_default_values.append(elem.value)
                    script_args[script.args_from:script.args_to] = ui_default_values
        return script_args

    def init_script_args(self, p, request, default_script_args, selectable_scripts, selectable_script_idx, script_runner):
        script_args = default_script_args.copy()
        # position 0 in script_arg is the idx+1 of the selectable script that is going to be run when using scripts.scripts_*2img.run()
        if selectable_scripts:
            script_args[selectable_scripts.args_from:selectable_scripts.args_to] = request.script_args
            script_args[0] = selectable_script_idx + 1
        # Now check for always on scripts
        if request.alwayson_scripts and (len(request.alwayson_scripts) > 0):
            for alwayson_script_name in request.alwayson_scripts.keys():
                alwayson_script = self.get_script(alwayson_script_name, script_runner)
                if alwayson_script is None:
                    raise HTTPException(status_code=422, detail=f"Always on script not found: {alwayson_script_name}")
                if not alwayson_script.alwayson:
                    raise HTTPException(status_code=422, detail=f"Selectable script cannot be in always on params: {alwayson_script_name}")
                if "args" in request.alwayson_scripts[alwayson_script_name]:
                    # min between arg length in scriptrunner and arg length in the request
                    for idx in range(0, min((alwayson_script.args_to - alwayson_script.args_from), len(request.alwayson_scripts[alwayson_script_name]["args"]))):
                        script_args[alwayson_script.args_from + idx] = request.alwayson_scripts[alwayson_script_name]["args"][idx]
                    p.per_script_args[alwayson_script.title()] = request.alwayson_scripts[alwayson_script_name]["args"]
        return script_args

    def prepare_img_gen_request(self, request, img_gen_type: str):
        if hasattr(request, "face_id") and request.face_id and not request.script_name and (not request.alwayson_scripts or "FaceID" not in request.alwayson_scripts.keys()):
            request.script_name = "FaceID"
            request.script_args = [
                request.face_id.model,
                request.face_id.scale,
                request.face_id.image,
                request.face_id.override_sampler,
                request.face_id.rank,
                request.face_id.tokens,
                request.face_id.structure,
                request.face_id.cache_model
            ]
            del request.face_id

        if hasattr(request, "ip_adapter") and request.ip_adapter and request.script_name != "IP Adapter" and (not request.alwayson_scripts or "IP Adapter" not in request.alwayson_scripts.keys()):
            request.alwayson_scripts = {} if request.alwayson_scripts is None else request.alwayson_scripts
            request.alwayson_scripts["IP Adapter"] = {
                "args": [request.ip_adapter.adapter, request.ip_adapter.scale, request.ip_adapter.image]
            }
            del request.ip_adapter

    def sanitize_args(self, args: list):
        for idx in range(0, len(args)):
            if isinstance(args[idx], str) and len(args[idx]) >= 1000:
                args[idx] = f"<str {len(args[idx])}>"

    def sanitize_img_gen_request(self, request, img_gen_type: str):
        if hasattr(request, "alwayson_scripts") and request.alwayson_scripts:
            for script_name in request.alwayson_scripts.keys():
                script_obj = request.alwayson_scripts[script_name]

                if script_obj and "args" in script_obj and script_obj["args"]:
                    self.sanitize_args(script_obj["args"])

        if hasattr(request, "script_args") and request.script_args:
            self.sanitize_args(request.script_args)

    def text2imgapi(self, txt2imgreq: models.StableDiffusionTxt2ImgProcessingAPI):
        self.prepare_img_gen_request(txt2imgreq, "txt2img")

        script_runner = scripts.scripts_txt2img
        if not script_runner.scripts:
            script_runner.initialize_scripts(False)
            ui.create_ui(None)
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
        args.pop('face_id', None)
        args.pop('ip_adapter', None)
        args.pop('alwayson_scripts', None)
        send_images = args.pop('send_images', True)
        args.pop('save_images', None)

        with self.queue_lock:
            p = StableDiffusionProcessingTxt2Img(sd_model=shared.sd_model, **args)
            p.scripts = script_runner
            p.outpath_grids = shared.opts.outdir_grids or shared.opts.outdir_txt2img_grids
            p.outpath_samples = shared.opts.outdir_samples or shared.opts.outdir_txt2img_samples
            shared.state.begin('api-txt2img', api=True)
            script_args = self.init_script_args(p, txt2imgreq, self.default_script_arg_txt2img, selectable_scripts, selectable_script_idx, script_runner)
            if selectable_scripts is not None:
                processed = scripts.scripts_txt2img.run(p, *script_args) # Need to pass args as list here
            else:
                p.script_args = tuple(script_args) # Need to pass args as tuple here
                processed = process_images(p)
            shared.state.end(api=False)

        b64images = list(map(encode_pil_to_base64, processed.images)) if send_images else []
        self.sanitize_img_gen_request(txt2imgreq, "txt2img")
        return models.TextToImageResponse(images=b64images, parameters=vars(txt2imgreq), info=processed.js())

    def img2imgapi(self, img2imgreq: models.StableDiffusionImg2ImgProcessingAPI):
        self.prepare_img_gen_request(img2imgreq, "img2img")

        init_images = img2imgreq.init_images
        if init_images is None:
            raise HTTPException(status_code=404, detail="Init image not found")
        mask = img2imgreq.mask
        if mask:
            mask = decode_base64_to_image(mask)
        script_runner = scripts.scripts_img2img
        if not script_runner.scripts:
            script_runner.initialize_scripts(True)
            ui.create_ui(None)
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
        args.pop('face_id', None)
        args.pop('ip_adapter', None)
        send_images = args.pop('send_images', True)
        args.pop('save_images', None)

        with self.queue_lock:
            p = StableDiffusionProcessingImg2Img(sd_model=shared.sd_model, **args)
            p.init_images = [decode_base64_to_image(x) for x in init_images]
            p.scripts = script_runner
            p.outpath_grids = shared.opts.outdir_img2img_grids
            p.outpath_samples = shared.opts.outdir_img2img_samples
            shared.state.begin('api-img2img', api=True)
            script_args = self.init_script_args(p, img2imgreq, self.default_script_arg_img2img, selectable_scripts, selectable_script_idx, script_runner)
            if selectable_scripts is not None:
                processed = scripts.scripts_img2img.run(p, *script_args) # Need to pass args as list here
            else:
                p.script_args = tuple(script_args) # Need to pass args as tuple here
                processed = process_images(p)
            shared.state.end(api=False)

        b64images = list(map(encode_pil_to_base64, processed.images)) if send_images else []
        if not img2imgreq.include_init_images:
            img2imgreq.init_images = None
            img2imgreq.mask = None
        self.sanitize_img_gen_request(img2imgreq, "img2img")
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
        if not req.image.strip():
            return models.PNGInfoResponse(info="")

        image = decode_base64_to_image(req.image.strip())
        if image is None:
            return models.PNGInfoResponse(info="")

        geninfo, items = images.read_info_from_image(image)
        if geninfo is None:
            geninfo = ""

        if items and items['parameters']:
            del items['parameters']

        params = generation_parameters_copypaste.parse_generation_parameters(geninfo)
        script_callbacks.infotext_pasted_callback(geninfo, params)

        return models.PNGInfoResponse(info=geninfo, items=items, parameters=params)

    def progressapi(self, req: models.ProgressRequest = Depends()):
        if shared.state.job_count == 0:
            return models.ProgressResponse(progress=0, eta_relative=0, state=shared.state.dict(), textinfo=shared.state.textinfo)

        shared.state.do_set_current_image()
        current_image = None
        if shared.state.current_image and not req.skip_current_image:
            current_image = encode_pil_to_base64(shared.state.current_image)

        batch_x = max(shared.state.job_no, 0)
        batch_y = max(shared.state.job_count, 1)
        step_x = max(shared.state.sampling_step, 0)
        step_y = max(shared.state.sampling_steps, 1)
        current = step_y * batch_x + step_x
        total = step_y * batch_y
        progress = current / total if current > 0 and total > 0 else 0
        time_since_start = time.time() - shared.state.time_start
        eta_relative = (time_since_start / progress) - time_since_start if progress > 0 else 0

        res = models.ProgressResponse(progress=progress, eta_relative=eta_relative, state=shared.state.dict(), current_image=current_image, textinfo=shared.state.textinfo)
        return res


    def interrogateapi(self, interrogatereq: models.InterrogateRequest):
        image_b64 = interrogatereq.image
        if image_b64 is None:
            raise HTTPException(status_code=404, detail="Image not found")
        img = decode_base64_to_image(image_b64)
        img = img.convert('RGB')
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
        unload_model_weights(op='model')
        unload_model_weights(op='refiner')
        return {}

    def reloadapi(self):
        reload_model_weights()
        return {}

    def skip(self):
        shared.state.skip()

    def get_config(self):
        options = {}
        for k in shared.opts.data.keys():
            if shared.opts.data_labels.get(k) is not None:
                options.update({k: shared.opts.data.get(k, shared.opts.data_labels.get(k).default)})
            else:
                options.update({k: shared.opts.data.get(k, None)})
        if 'sd_lyco' in options:
            del options['sd_lyco']
        if 'sd_lora' in options:
            del options['sd_lora']
        return options

    def set_config(self, req: Dict[str, Any]):
        updated = []
        for k, v in req.items():
            updated.append({ k: shared.opts.set(k, v) })
        shared.opts.save(shared.config_filename)
        return { "updated": updated }

    def get_cmd_flags(self):
        return vars(shared.cmd_opts)

    def get_samplers(self):
        return [{"name": sampler[0], "aliases":sampler[2], "options":sampler[3]} for sampler in sd_samplers.all_samplers]

    def get_sd_vaes(self):
        return [{"model_name": x, "filename": vae_dict[x]} for x in vae_dict.keys()]

    def get_upscalers(self):
        return [{"name": upscaler.name, "model_name": upscaler.scaler.model_name, "model_path": upscaler.data_path, "model_url": None, "scale": upscaler.scale} for upscaler in shared.sd_upscalers]

    def get_sd_models(self):
        return [{"title": x.title, "model_name": x.name, "filename": x.filename, "type": x.type, "hash": x.shorthash, "sha256": x.sha256, "config": find_checkpoint_config_near_filename(x)} for x in checkpoints_list.values()]

    def get_hypernetworks(self):
        return [{"name": name, "path": shared.hypernetworks[name]} for name in shared.hypernetworks]

    def get_face_restorers(self):
        return [{"name":x.name(), "cmd_dir": getattr(x, "cmd_dir", None)} for x in shared.face_restorers]

    def get_prompt_styles(self):
        return [{ 'name': v.name, 'prompt': v.prompt, 'negative_prompt': v.negative_prompt, 'extra': v.extra, 'filename': v.filename, 'preview': v.preview} for v in shared.prompt_styles.styles.values()]

    def get_embeddings(self):
        db = sd_hijack.model_hijack.embedding_db
        def convert_embedding(embedding):
            return {"step": embedding.step, "sd_checkpoint": embedding.sd_checkpoint, "sd_checkpoint_name": embedding.sd_checkpoint_name, "shape": embedding.shape, "vectors": embedding.vectors}

        def convert_embeddings(embeddings):
            return {embedding.name: convert_embedding(embedding) for embedding in embeddings.values()}

        return {"loaded": convert_embeddings(db.word_embeddings), "skipped": convert_embeddings(db.skipped_embeddings)}

    def get_extra_networks(self, page: Optional[str] = None, name: Optional[str] = None, filename: Optional[str] = None, title: Optional[str] = None, fullname: Optional[str] = None, hash: Optional[str] = None): # pylint: disable=redefined-builtin
        res = []
        for pg in shared.extra_networks:
            if page is not None and pg.name != page.lower():
                continue
            for item in pg.items:
                if name is not None and item.get('name', '') != name:
                    continue
                if title is not None and item.get('title', '') != title:
                    continue
                if filename is not None and item.get('filename', '') != filename:
                    continue
                if fullname is not None and item.get('fullname', '') != fullname:
                    continue
                if hash is not None and (item.get('shorthash', None) or item.get('hash')) != hash:
                    continue
                res.append({
                    'name': item.get('name', ''),
                    'type': pg.name,
                    'title': item.get('title', None),
                    'fullname': item.get('fullname', None),
                    'filename': item.get('filename', None),
                    'hash': item.get('shorthash', None) or item.get('hash'),
                    "preview": item.get('preview', None),
                })
        return res

    def refresh_checkpoints(self):
        return shared.refresh_checkpoints()

    def refresh_vaes(self):
        return shared.refresh_vaes()

    def create_embedding(self, args: dict):
        try:
            shared.state.begin('api-embedding')
            filename = create_embedding(**args) # create empty embedding
            sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings() # reload embeddings so new one can be immediately used
            shared.state.end()
            return models.CreateResponse(info = f"create embedding filename: {filename}")
        except AssertionError as e:
            shared.state.end()
            return models.TrainResponse(info = f"create embedding error: {e}")

    def create_hypernetwork(self, args: dict):
        try:
            shared.state.begin('api-hypernetwork')
            filename = create_hypernetwork(**args) # create empty embedding # pylint: disable=E1111
            shared.state.end()
            return models.CreateResponse(info = f"create hypernetwork filename: {filename}")
        except AssertionError as e:
            shared.state.end()
            return models.TrainResponse(info = f"create hypernetwork error: {e}")

    def preprocess(self, args: dict):
        try:
            shared.state.begin('api-preprocess')
            preprocess(**args) # quick operation unless blip/booru interrogation is enabled
            shared.state.end()
            return models.PreprocessResponse(info = 'preprocess complete')
        except KeyError as e:
            shared.state.end()
            return models.PreprocessResponse(info = f"preprocess error: invalid token: {e}")
        except AssertionError as e:
            shared.state.end()
            return models.PreprocessResponse(info = f"preprocess error: {e}")
        except FileNotFoundError as e:
            shared.state.end()
            return models.PreprocessResponse(info = f'preprocess error: {e}')

    def train_embedding(self, args: dict):
        try:
            shared.state.begin('api-embedding')
            apply_optimizations = False
            error = None
            filename = ''
            if not apply_optimizations:
                sd_hijack.undo_optimizations()
            try:
                _embedding, filename = train_embedding(**args) # can take a long time to complete
            except Exception as e:
                error = e
            finally:
                if not apply_optimizations:
                    sd_hijack.apply_optimizations()
                shared.state.end()
            return models.TrainResponse(info = f"train embedding complete: filename: {filename} error: {error}")
        except AssertionError as msg:
            shared.state.end()
            return models.TrainResponse(info = f"train embedding error: {msg}")

    def train_hypernetwork(self, args: dict):
        try:
            shared.state.begin('api-hypernetwork')
            shared.loaded_hypernetworks = []
            apply_optimizations = False
            error = None
            filename = ''
            if not apply_optimizations:
                sd_hijack.undo_optimizations()
            try:
                _hypernetwork, filename = train_hypernetwork(**args)
            except Exception as e:
                error = e
            finally:
                shared.sd_model.cond_stage_model.to(devices.device)
                shared.sd_model.first_stage_model.to(devices.device)
                if not apply_optimizations:
                    sd_hijack.apply_optimizations()
                shared.state.end()
            return models.TrainResponse(info=f"train embedding complete: filename: {filename} error: {error}")
        except AssertionError:
            shared.state.end()
            return models.TrainResponse(info=f"train embedding error: {error}")

    def shutdown(self):
        shared.log.info('Shutdown request received')
        import sys
        sys.exit(0)

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
                cuda = { 'error': 'unavailable' }
        except Exception as err:
            cuda = { 'error': f'{err}' }
        return models.MemoryResponse(ram = ram, cuda = cuda)

    def get_extensions_list(self):
        from modules import extensions
        extensions.list_extensions()
        ext_list = []
        for ext in extensions.extensions:
            ext: extensions.Extension
            ext.read_info()
            if ext.remote is not None:
                ext_list.append({
                    "name": ext.name,
                    "remote": ext.remote,
                    "branch": ext.branch,
                    "commit_hash":ext.commit_hash,
                    "commit_date":ext.commit_date,
                    "version":ext.version,
                    "enabled":ext.enabled
                })
        return ext_list

    def launch(self):
        config = {
            "listen": shared.cmd_opts.listen,
            "port": shared.cmd_opts.port,
            "keyfile": shared.cmd_opts.tls_keyfile,
            "certfile": shared.cmd_opts.tls_certfile,
            "loop": "auto", # auto, asyncio, uvloop
            "http": "auto", # auto, h11, httptools
        }
        from modules.server import UvicornServer
        server = UvicornServer(self.app, **config)
        # from modules.server import HypercornServer
        # server = HypercornServer(self.app, **config)
        server.start()
        shared.log.info(f'API server: Uvicorn options={config}')
        return server
