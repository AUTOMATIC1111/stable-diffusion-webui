from typing import List, Optional
from threading import Lock
from secrets import compare_digest
from fastapi import FastAPI, APIRouter, Depends, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.exceptions import HTTPException
from modules import errors, shared, sd_samplers, scripts, ui, postprocessing
from modules.api import models, endpoints, script, train, helpers, server, nvml
from modules.processing import StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img, process_images


errors.install()
decode_base64_to_image = helpers.decode_base64_to_image
encode_pil_to_base64 = helpers.encode_pil_to_base64


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

        # server api
        self.add_api_route("/sdapi/v1/motd", server.get_motd, methods=["GET"], response_model=str)
        self.add_api_route("/sdapi/v1/log", server.get_log_buffer, methods=["GET"], response_model=List[str])
        self.add_api_route("/sdapi/v1/start", self.get_session_start, methods=["GET"])
        self.add_api_route("/sdapi/v1/progress", server.get_progress, methods=["GET"], response_model=models.ResProgress)
        self.add_api_route("/sdapi/v1/interrupt", server.post_interrupt, methods=["POST"])
        self.add_api_route("/sdapi/v1/skip", server.post_skip, methods=["POST"])
        self.add_api_route("/sdapi/v1/shutdown", server.post_shutdown, methods=["POST"])
        self.add_api_route("/sdapi/v1/memory", server.get_memory, methods=["GET"], response_model=models.ResMemory)
        self.add_api_route("/sdapi/v1/options", server.get_config, methods=["GET"], response_model=models.OptionsModel)
        self.add_api_route("/sdapi/v1/options", server.set_config, methods=["POST"])
        self.add_api_route("/sdapi/v1/cmd-flags", server.get_cmd_flags, methods=["GET"], response_model=models.FlagsModel)
        app.add_api_route("/sdapi/v1/nvml", nvml.get_nvml, methods=["GET"], response_model=List[models.ResNVML])


        # core api using locking
        self.add_api_route("/sdapi/v1/txt2img", self.post_text2img, methods=["POST"], response_model=models.ResTxt2Img)
        self.add_api_route("/sdapi/v1/img2img", self.post_img2img, methods=["POST"], response_model=models.ResImg2Img)
        self.add_api_route("/sdapi/v1/extra-single-image", self.extras_single_image_api, methods=["POST"], response_model=models.ResProcessImage)
        self.add_api_route("/sdapi/v1/extra-batch-images", self.extras_batch_images_api, methods=["POST"], response_model=models.ResProcessBatch)

        # api dealing with optional scripts
        self.add_api_route("/sdapi/v1/scripts", script.get_scripts_list, methods=["GET"], response_model=models.ResScripts)
        self.add_api_route("/sdapi/v1/script-info", script.get_script_info, methods=["GET"], response_model=List[models.ItemScript])

        # enumerator api
        self.add_api_route("/sdapi/v1/interrogate", endpoints.get_interrogate, methods=["GET"], response_model=List[str])
        self.add_api_route("/sdapi/v1/samplers", endpoints.get_samplers, methods=["GET"], response_model=List[models.ItemSampler])
        self.add_api_route("/sdapi/v1/upscalers", endpoints.get_upscalers, methods=["GET"], response_model=List[models.ItemUpscaler])
        self.add_api_route("/sdapi/v1/sd-models", endpoints.get_sd_models, methods=["GET"], response_model=List[models.ItemModel])
        self.add_api_route("/sdapi/v1/hypernetworks", endpoints.get_hypernetworks, methods=["GET"], response_model=List[models.ItemHypernetwork])
        self.add_api_route("/sdapi/v1/face-restorers", endpoints.get_face_restorers, methods=["GET"], response_model=List[models.ItemFaceRestorer])
        self.add_api_route("/sdapi/v1/prompt-styles", endpoints.get_prompt_styles, methods=["GET"], response_model=List[models.ItemStyle])
        self.add_api_route("/sdapi/v1/embeddings", endpoints.get_embeddings, methods=["GET"], response_model=models.ResEmbeddings)
        self.add_api_route("/sdapi/v1/sd-vae", endpoints.get_sd_vaes, methods=["GET"], response_model=List[models.ItemVae])
        self.add_api_route("/sdapi/v1/extensions", endpoints.get_extensions_list, methods=["GET"], response_model=List[models.ItemExtension])
        self.add_api_route("/sdapi/v1/extra-networks", endpoints.get_extra_networks, methods=["GET"], response_model=List[models.ItemExtraNetwork])

        # functional api
        self.add_api_route("/sdapi/v1/png-info", endpoints.post_pnginfo, methods=["POST"], response_model=models.ResImageInfo)
        self.add_api_route("/sdapi/v1/interrogate", endpoints.post_interrogate, methods=["POST"])
        self.add_api_route("/sdapi/v1/refresh-checkpoints", endpoints.post_refresh_checkpoints, methods=["POST"])
        self.add_api_route("/sdapi/v1/unload-checkpoint", endpoints.post_unload_checkpoint, methods=["POST"])
        self.add_api_route("/sdapi/v1/reload-checkpoint", endpoints.post_reload_checkpoint, methods=["POST"])
        self.add_api_route("/sdapi/v1/refresh-vae", endpoints.post_refresh_vae, methods=["POST"])

        # train api
        self.add_api_route("/sdapi/v1/create/embedding", train.post_create_embedding, methods=["POST"], response_model=models.ResCreate)
        self.add_api_route("/sdapi/v1/create/hypernetwork", train.post_create_hypernetwork, methods=["POST"], response_model=models.ResCreate)
        self.add_api_route("/sdapi/v1/preprocess", train.post_preprocess, methods=["POST"], response_model=models.ResPreprocess)
        self.add_api_route("/sdapi/v1/train/embedding", train.post_train_embedding, methods=["POST"], response_model=models.ResTrain)
        self.add_api_route("/sdapi/v1/train/hypernetwork", train.post_train_hypernetwork, methods=["POST"], response_model=models.ResTrain)

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

    def get_session_start(self, req: Request, agent: Optional[str] = None):
        token = req.cookies.get("access-token") or req.cookies.get("access-token-unsecure")
        user = self.app.tokens.get(token) if hasattr(self.app, 'tokens') else None
        shared.log.info(f'Browser session: user={user} client={req.client.host} agent={agent}')
        return {}

    def prepare_img_gen_request(self, request):
        if hasattr(request, "face") and request.face and not request.script_name and (not request.alwayson_scripts or "Face" not in request.alwayson_scripts.keys()):
            request.script_name = "Face"
            request.script_args = [
                request.face.mode,
                request.face.source_images,
                request.face.ip_model,
                request.face.ip_override_sampler,
                request.face.ip_cache_model,
                request.face.ip_strength,
                request.face.ip_structure,
                request.face.id_strength,
                request.face.id_conditioning,
                request.face.id_cache,
                request.face.pm_trigger,
                request.face.pm_strength,
                request.face.pm_start,
                request.face.fs_cache
            ]
            del request.face

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

    def sanitize_img_gen_request(self, request):
        if hasattr(request, "alwayson_scripts") and request.alwayson_scripts:
            for script_name in request.alwayson_scripts.keys():
                script_obj = request.alwayson_scripts[script_name]

                if script_obj and "args" in script_obj and script_obj["args"]:
                    self.sanitize_args(script_obj["args"])

        if hasattr(request, "script_args") and request.script_args:
            self.sanitize_args(request.script_args)

    def validate_sampler_name(self, name):
        config = sd_samplers.all_samplers_map.get(name, None)
        if config is None:
            raise HTTPException(status_code=404, detail="Sampler not found")
        return name

    def post_text2img(self, txt2imgreq: models.ReqTxt2Img):
        self.prepare_img_gen_request(txt2imgreq)

        script_runner = scripts.scripts_txt2img
        if not script_runner.scripts:
            script_runner.initialize_scripts(False)
            ui.create_ui(None)
        if not self.default_script_arg_txt2img:
            self.default_script_arg_txt2img = script.init_default_script_args(script_runner)
        selectable_scripts, selectable_script_idx = script.get_selectable_script(txt2imgreq.script_name, script_runner)
        populate = txt2imgreq.copy(update={  # Override __init__ params
            "sampler_name": self.validate_sampler_name(txt2imgreq.sampler_name or txt2imgreq.sampler_index),
            "do_not_save_samples": not txt2imgreq.save_images,
            "do_not_save_grid": not txt2imgreq.save_images,
        })
        if populate.sampler_name:
            populate.sampler_index = None  # prevent a warning later on
        args = vars(populate)
        args.pop('script_name', None)
        args.pop('script_args', None) # will refeed them to the pipeline directly after initializing them
        args.pop('face', None)
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
            script_args = script.init_script_args(p, txt2imgreq, self.default_script_arg_txt2img, selectable_scripts, selectable_script_idx, script_runner)
            if selectable_scripts is not None:
                processed = scripts.scripts_txt2img.run(p, *script_args) # Need to pass args as list here
            else:
                p.script_args = tuple(script_args) # Need to pass args as tuple here
                processed = process_images(p)
            shared.state.end(api=False)

        b64images = list(map(helpers.encode_pil_to_base64, processed.images)) if send_images else []
        self.sanitize_img_gen_request(txt2imgreq)
        return models.ResTxt2Img(images=b64images, parameters=vars(txt2imgreq), info=processed.js())

    def post_img2img(self, img2imgreq: models.ReqImg2Img):
        self.prepare_img_gen_request(img2imgreq)

        init_images = img2imgreq.init_images
        if init_images is None:
            raise HTTPException(status_code=404, detail="Init image not found")
        mask = img2imgreq.mask
        if mask:
            mask = helpers.decode_base64_to_image(mask)
        script_runner = scripts.scripts_img2img
        if not script_runner.scripts:
            script_runner.initialize_scripts(True)
            ui.create_ui(None)
        if not self.default_script_arg_img2img:
            self.default_script_arg_img2img = script.init_default_script_args(script_runner)
        selectable_scripts, selectable_script_idx = script.get_selectable_script(img2imgreq.script_name, script_runner)
        populate = img2imgreq.copy(update={  # Override __init__ params
            "sampler_name": self.validate_sampler_name(img2imgreq.sampler_name or img2imgreq.sampler_index),
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
            p.init_images = [helpers.decode_base64_to_image(x) for x in init_images]
            p.scripts = script_runner
            p.outpath_grids = shared.opts.outdir_img2img_grids
            p.outpath_samples = shared.opts.outdir_img2img_samples
            shared.state.begin('api-img2img', api=True)
            script_args = script.init_script_args(p, img2imgreq, self.default_script_arg_img2img, selectable_scripts, selectable_script_idx, script_runner)
            if selectable_scripts is not None:
                processed = scripts.scripts_img2img.run(p, *script_args) # Need to pass args as list here
            else:
                p.script_args = tuple(script_args) # Need to pass args as tuple here
                processed = process_images(p)
            shared.state.end(api=False)

        b64images = list(map(helpers.encode_pil_to_base64, processed.images)) if send_images else []
        if not img2imgreq.include_init_images:
            img2imgreq.init_images = None
            img2imgreq.mask = None
        self.sanitize_img_gen_request(img2imgreq)
        return models.ResImg2Img(images=b64images, parameters=vars(img2imgreq), info=processed.js())

    def set_upscalers(self, req: dict):
        reqDict = vars(req)
        reqDict['extras_upscaler_1'] = reqDict.pop('upscaler_1', None)
        reqDict['extras_upscaler_2'] = reqDict.pop('upscaler_2', None)
        return reqDict

    def extras_single_image_api(self, req: models.ReqProcessImage):
        reqDict = self.set_upscalers(req)
        reqDict['image'] = helpers.decode_base64_to_image(reqDict['image'])
        with self.queue_lock:
            result = postprocessing.run_extras(extras_mode=0, image_folder="", input_dir="", output_dir="", save_output=False, **reqDict)
        return models.ResProcessImage(image=helpers.encode_pil_to_base64(result[0][0]), html_info=result[1])

    def extras_batch_images_api(self, req: models.ReqProcessBatch):
        reqDict = self.set_upscalers(req)
        image_list = reqDict.pop('imageList', [])
        image_folder = [helpers.decode_base64_to_image(x.data) for x in image_list]
        with self.queue_lock:
            result = postprocessing.run_extras(extras_mode=1, image_folder=image_folder, image="", input_dir="", output_dir="", save_output=False, **reqDict)
        return models.ResProcessBatch(images=list(map(helpers.encode_pil_to_base64, result[0])), html_info=result[1])

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
        http_server = UvicornServer(self.app, **config)
        # from modules.server import HypercornServer
        # server = HypercornServer(self.app, **config)
        http_server.start()
        shared.log.info(f'API server: Uvicorn options={config}')
        return http_server
