import os
import json
import torch
import shutil
import inspect
from packaging import version
import numpy as np
import diffusers
import onnxruntime as ort
import optimum.onnxruntime
from abc import ABCMeta
from typing import Union, Optional, Callable, Type, Tuple, List, Any, Dict
from diffusers.pipelines.onnx_utils import ORT_TO_NP_TYPE
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput
from diffusers.image_processor import VaeImageProcessor, PipelineImageInput
from optimum.pipelines.diffusers.pipeline_stable_diffusion_xl import rescale_noise_cfg
from installer import log
from modules import shared
from modules.paths import sd_configs_path
from modules.sd_models import CheckpointInfo
from modules.processing import StableDiffusionProcessing
from modules.olive import config
from modules.onnx import OnnxFakeModule, submodels_sd, submodels_sdxl, submodels_sdxl_refiner
from modules.onnx_utils import check_pipeline_sdxl, load_init_dict, load_submodel, load_submodels, load_pipeline, get_sess_options, patch_kwargs, construct_refiner_pipeline
from modules.onnx_ep import ExecutionProvider, EP_TO_NAME, get_provider


class OnnxPipelineBase(OnnxFakeModule, diffusers.DiffusionPipeline, metaclass=ABCMeta):
    model_type: str
    sd_model_hash: str
    sd_checkpoint_info: CheckpointInfo
    sd_model_checkpoint: str

    def __init__(self):
        self.model_type = self.__class__.__name__

    def override_processing(self, p: StableDiffusionProcessing):
        disable_classifier_free_guidance = p.cfg_scale < 0.01 or "turbo" in self.sd_checkpoint_info.model_name.lower()
        if disable_classifier_free_guidance:
            p.cfg_scale = 0.0

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **_):
        return OnnxRawPipeline(
            cls,
            pretrained_model_name_or_path,
        )

    @classmethod
    def from_single_file(cls, pretrained_model_name_or_path, **_):
        return OnnxRawPipeline(
            cls,
            pretrained_model_name_or_path,
        )

    @classmethod
    def from_ckpt(cls, pretrained_model_name_or_path, **_):
        return cls.from_single_file(pretrained_model_name_or_path)


class OnnxRawPipeline(OnnxPipelineBase):
    config = {}
    _is_sdxl: bool
    is_refiner: bool
    from_huggingface_cache: bool
    path: os.PathLike
    original_filename: str

    constructor: Type[OnnxPipelineBase]
    submodels: List[str]
    init_dict: Dict[str, Tuple[str]] = {}

    scheduler: Any = None # for Img2Img

    def __init__(self, constructor: Type[OnnxPipelineBase], path: os.PathLike):
        self.model_type = constructor.__name__
        self._is_sdxl = check_pipeline_sdxl(constructor)
        self.is_refiner = "refiner" in str(path).lower()
        self.from_huggingface_cache = shared.opts.diffusers_dir in os.path.abspath(path)
        self.path = path
        self.original_filename = os.path.basename(path)

        self.constructor = construct_refiner_pipeline if self.is_refiner else constructor
        self.submodels = (submodels_sdxl_refiner if self.is_refiner else submodels_sdxl) if self._is_sdxl else submodels_sd
        if os.path.isdir(path):
            self.init_dict = load_init_dict(constructor, path)
            self.scheduler = load_submodel(self.path, None, "scheduler", self.init_dict["scheduler"])
        else:
            try:
                cls = None
                if self._is_sdxl:
                    cls = diffusers.StableDiffusionXLPipeline
                else:
                    cls = diffusers.StableDiffusionPipeline
                pipeline = cls.from_single_file(path)
                self.scheduler = pipeline.scheduler
                if os.path.isdir(shared.opts.onnx_temp_dir):
                    shutil.rmtree(shared.opts.onnx_temp_dir)
                os.mkdir(shared.opts.onnx_temp_dir)
                pipeline.save_pretrained(shared.opts.onnx_temp_dir)
                del pipeline
                self.init_dict = load_init_dict(constructor, shared.opts.onnx_temp_dir)
            except Exception:
                log.error('Failed to load pipeline to optimize.')
        if "vae" in self.init_dict:
            del self.init_dict["vae"]

    def derive_properties(self, pipeline: diffusers.DiffusionPipeline):
        pipeline.sd_model_hash = self.sd_model_hash
        pipeline.sd_checkpoint_info = self.sd_checkpoint_info
        pipeline.sd_model_checkpoint = self.sd_model_checkpoint
        pipeline.scheduler = self.scheduler
        return pipeline

    def convert(self, in_dir: os.PathLike):
        if not shared.cmd_opts.debug:
            ort.set_default_logger_severity(3)

        out_dir = os.path.join(shared.opts.onnx_cached_models_path, self.original_filename)
        if os.path.isdir(out_dir): # already converted (cached)
            return out_dir

        try:
            from olive.workflows import run
            from olive.model import ONNXModel

            shutil.rmtree("cache", ignore_errors=True)
            shutil.rmtree("footprints", ignore_errors=True)

            if shared.opts.onnx_cache_converted:
                shutil.copytree(
                    in_dir, out_dir, ignore=shutil.ignore_patterns("weights.pb", "*.onnx", "*.safetensors", "*.ckpt")
                )

            converted_model_paths = {}

            for submodel in self.submodels:
                log.info(f"\nConverting {submodel}")

                with open(os.path.join(sd_configs_path, "onnx", f"{'sdxl' if self._is_sdxl else 'sd'}_{submodel}.json"), "r") as config_file:
                    conversion_config = json.load(config_file)
                conversion_config["input_model"]["config"]["model_path"] = os.path.abspath(in_dir)
                conversion_config["engine"]["execution_providers"] = [shared.opts.onnx_execution_provider]

                run(conversion_config)

                with open(os.path.join("footprints", f"{submodel}_{EP_TO_NAME[shared.opts.onnx_execution_provider]}_footprints.json"), "r") as footprint_file:
                    footprints = json.load(footprint_file)
                conversion_footprint = None
                for _, footprint in footprints.items():
                    if footprint["from_pass"] == "OnnxConversion":
                        conversion_footprint = footprint

                assert conversion_footprint, "Failed to convert model"

                converted_model_paths[submodel] = ONNXModel(
                    **conversion_footprint["model_config"]["config"]
                ).model_path

                log.info(f"Converted {submodel}")

            for submodel in self.submodels:
                src_path = converted_model_paths[submodel]
                src_parent = os.path.dirname(src_path)
                dst_parent = os.path.join(out_dir, submodel)
                dst_path = os.path.join(dst_parent, "model.onnx")
                if not os.path.isdir(dst_parent):
                    os.mkdir(dst_parent)
                shutil.copyfile(src_path, dst_path)

                data_src_path = os.path.join(src_parent, (os.path.basename(src_path) + ".data"))
                if os.path.isfile(data_src_path):
                    data_dst_path = os.path.join(dst_parent, (os.path.basename(dst_path) + ".data"))
                    shutil.copyfile(data_src_path, data_dst_path)

                weights_src_path = os.path.join(src_parent, "weights.pb")
                if os.path.isfile(weights_src_path):
                    weights_dst_path = os.path.join(dst_parent, "weights.pb")
                    shutil.copyfile(weights_src_path, weights_dst_path)
            del converted_model_paths

            kwargs = {}

            init_dict = self.init_dict.copy()
            for submodel in self.submodels:
                kwargs[submodel] = diffusers.OnnxRuntimeModel.load_model(
                    os.path.join(out_dir, submodel, "model.onnx"),
                    provider=get_provider(),
                ) if self._is_sdxl else diffusers.OnnxRuntimeModel.from_pretrained(
                    os.path.join(out_dir, submodel),
                    provider=get_provider(),
                )
                if submodel in init_dict:
                    del init_dict[submodel] # already loaded as OnnxRuntimeModel.
            kwargs.update(load_submodels(in_dir, self._is_sdxl, init_dict)) # load others.
            kwargs = patch_kwargs(self.constructor, kwargs)

            pipeline = self.constructor(**kwargs)
            model_index = json.loads(pipeline.to_json_string())
            del pipeline

            for k, v in init_dict.items(): # copy missing submodels. (ORTStableDiffusionXLPipeline)
                if k not in model_index:
                    model_index[k] = v

            with open(os.path.join(out_dir, "model_index.json"), 'w') as file:
                json.dump(model_index, file)

            return out_dir
        except Exception as e:
            log.error(f"Failed to convert model '{self.original_filename}'.")
            log.error(e) # for test.
            shutil.rmtree(shared.opts.onnx_temp_dir, ignore_errors=True)
            shutil.rmtree(out_dir, ignore_errors=True)
            return None

    def optimize(self, in_dir: os.PathLike):
        if not shared.cmd_opts.debug:
            ort.set_default_logger_severity(4)

        out_dir = os.path.join(shared.opts.onnx_cached_models_path, f"{self.original_filename}-{config.width}w-{config.height}h")
        if os.path.isdir(out_dir): # already optimized (cached)
            return out_dir

        if not shared.opts.onnx_cache_optimized:
            out_dir = shared.opts.onnx_temp_dir

        try:
            from olive.workflows import run
            from olive.model import ONNXModel

            shutil.rmtree("cache", ignore_errors=True)
            shutil.rmtree("footprints", ignore_errors=True)

            if shared.opts.onnx_cache_optimized:
                shutil.copytree(
                    in_dir, out_dir, ignore=shutil.ignore_patterns("weights.pb", "*.onnx", "*.safetensors", "*.ckpt")
                )

            optimized_model_paths = {}

            for submodel in self.submodels:
                log.info(f"\nOptimizing {submodel}")

                with open(os.path.join(sd_configs_path, "olive", f"{'sdxl' if self._is_sdxl else 'sd'}_{submodel}.json"), "r") as config_file:
                    olive_config = json.load(config_file)
                pass_key = f"optimize_{shared.opts.onnx_execution_provider}"
                olive_config["pass_flows"] = [[pass_key]]
                olive_config["input_model"]["config"]["model_path"] = os.path.abspath(os.path.join(in_dir, submodel, "model.onnx"))
                olive_config["passes"][pass_key]["config"]["float16"] = shared.opts.onnx_olive_float16
                olive_config["engine"]["execution_providers"] = [shared.opts.onnx_execution_provider]
                if shared.opts.onnx_execution_provider == ExecutionProvider.CUDA or shared.opts.onnx_execution_provider == ExecutionProvider.ROCm:
                    if version.parse(ort.__version__) < version.parse("1.17.0"):
                        olive_config["passes"][pass_key]["config"]["optimization_options"] = {"enable_skip_group_norm": False}
                    if shared.opts.onnx_olive_float16:
                        olive_config["passes"][pass_key]["config"]["keep_io_types"] = False

                run(olive_config)

                with open(os.path.join("footprints", f"{submodel}_{EP_TO_NAME[shared.opts.onnx_execution_provider]}_footprints.json"), "r") as footprint_file:
                    footprints = json.load(footprint_file)
                optimizer_footprint = None
                for _, footprint in footprints.items():
                    if footprint["from_pass"] == "OrtTransformersOptimization":
                        optimizer_footprint = footprint

                assert optimizer_footprint, "Failed to optimize model"

                optimized_model_paths[submodel] = ONNXModel(
                    **optimizer_footprint["model_config"]["config"]
                ).model_path

                log.info(f"Optimized {submodel}")

            for submodel in self.submodels:
                src_path = optimized_model_paths[submodel]
                src_parent = os.path.dirname(src_path)
                dst_parent = os.path.join(out_dir, submodel)
                dst_path = os.path.join(dst_parent, "model.onnx")
                if not os.path.isdir(dst_parent):
                    os.mkdir(dst_parent)
                shutil.copyfile(src_path, dst_path)

                data_src_path = os.path.join(src_parent, (os.path.basename(src_path) + ".data"))
                if os.path.isfile(data_src_path):
                    data_dst_path = os.path.join(dst_parent, (os.path.basename(dst_path) + ".data"))
                    shutil.copyfile(data_src_path, data_dst_path)

                weights_src_path = os.path.join(src_parent, "weights.pb")
                if os.path.isfile(weights_src_path):
                    weights_dst_path = os.path.join(dst_parent, "weights.pb")
                    shutil.copyfile(weights_src_path, weights_dst_path)
            del optimized_model_paths

            kwargs = {}

            init_dict = self.init_dict.copy()
            for submodel in self.submodels:
                kwargs[submodel] = diffusers.OnnxRuntimeModel.load_model(
                    os.path.join(out_dir, submodel, "model.onnx"),
                    provider=get_provider(),
                ) if self._is_sdxl else diffusers.OnnxRuntimeModel.from_pretrained(
                    os.path.join(out_dir, submodel),
                    provider=get_provider(),
                )
                if submodel in init_dict:
                    del init_dict[submodel] # already loaded as OnnxRuntimeModel.
            kwargs.update(load_submodels(in_dir, self._is_sdxl, init_dict)) # load others.
            kwargs = patch_kwargs(self.constructor, kwargs)

            pipeline = self.constructor(**kwargs)
            model_index = json.loads(pipeline.to_json_string())
            del pipeline

            for k, v in init_dict.items(): # copy missing submodels. (ORTStableDiffusionXLPipeline)
                if k not in model_index:
                    model_index[k] = v

            with open(os.path.join(out_dir, "model_index.json"), 'w') as file:
                json.dump(model_index, file)

            return out_dir
        except Exception as e:
            log.error(f"Failed to optimize model '{self.original_filename}'.")
            log.error(e) # for test.
            shutil.rmtree(shared.opts.onnx_temp_dir, ignore_errors=True)
            shutil.rmtree(out_dir, ignore_errors=True)
            return None

    def preprocess(self, p: StableDiffusionProcessing):
        in_dir = self.path if os.path.isdir(self.path) else shared.opts.onnx_temp_dir

        config.from_huggingface_cache = self.from_huggingface_cache
        config.use_fp16_fixed_vae = self._is_sdxl and not shared.opts.diffusers_vae_upcast

        config.is_sdxl = self._is_sdxl

        config.width = p.width
        config.height = p.height
        config.batch_size = p.batch_size

        config.cross_attention_dim = 2048 if self._is_sdxl else (256 + p.height)
        config.time_ids_size = 6 if self._is_sdxl and not self.is_refiner else 5

        kwargs = {
            "provider": get_provider(),
            "sess_options": get_sess_options(p.batch_size if p.cfg_scale < 0.01 or "turbo" in str(self.path).lower() else p.batch_size * 2, p.height, p.width, self._is_sdxl),
        }

        converted_dir = self.convert(in_dir)
        if converted_dir is None:
            log.error('Failed to convert model. The generation will fall back to unconverted one.')
            return self.derive_properties(load_pipeline(diffusers.StableDiffusionXLPipeline if self._is_sdxl else diffusers.StableDiffusionPipeline, self.path, **kwargs))
        out_dir = converted_dir

        if shared.opts.onnx_enable_olive:
            log.warning("Olive implementation is experimental. It contains potentially an issue and is subject to change at any time.")
            if p.width != p.height:
                log.warning("Olive detected different width and height. The quality of the result is not guaranteed.")
            optimized_dir = self.optimize(converted_dir)
            if optimized_dir is None:
                log.error('Failed to optimize pipeline. The generation will fall back to unoptimized one.')
                return self.derive_properties(load_pipeline(diffusers.OnnxStableDiffusionXLPipeline if self._is_sdxl else diffusers.OnnxStableDiffusionPipeline, converted_dir, **kwargs))
            out_dir = optimized_dir

        pipeline = self.derive_properties(load_pipeline(diffusers.OnnxStableDiffusionXLPipeline if self._is_sdxl else diffusers.OnnxStableDiffusionPipeline, out_dir, **kwargs))

        if not shared.opts.onnx_cache_converted:
            shutil.rmtree(converted_dir)
        shutil.rmtree(shared.opts.onnx_temp_dir, ignore_errors=True)

        return pipeline


def prepare_latents(
    init_noise_sigma: float,
    batch_size: int,
    height: int,
    width: int,
    dtype: np.dtype,
    generator: Union[torch.Generator, List[torch.Generator]],
    latents: Union[np.ndarray, None]=None,
    num_channels_latents=4,
    vae_scale_factor=8,
):
    shape = (batch_size, num_channels_latents, height // vae_scale_factor, width // vae_scale_factor)
    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
            f" size of {batch_size}. Make sure the batch size matches the length of the generators."
        )

    if latents is None:
        if isinstance(generator, list):
            generator = [g.seed() for g in generator]
            if len(generator) == 1:
                generator = generator[0]

        latents = np.random.default_rng(generator).standard_normal(shape).astype(dtype)
    elif latents.shape != shape:
        raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")

    # scale the initial noise by the standard deviation required by the scheduler
    latents = latents * np.float64(init_noise_sigma)

    return latents


class OnnxStableDiffusionPipeline(diffusers.OnnxStableDiffusionPipeline, OnnxPipelineBase):
    def __init__(
        self,
        vae_encoder: diffusers.OnnxRuntimeModel,
        vae_decoder: diffusers.OnnxRuntimeModel,
        text_encoder: diffusers.OnnxRuntimeModel,
        tokenizer,
        unet: diffusers.OnnxRuntimeModel,
        scheduler,
        safety_checker: diffusers.OnnxRuntimeModel,
        feature_extractor,
        requires_safety_checker: bool = True
    ):
        super().__init__(vae_encoder, vae_decoder, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor, requires_safety_checker)

    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[np.ndarray] = None,
        prompt_embeds: Optional[np.ndarray] = None,
        negative_prompt_embeds: Optional[np.ndarray] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, np.ndarray], None]] = None,
        callback_steps: int = 1,
    ):
        # check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if generator is None:
            generator = torch.Generator("cpu")

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        prompt_embeds = self._encode_prompt(
            prompt,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # get the initial random noise unless the user supplied it
        latents = prepare_latents(
            self.scheduler.init_noise_sigma,
            batch_size * num_images_per_prompt,
            height,
            width,
            prompt_embeds.dtype,
            generator,
            latents
        )

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        timestep_dtype = next(
            (input.type for input in self.unet.model.get_inputs() if input.name == "timestep"), "tensor(float)"
        )
        timestep_dtype = ORT_TO_NP_TYPE[timestep_dtype]

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = np.concatenate([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(torch.from_numpy(latent_model_input), t)
            latent_model_input = latent_model_input.cpu().numpy()

            # predict the noise residual
            timestep = np.array([t], dtype=timestep_dtype)
            noise_pred = self.unet(sample=latent_model_input, timestep=timestep, encoder_hidden_states=prompt_embeds)
            noise_pred = noise_pred[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            scheduler_output = self.scheduler.step(
                torch.from_numpy(noise_pred), t, torch.from_numpy(latents), **extra_step_kwargs
            )
            latents = scheduler_output.prev_sample.numpy()

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, torch.from_numpy(latents))

        latents /= self.vae_decoder.config.get("scaling_factor", 0.18215)

        has_nsfw_concept = None

        if not output_type == "latent":
            # image = self.vae_decoder(latent_sample=latents)[0]
            # it seems likes there is a strange result for using half-precision vae decoder if batchsize>1
            image = np.concatenate(
                [self.vae_decoder(latent_sample=latents[i : i + 1])[0] for i in range(latents.shape[0])]
            )

            image = np.clip(image / 2 + 0.5, 0, 1)
            image = image.transpose((0, 2, 3, 1))

            if self.safety_checker is not None:
                safety_checker_input = self.feature_extractor(
                    self.numpy_to_pil(image), return_tensors="np"
                ).pixel_values.astype(image.dtype)

                images, has_nsfw_concept = [], []
                for i in range(image.shape[0]):
                    image_i, has_nsfw_concept_i = self.safety_checker(
                        clip_input=safety_checker_input[i : i + 1], images=image[i : i + 1]
                    )
                    images.append(image_i)
                    has_nsfw_concept.append(has_nsfw_concept_i[0])
                image = np.concatenate(images)

            if output_type == "pil":
                image = self.numpy_to_pil(image)
        else:
            image = latents

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


class OnnxStableDiffusionImg2ImgPipeline(diffusers.OnnxStableDiffusionImg2ImgPipeline, OnnxPipelineBase):
    image_processor: VaeImageProcessor

    def __init__(
        self,
        vae_encoder: diffusers.OnnxRuntimeModel,
        vae_decoder: diffusers.OnnxRuntimeModel,
        text_encoder: diffusers.OnnxRuntimeModel,
        tokenizer,
        unet: diffusers.OnnxRuntimeModel,
        scheduler,
        safety_checker: diffusers.OnnxRuntimeModel,
        feature_extractor,
        requires_safety_checker: bool = True
    ):
        super().__init__(vae_encoder, vae_decoder, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor, requires_safety_checker)
        self.image_processor = VaeImageProcessor(vae_scale_factor=64)

    def __call__(
        self,
        prompt: Union[str, List[str]],
        image: PipelineImageInput = None,
        strength: float = 0.8,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[np.ndarray] = None,
        negative_prompt_embeds: Optional[np.ndarray] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, np.ndarray], None]] = None,
        callback_steps: int = 1,
    ):
        # check inputs. Raise error if not correct
        self.check_inputs(prompt, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds)

        # define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if generator is None:
            generator = torch.Generator("cpu")

        if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        image = self.image_processor.preprocess(image).cpu().numpy()

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        prompt_embeds = self._encode_prompt(
            prompt,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        scaling_factor = self.vae_decoder.config.get("scaling_factor", 0.18215)

        latents_dtype = prompt_embeds.dtype
        image = image.astype(latents_dtype)
        # encode the init image into latents and scale the latents
        init_latents = self.vae_encoder(sample=image)[0]
        init_latents = scaling_factor * init_latents

        if isinstance(prompt, str):
            prompt = [prompt]

        init_latents = np.concatenate([init_latents] * num_images_per_prompt, axis=0)

        # get the original timestep using init_timestep
        offset = self.scheduler.config.get("steps_offset", 0)
        init_timestep = int(num_inference_steps * strength) + offset
        init_timestep = min(init_timestep, num_inference_steps)

        timesteps = self.scheduler.timesteps.numpy()[-init_timestep]
        timesteps = np.array([timesteps] * batch_size * num_images_per_prompt)

        if isinstance(generator, list):
            generator = [g.seed() for g in generator]
            if len(generator) == 1:
                generator = generator[0]

        # add noise to latents using the timesteps
        noise = np.random.default_rng(generator).standard_normal(init_latents.shape).astype(latents_dtype)
        init_latents = self.scheduler.add_noise(
            torch.from_numpy(init_latents), torch.from_numpy(noise), torch.from_numpy(timesteps)
        )
        init_latents = init_latents.numpy()

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        latents = init_latents

        t_start = max(num_inference_steps - init_timestep + offset, 0)
        timesteps = self.scheduler.timesteps[t_start:].numpy()

        timestep_dtype = next(
            (input.type for input in self.unet.model.get_inputs() if input.name == "timestep"), "tensor(float)"
        )
        timestep_dtype = ORT_TO_NP_TYPE[timestep_dtype]

        for i, t in enumerate(self.progress_bar(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = np.concatenate([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(torch.from_numpy(latent_model_input), t)
            latent_model_input = latent_model_input.cpu().numpy()

            # predict the noise residual
            timestep = np.array([t], dtype=timestep_dtype)
            noise_pred = self.unet(sample=latent_model_input, timestep=timestep, encoder_hidden_states=prompt_embeds)[
                0
            ]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            scheduler_output = self.scheduler.step(
                torch.from_numpy(noise_pred), t, torch.from_numpy(latents), **extra_step_kwargs
            )
            latents = scheduler_output.prev_sample.numpy()

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, torch.from_numpy(latents))

        latents /= scaling_factor

        has_nsfw_concept = None

        if not output_type == "latent":
            # image = self.vae_decoder(latent_sample=latents)[0]
            # it seems likes there is a strange result for using half-precision vae decoder if batchsize>1
            image = np.concatenate(
                [self.vae_decoder(latent_sample=latents[i : i + 1])[0] for i in range(latents.shape[0])]
            )

            image = np.clip(image / 2 + 0.5, 0, 1)
            image = image.transpose((0, 2, 3, 1))

            if self.safety_checker is not None:
                safety_checker_input = self.feature_extractor(
                    self.numpy_to_pil(image), return_tensors="np"
                ).pixel_values.astype(image.dtype)

                images, has_nsfw_concept = [], []
                for i in range(image.shape[0]):
                    image_i, has_nsfw_concept_i = self.safety_checker(
                        clip_input=safety_checker_input[i : i + 1], images=image[i : i + 1]
                    )
                    images.append(image_i)
                    has_nsfw_concept.append(has_nsfw_concept_i[0])
                image = np.concatenate(images)

            if output_type == "pil":
                image = self.numpy_to_pil(image)
        else:
            image = latents

        # skip postprocess

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


class OnnxStableDiffusionInpaintPipeline(diffusers.OnnxStableDiffusionInpaintPipeline, OnnxPipelineBase):
    def __init__(
        self,
        vae_encoder: diffusers.OnnxRuntimeModel,
        vae_decoder: diffusers.OnnxRuntimeModel,
        text_encoder: diffusers.OnnxRuntimeModel,
        tokenizer,
        unet: diffusers.OnnxRuntimeModel,
        scheduler,
        safety_checker: diffusers.OnnxRuntimeModel,
        feature_extractor,
        requires_safety_checker: bool = True
    ):
        super().__init__(vae_encoder, vae_decoder, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor, requires_safety_checker)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        image: PipelineImageInput,
        mask_image: PipelineImageInput,
        masked_image_latents: torch.FloatTensor = None,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        strength: float = 1.0,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[np.ndarray] = None,
        prompt_embeds: Optional[np.ndarray] = None,
        negative_prompt_embeds: Optional[np.ndarray] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, np.ndarray], None]] = None,
        callback_steps: int = 1,
    ):
        # check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if generator is None:
            generator = torch.Generator("cpu")

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        prompt_embeds = self._encode_prompt(
            prompt,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        num_channels_latents = diffusers.pipelines.stable_diffusion.pipeline_onnx_stable_diffusion_inpaint.NUM_LATENT_CHANNELS
        latents = prepare_latents(
            self.scheduler.init_noise_sigma,
            batch_size * num_images_per_prompt,
            height,
            width,
            prompt_embeds.dtype,
            generator,
            latents,
            num_channels_latents,
            8,
        )

        scaling_factor = self.vae_decoder.config.get("scaling_factor", 0.18215)

        # prepare mask and masked_image
        mask, masked_image = diffusers.pipelines.stable_diffusion.pipeline_onnx_stable_diffusion_inpaint.prepare_mask_and_masked_image(
            image[0],
            mask_image,
            (height // 8, width // 8),
        )
        mask = mask.astype(latents.dtype)
        masked_image = masked_image.astype(latents.dtype)

        masked_image_latents = self.vae_encoder(sample=masked_image)[0]
        masked_image_latents = scaling_factor * masked_image_latents

        # duplicate mask and masked_image_latents for each generation per prompt
        mask = mask.repeat(batch_size * num_images_per_prompt, 0)
        masked_image_latents = masked_image_latents.repeat(batch_size * num_images_per_prompt, 0)

        mask = np.concatenate([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            np.concatenate([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )

        num_channels_mask = mask.shape[1]
        num_channels_masked_image = masked_image_latents.shape[1]

        unet_input_channels = diffusers.pipelines.stable_diffusion.pipeline_onnx_stable_diffusion_inpaint.NUM_UNET_INPUT_CHANNELS
        if num_channels_latents + num_channels_mask + num_channels_masked_image != unet_input_channels:
            raise ValueError(
                "Incorrect configuration settings! The config of `pipeline.unet` expects"
                f" {unet_input_channels} but received `num_channels_latents`: {num_channels_latents} +"
                f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                f" = {num_channels_latents+num_channels_masked_image+num_channels_mask}. Please verify the config of"
                " `pipeline.unet` or your `mask_image` or `image` input."
            )

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        timestep_dtype = next(
            (input.type for input in self.unet.model.get_inputs() if input.name == "timestep"), "tensor(float)"
        )
        timestep_dtype = ORT_TO_NP_TYPE[timestep_dtype]

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = np.concatenate([latents] * 2) if do_classifier_free_guidance else latents
            # concat latents, mask, masked_image_latnets in the channel dimension
            latent_model_input = self.scheduler.scale_model_input(torch.from_numpy(latent_model_input), t)
            latent_model_input = latent_model_input.cpu().numpy()
            latent_model_input = np.concatenate([latent_model_input, mask, masked_image_latents], axis=1)

            # predict the noise residual
            timestep = np.array([t], dtype=timestep_dtype)
            noise_pred = self.unet(sample=latent_model_input, timestep=timestep, encoder_hidden_states=prompt_embeds)[
                0
            ]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            scheduler_output = self.scheduler.step(
                torch.from_numpy(noise_pred), t, torch.from_numpy(latents), **extra_step_kwargs
            )
            latents = scheduler_output.prev_sample.numpy()

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                step_idx = i // getattr(self.scheduler, "order", 1)
                callback(step_idx, t, torch.from_numpy(latents))

        latents /= scaling_factor

        has_nsfw_concept = None

        if not output_type == "latent":
            # image = self.vae_decoder(latent_sample=latents)[0]
            # it seems likes there is a strange result for using half-precision vae decoder if batchsize>1
            image = np.concatenate(
                [self.vae_decoder(latent_sample=latents[i : i + 1])[0] for i in range(latents.shape[0])]
            )

            image = np.clip(image / 2 + 0.5, 0, 1)
            image = image.transpose((0, 2, 3, 1))

            if self.safety_checker is not None:
                safety_checker_input = self.feature_extractor(
                    self.numpy_to_pil(image), return_tensors="np"
                ).pixel_values.astype(image.dtype)

                images, has_nsfw_concept = [], []
                for i in range(image.shape[0]):
                    image_i, has_nsfw_concept_i = self.safety_checker(
                        clip_input=safety_checker_input[i : i + 1], images=image[i : i + 1]
                    )
                    images.append(image_i)
                    has_nsfw_concept.append(has_nsfw_concept_i[0])
                image = np.concatenate(images)

            if output_type == "pil":
                image = self.numpy_to_pil(image)
        else:
            image = latents

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


class OnnxStableDiffusionXLPipeline(OnnxPipelineBase, optimum.onnxruntime.ORTStableDiffusionXLPipeline):
    def __init__(
        self,
        vae_decoder,
        text_encoder,
        unet,
        config: Dict[str, Any],
        tokenizer,
        scheduler,
        feature_extractor = None,
        vae_encoder = None,
        text_encoder_2 = None,
        tokenizer_2 = None,
        use_io_binding: bool | None = None,
        model_save_dir = None,
        add_watermarker: bool | None = None
    ):
        super(optimum.onnxruntime.ORTStableDiffusionXLPipeline, self).__init__(vae_decoder, text_encoder, unet, config, tokenizer, scheduler, feature_extractor, vae_encoder, text_encoder_2, tokenizer_2, use_io_binding, model_save_dir, add_watermarker)

    # Adapted from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline.__call__
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[np.ndarray] = None,
        prompt_embeds: Optional[np.ndarray] = None,
        negative_prompt_embeds: Optional[np.ndarray] = None,
        pooled_prompt_embeds: Optional[np.ndarray] = None,
        negative_pooled_prompt_embeds: Optional[np.ndarray] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, np.ndarray], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config["sample_size"] * self.vae_scale_factor
        width = width or self.unet.config["sample_size"] * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

        # 2. Define call parameters
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if generator is None:
            generator = torch.Generator("cpu")

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self._encode_prompt(
            prompt,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        latents = prepare_latents(
            self.scheduler.init_noise_sigma,
            batch_size * num_images_per_prompt,
            height,
            width,
            prompt_embeds.dtype,
            generator,
            latents,
            self.unet.config.get("in_channels", 4),
            self.vae_scale_factor,
        )

        # 6. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        add_time_ids = (original_size + crops_coords_top_left + target_size,)
        add_time_ids = np.array(add_time_ids, dtype=prompt_embeds.dtype)

        if do_classifier_free_guidance:
            prompt_embeds = np.concatenate((negative_prompt_embeds, prompt_embeds), axis=0)
            add_text_embeds = np.concatenate((negative_pooled_prompt_embeds, add_text_embeds), axis=0)
            add_time_ids = np.concatenate((add_time_ids, add_time_ids), axis=0)
        add_time_ids = np.repeat(add_time_ids, batch_size * num_images_per_prompt, axis=0)

        # Adapted from diffusers to extend it for other runtimes than ORT
        timestep_dtype = self.unet.input_dtype.get("timestep", np.float32)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        for i, t in enumerate(self.progress_bar(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = np.concatenate([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(torch.from_numpy(latent_model_input), t)
            latent_model_input = latent_model_input.cpu().numpy()

            # predict the noise residual
            timestep = np.array([t], dtype=timestep_dtype)
            noise_pred = self.unet(
                sample=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                text_embeds=add_text_embeds,
                time_ids=add_time_ids,
            )
            noise_pred = noise_pred[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                if guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

            # compute the previous noisy sample x_t -> x_t-1
            scheduler_output = self.scheduler.step(
                torch.from_numpy(noise_pred), t, torch.from_numpy(latents), **extra_step_kwargs
            )
            latents = scheduler_output.prev_sample.numpy()

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, torch.from_numpy(latents))

        if output_type == "latent":
            image = latents
        else:
            latents /= self.vae_decoder.config.get("scaling_factor", 0.18215)
            # it seems likes there is a strange result for using half-precision vae decoder if batchsize>1
            image = np.concatenate(
                [self.vae_decoder(latent_sample=latents[i : i + 1])[0] for i in range(latents.shape[0])]
            )
            # apply watermark if available
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)
            image = self.image_processor.postprocess(image, output_type=output_type)

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)


class OnnxStableDiffusionXLImg2ImgPipeline(OnnxPipelineBase, optimum.onnxruntime.ORTStableDiffusionXLImg2ImgPipeline):
    def __init__(
        self,
        vae_decoder,
        text_encoder,
        unet,
        config: Dict[str, Any],
        tokenizer,
        scheduler,
        feature_extractor = None,
        vae_encoder = None,
        text_encoder_2 = None,
        tokenizer_2 = None,
        use_io_binding: bool | None = None,
        model_save_dir = None,
        add_watermarker: bool | None = None
    ):
        super(optimum.onnxruntime.ORTStableDiffusionXLImg2ImgPipeline, self).__init__(vae_decoder, text_encoder, unet, config, tokenizer, scheduler, feature_extractor, vae_encoder, text_encoder_2, tokenizer_2, use_io_binding, model_save_dir, add_watermarker)

    # Adapted from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline.__call__
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        image: PipelineImageInput = None,
        strength: float = 0.3,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[np.ndarray] = None,
        prompt_embeds: Optional[np.ndarray] = None,
        negative_prompt_embeds: Optional[np.ndarray] = None,
        pooled_prompt_embeds: Optional[np.ndarray] = None,
        negative_pooled_prompt_embeds: Optional[np.ndarray] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, np.ndarray], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        aesthetic_score: float = 6.0,
        negative_aesthetic_score: float = 2.5,
    ):
        # 0. Check inputs. Raise error if not correct
        self.check_inputs(prompt, strength, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds)

        # 1. Define call parameters
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if generator is None:
            generator = torch.Generator("cpu")

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 2. Encode input prompt
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self._encode_prompt(
            prompt,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        )

        # 3. Preprocess image
        image = self.image_processor.preprocess(image)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength)
        latent_timestep = np.repeat(timesteps[:1], batch_size * num_images_per_prompt, axis=0)
        timestep_dtype = self.unet.input_dtype.get("timestep", np.float32)

        latents_dtype = prompt_embeds.dtype
        image = image.astype(latents_dtype)

        # 5. Prepare latent variables
        latents = self.prepare_latents(
            image, latent_timestep, batch_size, num_images_per_prompt, latents_dtype, generator
        )

        # 6. Prepare extra step kwargs
        extra_step_kwargs = {}
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        height, width = latents.shape[-2:]
        height = height * self.vae_scale_factor
        width = width * self.vae_scale_factor
        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 8. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        add_time_ids, add_neg_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            aesthetic_score,
            negative_aesthetic_score,
            dtype=prompt_embeds.dtype,
        )

        if do_classifier_free_guidance:
            prompt_embeds = np.concatenate((negative_prompt_embeds, prompt_embeds), axis=0)
            add_text_embeds = np.concatenate((negative_pooled_prompt_embeds, add_text_embeds), axis=0)
            add_time_ids = np.concatenate((add_time_ids, add_time_ids), axis=0)
        add_time_ids = np.repeat(add_time_ids, batch_size * num_images_per_prompt, axis=0)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        for i, t in enumerate(self.progress_bar(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = np.concatenate([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(torch.from_numpy(latent_model_input), t)
            latent_model_input = latent_model_input.cpu().numpy()

            # predict the noise residual
            timestep = np.array([t], dtype=timestep_dtype)
            noise_pred = self.unet(
                sample=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                text_embeds=add_text_embeds,
                time_ids=add_time_ids,
            )
            noise_pred = noise_pred[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                if guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

            # compute the previous noisy sample x_t -> x_t-1
            scheduler_output = self.scheduler.step(
                torch.from_numpy(noise_pred), t, torch.from_numpy(latents), **extra_step_kwargs
            )
            latents = scheduler_output.prev_sample.numpy()

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, torch.from_numpy(latents))

        if output_type == "latent":
            image = latents
        else:
            latents /= self.vae_decoder.config.get("scaling_factor", 0.18215)
            # it seems likes there is a strange result for using half-precision vae decoder if batchsize>1
            image = np.concatenate(
                [self.vae_decoder(latent_sample=latents[i : i + 1])[0] for i in range(latents.shape[0])]
            )
            # apply watermark if available
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)
            image = self.image_processor.postprocess(image, output_type=output_type)

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)


def do_diffusers_hijack():
    diffusers.OnnxStableDiffusionPipeline = OnnxStableDiffusionPipeline
    diffusers.pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING["onnx-stable-diffusion"] = diffusers.OnnxStableDiffusionPipeline

    OnnxStableDiffusionImg2ImgPipeline.__module__ = 'diffusers'
    OnnxStableDiffusionImg2ImgPipeline.__name__ = 'OnnxStableDiffusionImg2ImgPipeline'
    diffusers.OnnxStableDiffusionImg2ImgPipeline = OnnxStableDiffusionImg2ImgPipeline
    diffusers.pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING["onnx-stable-diffusion"] = diffusers.OnnxStableDiffusionImg2ImgPipeline

    OnnxStableDiffusionInpaintPipeline.__module__ = 'diffusers'
    OnnxStableDiffusionInpaintPipeline.__name__ = 'OnnxStableDiffusionInpaintPipeline'
    diffusers.OnnxStableDiffusionInpaintPipeline = OnnxStableDiffusionInpaintPipeline
    diffusers.pipelines.auto_pipeline.AUTO_INPAINT_PIPELINES_MAPPING["onnx-stable-diffusion"] = diffusers.OnnxStableDiffusionInpaintPipeline

    OnnxStableDiffusionXLPipeline.__module__ = 'optimum.onnxruntime.modeling_diffusion'
    OnnxStableDiffusionXLPipeline.__name__ = 'ORTStableDiffusionXLPipeline'
    diffusers.OnnxStableDiffusionXLPipeline = OnnxStableDiffusionXLPipeline
    diffusers.pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING["onnx-stable-diffusion-xl"] = diffusers.OnnxStableDiffusionXLPipeline

    OnnxStableDiffusionXLImg2ImgPipeline.__module__ = 'optimum.onnxruntime.modeling_diffusion'
    OnnxStableDiffusionXLImg2ImgPipeline.__name__ = 'ORTStableDiffusionXLImg2ImgPipeline'
    diffusers.OnnxStableDiffusionXLImg2ImgPipeline = OnnxStableDiffusionXLImg2ImgPipeline
    diffusers.pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING["onnx-stable-diffusion-xl"] = diffusers.OnnxStableDiffusionXLImg2ImgPipeline
