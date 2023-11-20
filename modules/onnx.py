import os
import json
import torch
import shutil
import inspect
import importlib
import numpy as np
import onnxruntime as ort
import diffusers
import optimum.onnxruntime
from enum import Enum
from abc import ABCMeta
from typing import Union, Optional, Callable, Type, Tuple, List, Any, Dict
from diffusers.pipelines.onnx_utils import ORT_TO_NP_TYPE
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.image_processor import VaeImageProcessor, PipelineImageInput
from installer import log
from modules import shared, olive
from modules.paths import sd_configs_path
from modules.sd_models import CheckpointInfo

class ExecutionProvider(str, Enum):
    CPU = "CPUExecutionProvider"
    DirectML = "DmlExecutionProvider"
    CUDA = "CUDAExecutionProvider"
    ROCm = "ROCMExecutionProvider"
    OpenVINO = "OpenVINOExecutionProvider"

submodels_sd = ("text_encoder", "unet", "vae_encoder", "vae_decoder",)
submodels_sdxl = ("text_encoder", "text_encoder_2", "unet", "vae_encoder", "vae_decoder",)
available_execution_providers: List[ExecutionProvider] = ort.get_available_providers()

EP_TO_NAME = {
    ExecutionProvider.CPU: "cpu",
    ExecutionProvider.DirectML: "gpu-dml",
    ExecutionProvider.CUDA: "gpu-?", # TODO
    ExecutionProvider.ROCm: "gpu-rocm",
    ExecutionProvider.OpenVINO: "gpu", # Other devices can use --use-openvino instead of olive
}

def get_default_execution_provider() -> ExecutionProvider:
    from modules import devices
    if devices.backend == "cpu":
        return ExecutionProvider.CPU
    elif devices.backend == "directml":
        return ExecutionProvider.DirectML
    elif devices.backend == "cuda":
        return ExecutionProvider.CUDA
    elif devices.backend == "rocm":
        if ExecutionProvider.ROCm in available_execution_providers:
            return ExecutionProvider.ROCm
        else:
            log.warning("Currently, there's no pypi release for onnxruntime-rocm. Please download and install .whl file from https://download.onnxruntime.ai/")
    elif devices.backend == "ipex" or devices.backend == "openvino":
        return ExecutionProvider.OpenVINO
    return ExecutionProvider.CPU

def get_execution_provider_options():
    execution_provider_options = {
        "device_id": int(shared.cmd_opts.device_id or 0),
    }

    if shared.opts.onnx_execution_provider == ExecutionProvider.ROCm:
        if ExecutionProvider.ROCm in available_execution_providers:
            execution_provider_options["tunable_op_enable"] = 1
            execution_provider_options["tunable_op_tuning_enable"] = 1
        else:
            log.warning("Currently, there's no pypi release for onnxruntime-rocm. Please download and install .whl file from https://download.onnxruntime.ai/ The inference will be fall back to CPU.")
    elif shared.opts.onnx_execution_provider == ExecutionProvider.OpenVINO:
        from modules.intel.openvino import get_device as get_raw_openvino_device
        raw_openvino_device = get_raw_openvino_device()
        if shared.opts.onnx_olive_float16 and not shared.opts.openvino_hetero_gpu:
            raw_openvino_device = f"{raw_openvino_device}_FP16"
        execution_provider_options["device_type"] = raw_openvino_device
        del execution_provider_options["device_id"]

    return execution_provider_options


class OnnxFakeModule:
    device = torch.device("cpu")
    dtype = torch.float32

    def to(self, *args, **kwargs):
        return self

    def type(self, *args, **kwargs):
        return self


class OnnxRuntimeModel(OnnxFakeModule, diffusers.OnnxRuntimeModel):
    config = {} # dummy

    def named_modules(self): # dummy
        return ()


# OnnxRuntimeModel Hijack.
OnnxRuntimeModel.__module__ = 'diffusers'
diffusers.OnnxRuntimeModel = OnnxRuntimeModel


def load_init_dict(cls: Type[diffusers.DiffusionPipeline], path: os.PathLike):
    merged: Dict[str, Any] = {}
    extracted = cls.extract_init_dict(diffusers.DiffusionPipeline.load_config(path))
    for dict in extracted:
        merged.update(dict)
    merged = merged.items()
    R: Dict[str, Tuple[str]] = {}
    for k, v in merged:
        if isinstance(v, list):
            R[k] = v
    return R


def load_submodel(path: os.PathLike, submodel_name: str, item: List[Union[str, None]], **kwargs):
    lib, atr = item
    if lib is None or atr is None:
        return None
    library = importlib.import_module(lib)
    attribute = getattr(library, atr)
    if not issubclass(attribute, diffusers.OnnxRuntimeModel):
        kwargs.clear()
    return attribute.from_pretrained(
        os.path.join(path, submodel_name),
        **kwargs,
    )


def load_submodels(path: os.PathLike, init_dict: Dict[str, Type], **kwargs):
    sess_options = kwargs.get("sess_options", ort.SessionOptions())
    provider = kwargs.get("provider", (shared.opts.onnx_execution_provider, get_execution_provider_options(),))
    loaded = {}
    for k, v in init_dict.items():
        if not isinstance(v, list):
            loaded[k] = v
            continue
        try:
            loaded[k] = load_submodel(path, k, v, sess_options=sess_options, provider=provider)
        except Exception:
            pass
    return loaded


def load_pipeline(cls: Type[diffusers.DiffusionPipeline], path: os.PathLike):
    if os.path.isdir(path):
        return cls(**load_submodels(path, load_init_dict(cls, path)))
    else:
        return cls.from_single_file(path)


class OnnxPipelineBase(OnnxFakeModule, diffusers.DiffusionPipeline, metaclass=ABCMeta):
    model_type: str
    sd_model_hash: str
    sd_checkpoint_info: CheckpointInfo
    sd_model_checkpoint: str

    def __init__(self):
        self.model_type = self.__class__.__name__

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
    path: os.PathLike
    original_filename: str

    constructor: Type[OnnxPipelineBase]
    submodels: List[str]
    load_runtime_model: Callable
    init_dict: Dict[str, Tuple[str]] = {}

    scheduler: Any = None # for Img2Img

    def __init__(self, constructor: Type[OnnxPipelineBase], path: os.PathLike):
        self.model_type = constructor.__name__
        self._is_sdxl = 'XL' in self.model_type
        self.path = path
        self.original_filename = os.path.basename(path)

        self.constructor = constructor
        self.submodels = submodels_sdxl if self._is_sdxl else submodels_sd
        self.load_runtime_model = diffusers.OnnxRuntimeModel.load_model if self._is_sdxl else diffusers.OnnxRuntimeModel.from_pretrained
        if os.path.isdir(path):
            self.init_dict = load_init_dict(constructor, path)
            self.scheduler = load_submodel(self.path, "scheduler", self.init_dict["scheduler"])
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

            kwargs = {}

            init_dict = self.init_dict.copy()
            for submodel in self.submodels:
                kwargs[submodel] = self.load_runtime_model(
                    os.path.dirname(converted_model_paths[submodel]),
                    provider=(shared.opts.onnx_execution_provider, get_execution_provider_options(),),
                )
                if submodel in init_dict:
                    del init_dict[submodel] # already loaded as OnnxRuntimeModel.
            kwargs.update(load_submodels(in_dir, init_dict)) # load others.
            kwargs["safety_checker"] = None
            kwargs["requires_safety_checker"] = False

            pipeline = self.constructor(**kwargs)
            pipeline.to_json_file(os.path.join(out_dir, "model_index.json"))
            del pipeline

            for submodel in self.submodels:
                src_path = converted_model_paths[submodel]
                src_parent = os.path.dirname(src_path)
                dst_parent = os.path.join(out_dir, submodel)
                dst_path = os.path.join(dst_parent, "model.onnx")
                if not os.path.isdir(dst_parent):
                    os.mkdir(dst_parent)
                shutil.copyfile(src_path, dst_path)

                weights_src_path = os.path.join(src_parent, "weights.pb")
                if os.path.isfile(weights_src_path):
                    weights_dst_path = os.path.join(dst_parent, "weights.pb")
                    shutil.copyfile(weights_src_path, weights_dst_path)
            return out_dir
        except Exception as e:
            log.error(f"Failed to convert model '{self.original_filename}'.")
            log.error(e) # for test.
            shutil.rmtree(shared.opts.onnx_temp_dir, ignore_errors=True)
            shutil.rmtree(out_dir, ignore_errors=True)
            return None

    def optimize(self, in_dir: os.PathLike):
        sess_options = ort.SessionOptions()
        sess_options.add_free_dimension_override_by_name("unet_sample_batch", olive.batch_size * 2)
        sess_options.add_free_dimension_override_by_name("unet_sample_channels", 4)
        sess_options.add_free_dimension_override_by_name("unet_sample_height", olive.height // 8)
        sess_options.add_free_dimension_override_by_name("unet_sample_width", olive.width // 8)
        sess_options.add_free_dimension_override_by_name("unet_time_batch", 1)
        sess_options.add_free_dimension_override_by_name("unet_hidden_batch", olive.batch_size * 2)
        sess_options.add_free_dimension_override_by_name("unet_hidden_sequence", 77)
        if olive.is_sdxl:
            sess_options.add_free_dimension_override_by_name("unet_text_embeds_batch", olive.batch_size * 2)
            sess_options.add_free_dimension_override_by_name("unet_text_embeds_size", 1280)
            sess_options.add_free_dimension_override_by_name("unet_time_ids_batch", olive.batch_size * 2)
            sess_options.add_free_dimension_override_by_name("unet_time_ids_size", 6)

        out_dir = os.path.join(shared.opts.onnx_cached_models_path, f"{self.original_filename}-{olive.width}w-{olive.height}h")
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
                olive_config["input_model"]["config"]["model_path"] = os.path.abspath(os.path.join(in_dir, submodel, "model.onnx"))
                olive_config["passes"]["optimize"]["config"]["float16"] = shared.opts.onnx_olive_float16
                if (submodel == "unet" or "vae" in submodel) and (shared.opts.onnx_execution_provider == ExecutionProvider.CUDA or shared.opts.onnx_execution_provider == ExecutionProvider.ROCm):
                    olive_config["passes"]["optimize"]["config"]["optimization_options"]["group_norm_channels_last"] = True
                olive_config["engine"]["execution_providers"] = [shared.opts.onnx_execution_provider]

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

            kwargs = {}

            init_dict = self.init_dict.copy()
            for submodel in self.submodels:
                kwargs[submodel] = self.load_runtime_model(
                    os.path.dirname(optimized_model_paths[submodel]),
                    sess_options=sess_options,
                    provider=(shared.opts.onnx_execution_provider, get_execution_provider_options(),),
                )
                if submodel in init_dict:
                    del init_dict[submodel] # already loaded as OnnxRuntimeModel.
            kwargs.update(load_submodels(in_dir, init_dict)) # load others.
            kwargs["safety_checker"] = None
            kwargs["requires_safety_checker"] = False

            pipeline = self.constructor(**kwargs)
            pipeline.to_json_file(os.path.join(out_dir, "model_index.json"))

            for submodel in self.submodels:
                src_path = optimized_model_paths[submodel]
                src_parent = os.path.dirname(src_path)
                dst_parent = os.path.join(out_dir, submodel)
                dst_path = os.path.join(dst_parent, "model.onnx")
                if not os.path.isdir(dst_parent):
                    os.mkdir(dst_parent)
                shutil.copyfile(src_path, dst_path)

                weights_src_path = os.path.join(src_parent, (os.path.basename(src_path) + ".data"))
                if os.path.isfile(weights_src_path):
                    weights_dst_path = os.path.join(dst_parent, (os.path.basename(dst_path) + ".data"))
                    shutil.copyfile(weights_src_path, weights_dst_path)
            return out_dir
        except Exception as e:
            log.error(f"Failed to optimize model '{self.original_filename}'.")
            log.error(e) # for test.
            shutil.rmtree(shared.opts.onnx_temp_dir, ignore_errors=True)
            shutil.rmtree(out_dir, ignore_errors=True)
            return None

    def preprocess(self, width: int, height: int, batch_size: int):
        olive.width = width
        olive.height = height
        olive.batch_size = batch_size

        olive.is_sdxl = self._is_sdxl

        converted_dir = self.convert(self.path if os.path.isdir(self.path) else shared.opts.onnx_temp_dir)
        if converted_dir is None:
            log.error('Failed to convert model. The generation will fall back to unconverted one.')
            return self.derive_properties(load_pipeline(diffusers.StableDiffusionXLPipeline if self._is_sdxl else diffusers.StableDiffusionPipeline, self.path))
        out_dir = converted_dir

        if shared.opts.onnx_enable_olive:
            log.warning("Olive implementation is experimental. It contains potentially an issue and is subject to change at any time.")
            if width != height:
                log.warning("Olive detected different width and height. The quality of the result is not guaranteed.")
            optimized_dir = self.optimize(converted_dir)
            if optimized_dir is None:
                log.error('Failed to optimize pipeline. The generation will fall back to unoptimized one.')
                return self.derive_properties(load_pipeline(diffusers.OnnxStableDiffusionXLPipeline if self._is_sdxl else diffusers.OnnxStableDiffusionPipeline, converted_dir))
            out_dir = optimized_dir

        pipeline = self.derive_properties(load_pipeline(diffusers.OnnxStableDiffusionXLPipeline if self._is_sdxl else diffusers.OnnxStableDiffusionPipeline, out_dir))

        if not shared.opts.onnx_cache_converted:
            shutil.rmtree(converted_dir)
        shutil.rmtree(shared.opts.onnx_temp_dir)

        return pipeline


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
        latents_dtype = prompt_embeds.dtype
        latents_shape = (batch_size * num_images_per_prompt, 4, height // 8, width // 8)
        if latents is None:
            if isinstance(generator, list):
                generator = [g.seed() for g in generator]
                if len(generator) == 1:
                    generator = generator[0]

            latents = np.random.default_rng(generator).standard_normal(latents_shape).astype(latents_dtype)
        elif latents.shape != latents_shape:
            raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        latents = latents * np.float64(self.scheduler.init_noise_sigma)

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

        latents = 1 / 0.18215 * latents

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


diffusers.OnnxStableDiffusionPipeline = OnnxStableDiffusionPipeline
diffusers.pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING["onnx-stable-diffusion"] = diffusers.OnnxStableDiffusionPipeline


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

        latents_dtype = prompt_embeds.dtype
        image = image.astype(latents_dtype)
        # encode the init image into latents and scale the latents
        init_latents = self.vae_encoder(sample=image)[0]
        init_latents = 0.18215 * init_latents

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

        latents = 1 / 0.18215 * latents

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


OnnxStableDiffusionImg2ImgPipeline.__module__ = 'diffusers'
OnnxStableDiffusionImg2ImgPipeline.__name__ = 'OnnxStableDiffusionImg2ImgPipeline'
diffusers.OnnxStableDiffusionImg2ImgPipeline = OnnxStableDiffusionImg2ImgPipeline
diffusers.pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING["onnx-stable-diffusion"] = diffusers.OnnxStableDiffusionImg2ImgPipeline


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
        latents_shape = (batch_size * num_images_per_prompt, num_channels_latents, height // 8, width // 8)
        latents_dtype = prompt_embeds.dtype
        if latents is None:
            if isinstance(generator, list):
                generator = [g.seed() for g in generator]
                if len(generator) == 1:
                    generator = generator[0]

            latents = np.random.default_rng(generator).standard_normal(latents_shape).astype(latents_dtype)
        else:
            if latents.shape != latents_shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")

        # prepare mask and masked_image
        mask, masked_image = diffusers.pipelines.stable_diffusion.pipeline_onnx_stable_diffusion_inpaint.prepare_mask_and_masked_image(image[0], mask_image, latents_shape[-2:])
        mask = mask.astype(latents.dtype)
        masked_image = masked_image.astype(latents.dtype)

        masked_image_latents = self.vae_encoder(sample=masked_image)[0]
        masked_image_latents = 0.18215 * masked_image_latents

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

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * np.float64(self.scheduler.init_noise_sigma)

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
                callback(step_idx, t, latents)

        latents = 1 / 0.18215 * latents

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


OnnxStableDiffusionInpaintPipeline.__module__ = 'diffusers'
OnnxStableDiffusionInpaintPipeline.__name__ = 'OnnxStableDiffusionInpaintPipeline'
diffusers.OnnxStableDiffusionInpaintPipeline = OnnxStableDiffusionInpaintPipeline
diffusers.pipelines.auto_pipeline.AUTO_INPAINT_PIPELINES_MAPPING["onnx-stable-diffusion"] = diffusers.OnnxStableDiffusionInpaintPipeline


class OnnxStableDiffusionXLPipeline(OnnxPipelineBase, optimum.onnxruntime.ORTStableDiffusionXLPipeline):
    def __init__(
        self,
        vae_decoder_session,
        text_encoder_session,
        unet_session,
        config: Dict[str, Any],
        tokenizer,
        scheduler,
        feature_extractor = None,
        vae_encoder_session = None,
        text_encoder_2_session = None,
        tokenizer_2 = None,
        use_io_binding: bool | None = None,
        model_save_dir = None,
        add_watermarker: bool | None = None
    ):
        super(optimum.onnxruntime.ORTStableDiffusionXLPipeline, self).__init__(vae_decoder_session, text_encoder_session, unet_session, config, tokenizer, scheduler, feature_extractor, vae_encoder_session, text_encoder_2_session, tokenizer_2, use_io_binding, model_save_dir, add_watermarker)


OnnxStableDiffusionXLPipeline.__module__ = 'optimum.onnxruntime.modeling_diffusion'
OnnxStableDiffusionXLPipeline.__name__ = 'ORTStableDiffusionXLPipeline'
diffusers.OnnxStableDiffusionXLPipeline = OnnxStableDiffusionXLPipeline
diffusers.pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING["onnx-stable-diffusion-xl"] = diffusers.OnnxStableDiffusionXLPipeline


class OnnxStableDiffusionXLImg2ImgPipeline(OnnxPipelineBase, optimum.onnxruntime.ORTStableDiffusionXLImg2ImgPipeline):
    def __init__(
        self,
        vae_decoder_session,
        text_encoder_session,
        unet_session,
        config: Dict[str, Any],
        tokenizer,
        scheduler,
        feature_extractor = None,
        vae_encoder_session = None,
        text_encoder_2_session = None,
        tokenizer_2 = None,
        use_io_binding: bool | None = None,
        model_save_dir = None,
        add_watermarker: bool | None = None
    ):
        super(optimum.onnxruntime.ORTStableDiffusionXLImg2ImgPipeline, self).__init__(vae_decoder_session, text_encoder_session, unet_session, config, tokenizer, scheduler, feature_extractor, vae_encoder_session, text_encoder_2_session, tokenizer_2, use_io_binding, model_save_dir, add_watermarker)


OnnxStableDiffusionXLImg2ImgPipeline.__module__ = 'optimum.onnxruntime.modeling_diffusion'
OnnxStableDiffusionXLImg2ImgPipeline.__name__ = 'ORTStableDiffusionXLImg2ImgPipeline'
diffusers.OnnxStableDiffusionXLImg2ImgPipeline = OnnxStableDiffusionXLImg2ImgPipeline
diffusers.pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING["onnx-stable-diffusion-xl"] = diffusers.OnnxStableDiffusionXLImg2ImgPipeline
