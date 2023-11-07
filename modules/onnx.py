import os
import PIL
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
from typing import Union, Optional, Callable, Type, List, Any, Dict
from diffusers.image_processor import VaeImageProcessor
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

class OnnxRuntimeModel(diffusers.OnnxRuntimeModel):
    config = {}

    def named_modules(self):
        return ()


# OnnxRuntimeModel Hijack.
OnnxRuntimeModel.__module__ = 'diffusers'
diffusers.OnnxRuntimeModel = OnnxRuntimeModel


class OnnxPipelineBase(diffusers.DiffusionPipeline, metaclass=ABCMeta):
    model_type: str
    sd_model_hash: str
    sd_checkpoint_info: CheckpointInfo
    sd_model_checkpoint: str

    def __init__(self):
        log.warning("Olive implementation is experimental. It contains potentially an issue and is subject to change at any time.")
        self.model_type = self.__class__.__name__


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

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        sess_options = kwargs.get("sess_options", ort.SessionOptions())
        provider = kwargs.get("provider", (shared.opts.onnx_execution_provider, get_execution_provider_options(),))
        model_config = super(OnnxStableDiffusionPipeline, OnnxStableDiffusionPipeline).extract_init_dict(diffusers.DiffusionPipeline.load_config(pretrained_model_name_or_path))
        init_dict = {}
        for d in model_config:
            if 'unet' in d:
                init_dict = d
                break
        init_kwargs = {}
        for k, v in init_dict.items():
            if not isinstance(v, list):
                init_kwargs[k] = v
                continue
            library_name, constructor_name = v
            if library_name is None or constructor_name is None:
                init_kwargs[k] = None
                continue
            library = importlib.import_module(library_name)
            constructor = getattr(library, constructor_name)
            submodel_kwargs = {}
            if issubclass(constructor, diffusers.OnnxRuntimeModel):
                submodel_kwargs["sess_options"] = sess_options
                submodel_kwargs["provider"] = provider
            try:
                init_kwargs[k] = constructor.from_pretrained(
                    os.path.join(pretrained_model_name_or_path, k),
                    **submodel_kwargs,
                )
            except Exception:
                pass
        return OnnxStableDiffusionPipeline(**init_kwargs)

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
        timestep_dtype = diffusers.pipelines.onnx_utils.ORT_TO_NP_TYPE[timestep_dtype]

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

        return diffusers.pipelines.stable_diffusion.StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


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

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        sess_options = kwargs.get("sess_options", ort.SessionOptions())
        provider = kwargs.get("provider", (shared.opts.onnx_execution_provider, get_execution_provider_options(),))
        model_config = super(OnnxStableDiffusionImg2ImgPipeline, OnnxStableDiffusionImg2ImgPipeline).extract_init_dict(diffusers.DiffusionPipeline.load_config(pretrained_model_name_or_path))
        init_dict = {}
        for d in model_config:
            if 'unet' in d:
                init_dict = d
                break
        init_kwargs = {}
        for k, v in init_dict.items():
            if not isinstance(v, list):
                init_kwargs[k] = v
                continue
            library_name, constructor_name = v
            if library_name is None or constructor_name is None:
                init_kwargs[k] = None
                continue
            library = importlib.import_module(library_name)
            constructor = getattr(library, constructor_name)
            submodel_kwargs = {}
            if issubclass(constructor, diffusers.OnnxRuntimeModel):
                submodel_kwargs["sess_options"] = sess_options
                submodel_kwargs["provider"] = provider
            try:
                init_kwargs[k] = constructor.from_pretrained(
                    os.path.join(pretrained_model_name_or_path, k),
                    **submodel_kwargs,
                )
            except Exception:
                pass
        return OnnxStableDiffusionImg2ImgPipeline(**init_kwargs)

    def __call__(
        self,
        prompt: Union[str, List[str]],
        image: Union[np.ndarray, PIL.Image.Image] = None,
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
        timestep_dtype = diffusers.pipelines.onnx_utils.ORT_TO_NP_TYPE[timestep_dtype]

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

        return diffusers.pipelines.stable_diffusion.StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


OnnxStableDiffusionImg2ImgPipeline.__module__ = 'diffusers'
OnnxStableDiffusionImg2ImgPipeline.__name__ = 'OnnxStableDiffusionImg2ImgPipeline'
diffusers.OnnxStableDiffusionImg2ImgPipeline = OnnxStableDiffusionImg2ImgPipeline
diffusers.pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING["onnx-stable-diffusion"] = diffusers.OnnxStableDiffusionImg2ImgPipeline


class OnnxStableDiffusionXLPipeline(optimum.onnxruntime.ORTStableDiffusionXLPipeline, OnnxPipelineBase):
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
        super().__init__(vae_decoder_session, text_encoder_session, unet_session, config, tokenizer, scheduler, feature_extractor, vae_encoder_session, text_encoder_2_session, tokenizer_2, use_io_binding, model_save_dir, add_watermarker)


OnnxStableDiffusionXLPipeline.__module__ = 'optimum.onnxruntime.modeling_diffusion'
OnnxStableDiffusionXLPipeline.__name__ = 'ORTStableDiffusionXLPipeline'
diffusers.OnnxStableDiffusionXLPipeline = OnnxStableDiffusionXLPipeline
diffusers.pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING["onnx-stable-diffusion-xl"] = diffusers.OnnxStableDiffusionXLPipeline


class OnnxStableDiffusionXLImg2ImgPipeline(optimum.onnxruntime.ORTStableDiffusionXLImg2ImgPipeline, OnnxPipelineBase):
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
        super().__init__(vae_decoder_session, text_encoder_session, unet_session, config, tokenizer, scheduler, feature_extractor, vae_encoder_session, text_encoder_2_session, tokenizer_2, use_io_binding, model_save_dir, add_watermarker)


OnnxStableDiffusionXLImg2ImgPipeline.__module__ = 'optimum.onnxruntime.modeling_diffusion'
OnnxStableDiffusionXLImg2ImgPipeline.__name__ = 'ORTStableDiffusionXLImg2ImgPipeline'
diffusers.OnnxStableDiffusionXLImg2ImgPipeline = OnnxStableDiffusionXLImg2ImgPipeline
diffusers.pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING["onnx-stable-diffusion-xl"] = diffusers.OnnxStableDiffusionXLImg2ImgPipeline


class OnnxAutoPipelineBase(OnnxPipelineBase):
    constructor: Type[diffusers.DiffusionPipeline]
    config = {}

    pipeline: diffusers.DiffusionPipeline
    original_filename: str

    def __init__(self, path, pipeline: diffusers.DiffusionPipeline):
        self.original_filename = os.path.basename(path)
        self.pipeline = pipeline
        del pipeline

    @property
    def scheduler(self):
        return self.pipeline.scheduler

    @scheduler.setter
    def scheduler(self, scheduler):
        self.pipeline.scheduler = scheduler

    def derive_properties(self, pipeline: OnnxPipelineBase):
        pipeline.sd_model_hash = self.sd_model_hash
        pipeline.sd_checkpoint_info = self.sd_checkpoint_info
        pipeline.sd_model_checkpoint = self.sd_model_checkpoint
        pipeline.scheduler = self.scheduler
        return pipeline

    def to(self, *args, **kwargs):
        pass

    def convert(self):
        if shared.opts.onnx_execution_provider == ExecutionProvider.ROCm:
            from olive.hardware.accelerator import AcceleratorLookup
            if ExecutionProvider.ROCm not in AcceleratorLookup.EXECUTION_PROVIDERS["gpu"]:
                AcceleratorLookup.EXECUTION_PROVIDERS["gpu"].append(ExecutionProvider.ROCm)

        out_dir = os.path.join(shared.opts.onnx_cached_models_path, self.original_filename)
        if os.path.isdir(out_dir): # already converted (cached)
            self.pipeline = self.derive_properties(
                self.constructor.from_pretrained(
                    out_dir,
                )
            )
            return

        try:
            from olive.workflows import run
            from olive.model import ONNXModel

            shutil.rmtree("cache", ignore_errors=True)
            shutil.rmtree("footprints", ignore_errors=True)

            if os.path.exists(shared.opts.onnx_temp_dir):
                shutil.rmtree(shared.opts.onnx_temp_dir)
            os.mkdir(shared.opts.onnx_temp_dir)

            self.pipeline.save_pretrained(shared.opts.onnx_temp_dir)

            kwargs = {
                "tokenizer": self.pipeline.tokenizer,
                "scheduler": self.pipeline.scheduler,
                "safety_checker": self.pipeline.safety_checker if hasattr(self.pipeline, "safety_checker") else None,
                "feature_extractor": self.pipeline.feature_extractor,
            }
            del self.pipeline

            if shared.opts.onnx_cache_converted:
                shutil.copytree(
                    shared.opts.onnx_temp_dir, out_dir, ignore=shutil.ignore_patterns("weights.pb", "*.onnx", "*.safetensors", "*.ckpt")
                )

            submodels = submodels_sdxl if olive.is_sdxl else submodels_sd
            converted_model_paths = {}

            for submodel in submodels:
                log.info(f"\nConverting {submodel}")

                with open(os.path.join(sd_configs_path, "onnx", f"{'sdxl' if olive.is_sdxl else 'sd'}_{submodel}.json"), "r") as config_file:
                    conversion_config = json.load(config_file)
                conversion_config["input_model"]["config"]["model_path"] = os.path.abspath(shared.opts.onnx_temp_dir)
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
            shutil.rmtree(shared.opts.onnx_temp_dir)

            for submodel in submodels:
                kwargs[submodel] = diffusers.OnnxRuntimeModel.from_pretrained(
                    os.path.dirname(converted_model_paths[submodel]),
                    provider=(shared.opts.onnx_execution_provider, get_execution_provider_options(),),
                )

            self.pipeline = self.derive_properties(
                self.constructor(
                    **kwargs,
                    requires_safety_checker=False,
                )
            )

            self.pipeline.to_json_file(os.path.join(out_dir, "model_index.json"))

            for submodel in submodels:
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
        except Exception as e:
            log.error(f"Failed to convert model '{self.original_filename}'.")
            log.error(e) # for test.
            shutil.rmtree(shared.opts.onnx_temp_dir, ignore_errors=True)
            shutil.rmtree(out_dir, ignore_errors=True)

    def optimize(self):
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
        in_dir = os.path.join(shared.opts.onnx_cached_models_path, self.original_filename)
        out_dir = os.path.join(shared.opts.onnx_cached_models_path, f"{self.original_filename}-{olive.width}w-{olive.height}h")
        if os.path.isdir(out_dir): # already optimized (cached)
            self.pipeline = self.derive_properties(
                self.constructor.from_pretrained(
                    out_dir,
                    sess_options=sess_options,
                )
            )
            return

        try:
            from olive.workflows import run
            from olive.model import ONNXModel

            shutil.rmtree("cache", ignore_errors=True)
            shutil.rmtree("footprints", ignore_errors=True)

            kwargs = {
                "tokenizer": self.pipeline.tokenizer,
                "scheduler": self.pipeline.scheduler,
                "safety_checker": self.pipeline.safety_checker if hasattr(self.pipeline, "safety_checker") else None,
                "feature_extractor": self.pipeline.feature_extractor,
            }
            del self.pipeline

            if shared.opts.onnx_cache_optimized:
                shutil.copytree(
                    in_dir, out_dir, ignore=shutil.ignore_patterns("weights.pb", "*.onnx", "*.safetensors", "*.ckpt")
                )

            submodels = submodels_sdxl if olive.is_sdxl else submodels_sd
            optimized_model_paths = {}

            for submodel in submodels:
                log.info(f"\nOptimizing {submodel}")

                with open(os.path.join(sd_configs_path, "olive", f"{'sdxl' if olive.is_sdxl else 'sd'}_{submodel}.json"), "r") as config_file:
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

            for submodel in submodels:
                kwargs[submodel] = diffusers.OnnxRuntimeModel.from_pretrained(
                    os.path.dirname(optimized_model_paths[submodel]),
                    sess_options=sess_options,
                    provider=(shared.opts.onnx_execution_provider, get_execution_provider_options(),),
                )

            self.pipeline = self.derive_properties(
                self.constructor(
                    **kwargs,
                    requires_safety_checker=False,
                )
            )

            if shared.opts.onnx_cache_optimized:
                self.pipeline.to_json_file(os.path.join(out_dir, "model_index.json"))

                for submodel in submodels:
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
        except Exception as e:
            log.error(f"Failed to optimize model '{self.original_filename}'.")
            log.error(e) # for test.
            shutil.rmtree(out_dir, ignore_errors=True)

    def preprocess(self, width: int, height: int, batch_size: int):
        olive.width = width
        olive.height = height
        olive.batch_size = batch_size

        olive.is_sdxl = "XL" in self.constructor.__name__

        self.convert()

        if shared.opts.onnx_enable_olive:
            if width != height:
                log.warning("Olive detected different width and height. The quality of the result is not guaranteed.")
            self.optimize()

        if not shared.opts.onnx_cache_converted:
            shutil.rmtree(os.path.join(shared.opts.onnx_cached_models_path, self.original_filename))

        return self.pipeline


class OnnxAutoPipelineForText2Image(OnnxAutoPipelineBase):
    def __init__(self, path, pipeline: diffusers.DiffusionPipeline):
        super().__init__(path, pipeline)
        self.constructor = diffusers.OnnxStableDiffusionXLPipeline if hasattr(self.pipeline, "text_encoder_2") else diffusers.OnnxStableDiffusionPipeline
        self.model_type = self.constructor.__name__

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path, **kwargs):
        pipeline = None
        try: # load from Onnx SD model
            pipeline = diffusers.OnnxStableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path, **kwargs)
        except Exception:
            pass
        if pipeline is None:
            try: # load from Onnx SDXL model
                pipeline = diffusers.OnnxStableDiffusionXLPipeline.from_pretrained(pretrained_model_name_or_path, **kwargs)
            except Exception:
                pass
        if pipeline is None:
            try: # load from non-Onnx model
                pipeline = diffusers.AutoPipelineForText2Image.from_pretrained(pretrained_model_name_or_path, **kwargs)
            except Exception:
                pass
        return OnnxAutoPipelineForText2Image(pretrained_model_name_or_path, pipeline)

    @staticmethod
    def from_single_file(pretrained_model_name_or_path, **kwargs):
        return OnnxAutoPipelineForText2Image(pretrained_model_name_or_path, diffusers.StableDiffusionPipeline.from_single_file(pretrained_model_name_or_path, **kwargs))

    @staticmethod
    def from_ckpt(*args, **kwargs):
        return OnnxAutoPipelineForText2Image.from_single_file(**args, **kwargs)

class OnnxAutoPipelineForImage2Image(OnnxAutoPipelineBase):
    def __init__(self, path, pipeline: diffusers.DiffusionPipeline):
        super().__init__(path, pipeline)
        self.constructor = diffusers.OnnxStableDiffusionXLImg2ImgPipeline if hasattr(self.pipeline, "text_encoder_2") else diffusers.OnnxStableDiffusionImg2ImgPipeline
        self.model_type = self.constructor.__name__

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path, **kwargs):
        pipeline = None
        try: # load from Onnx SD model
            pipeline = diffusers.OnnxStableDiffusionImg2ImgPipeline.from_pretrained(pretrained_model_name_or_path, **kwargs)
        except Exception:
            pass
        if pipeline is None:
            try: # load from Onnx SDXL model
                pipeline = diffusers.OnnxStableDiffusionXLImg2ImgPipeline.from_pretrained(pretrained_model_name_or_path, **kwargs)
            except Exception:
                pass
        if pipeline is None:
            try: # load from non-Onnx model
                pipeline = diffusers.AutoPipelineForImage2Image.from_pretrained(pretrained_model_name_or_path, **kwargs)
            except Exception:
                pass
        return OnnxAutoPipelineForImage2Image(pretrained_model_name_or_path, pipeline)

    @staticmethod
    def from_single_file(pretrained_model_name_or_path, **kwargs):
        return OnnxAutoPipelineForImage2Image(pretrained_model_name_or_path, diffusers.StableDiffusionPipeline.from_single_file(pretrained_model_name_or_path, **kwargs))

    @staticmethod
    def from_ckpt(*args, **kwargs):
        return OnnxAutoPipelineForImage2Image.from_single_file(**args, **kwargs)
