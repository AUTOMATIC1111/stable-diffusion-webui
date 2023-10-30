import os
import json
import torch
import shutil
import diffusers
import numpy as np
import onnxruntime as ort
from enum import Enum
from typing import Union, Optional, Callable, List
from transformers.models.clip.modeling_clip import CLIPTextModel, CLIPTextModelWithProjection
from installer import log
from modules import shared
from modules.paths import sd_configs_path
from modules.sd_models import CheckpointInfo

class ExecutionProvider(str, Enum):
    CPU = "CPUExecutionProvider"
    DirectML = "DmlExecutionProvider"
    CUDA = "CUDAExecutionProvider"
    ROCm = "ROCMExecutionProvider"
    OpenVINO = "OpenVINOExecutionProvider"

submodels = ("text_encoder", "unet", "vae_encoder", "vae_decoder",)

available_execution_providers: List[ExecutionProvider] = ort.get_available_providers()

EP_TO_NAME = {
    ExecutionProvider.CPU: "cpu",
    ExecutionProvider.DirectML: "gpu-dml",
    ExecutionProvider.CUDA: "gpu-?", # TODO
    ExecutionProvider.ROCm: "gpu-rocm",
    ExecutionProvider.OpenVINO: "?", # TODO
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
            from olive.hardware.accelerator import AcceleratorLookup
            AcceleratorLookup.EXECUTION_PROVIDERS["gpu"].append(ExecutionProvider.ROCm)
            return ExecutionProvider.ROCm
        else:
            log.warning("Currently, there's no pypi release for onnxruntime-rocm. Please download and install .whl file from https://download.onnxruntime.ai/ The inference will be fall back to CPU.")
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
        if shared.opts.openvino_dtype != "Default" and not shared.opts.openvino_hetero_gpu:
            raw_openvino_device = f"{raw_openvino_device}_{shared.opts.openvino_dtype}"
        execution_provider_options["device_type"] = raw_openvino_device
        del execution_provider_options["device_id"]

    return execution_provider_options

class OnnxRuntimeModel(diffusers.OnnxRuntimeModel):
    config = {}

    def named_modules(self):
        return ()


diffusers.OnnxRuntimeModel = OnnxRuntimeModel


class OnnxStableDiffusionPipeline(diffusers.OnnxStableDiffusionPipeline):
    model_type: str
    sd_model_hash: str
    sd_checkpoint_info: CheckpointInfo
    sd_model_checkpoint: str

    @staticmethod
    def from_pretrained(*args, **kwargs):
        if "provider" not in kwargs:
            kwargs["provider"] = (shared.opts.onnx_execution_provider, get_execution_provider_options(),)
        components = diffusers.OnnxStableDiffusionPipeline.from_pretrained(*args, **kwargs).components
        return OnnxStableDiffusionPipeline(**components, requires_safety_checker=False)

    def apply(self, dummy_pipeline):
        self.sd_model_hash = dummy_pipeline.sd_model_hash
        self.sd_checkpoint_info = dummy_pipeline.sd_checkpoint_info
        self.sd_model_checkpoint = dummy_pipeline.sd_model_checkpoint
        return self

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
        import inspect

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

class OlivePipeline(diffusers.DiffusionPipeline):
    sd_model_hash: str
    sd_checkpoint_info: CheckpointInfo
    sd_model_checkpoint: str
    config = {}

    unoptimized: diffusers.DiffusionPipeline
    original_filename: str

    def __init__(self, path, pipeline: diffusers.DiffusionPipeline):
        self.original_filename = os.path.basename(path)
        self.unoptimized = pipeline
        del pipeline
        if not os.path.exists(shared.opts.olive_temp_dir):
            os.mkdir(shared.opts.olive_temp_dir)
        self.unoptimized.save_pretrained(shared.opts.olive_temp_dir)

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path, **kwargs):
        return OlivePipeline(pretrained_model_name_or_path, diffusers.DiffusionPipeline.from_pretrained(pretrained_model_name_or_path, **kwargs))

    @staticmethod
    def from_single_file(pretrained_model_name_or_path, **kwargs):
        return OlivePipeline(pretrained_model_name_or_path, diffusers.StableDiffusionPipeline.from_single_file(pretrained_model_name_or_path, **kwargs))

    @staticmethod
    def from_ckpt(*args, **kwargs):
        return OlivePipeline.from_single_file(**args, **kwargs)

    def to(self, *args, **kwargs):
        pass

    def optimize(self, width: int, height: int):
        from olive.workflows import run
        from olive.model import ONNXModel

        if width != height:
            log.warning("Olive received different width and height. The quality of the result is not guaranteed.")

        out_dir = os.path.join(shared.opts.olive_cached_models_path, f"{self.original_filename}-{width}w-{height}h")
        if os.path.isdir(out_dir):
            del self.unoptimized
            return OnnxStableDiffusionPipeline.from_pretrained(
                out_dir,
            ).apply(self)

        try:
            if shared.opts.onnx_cache_optimized:
                shutil.copytree(
                    shared.opts.olive_temp_dir, out_dir, ignore=shutil.ignore_patterns("weights.pb", "*.onnx", "*.safetensors", "*.ckpt")
                )

            optimize_config["width"] = width
            optimize_config["height"] = height

            optimized_model_paths = {}

            for submodel in submodels:
                log.info(f"\nOptimizing {submodel}")

                with open(os.path.join(sd_configs_path, "olive", f"config_{submodel}.json"), "r") as config_file:
                    olive_config = json.load(config_file)
                olive_config["engine"]["execution_providers"] = [shared.opts.onnx_execution_provider]
                olive_config["passes"]["optimize"]["config"]["float16"] = shared.opts.onnx_olive_float16

                run(olive_config)

                with open(os.path.join("footprints", f"{submodel}_{EP_TO_NAME[shared.opts.onnx_execution_provider]}_footprints.json"), "r") as footprint_file:
                    footprints = json.load(footprint_file)
                conversion_footprint = None
                optimizer_footprint = None
                for _, footprint in footprints.items():
                    if footprint["from_pass"] == "OnnxConversion":
                        conversion_footprint = footprint
                    elif footprint["from_pass"] == "OrtTransformersOptimization":
                        optimizer_footprint = footprint

                assert conversion_footprint and optimizer_footprint, "Failed to optimize model"

                optimized_model_paths[submodel] = ONNXModel(
                    **optimizer_footprint["model_config"]["config"]
                ).model_path

                log.info(f"Optimized {submodel}")
            shutil.rmtree(shared.opts.olive_temp_dir)

            kwargs = {
                "tokenizer": self.unoptimized.tokenizer,
                "scheduler": self.unoptimized.scheduler,
                "safety_checker": self.unoptimized.safety_checker if hasattr(self.unoptimized, "safety_checker") else None,
                "feature_extractor": self.unoptimized.feature_extractor,
            }
            del self.unoptimized
            for submodel in submodels:
                kwargs[submodel] = diffusers.OnnxRuntimeModel.from_pretrained(
                    os.path.dirname(optimized_model_paths[submodel]),
                )

            pipeline = OnnxStableDiffusionPipeline(
                **kwargs,
                requires_safety_checker=False,
            ).apply(self)
            del kwargs
            if shared.opts.onnx_cache_optimized:
                pipeline.to_json_file(os.path.join(out_dir, "model_index.json"))

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
        except Exception:
            log.error(f"Failed to optimize model '{self.original_filename}'.")
            shutil.rmtree(shared.opts.olive_temp_dir, ignore_errors=True)
            shutil.rmtree(out_dir, ignore_errors=True)
            pipeline = None
        shutil.rmtree("cache", ignore_errors=True)
        shutil.rmtree("footprints", ignore_errors=True)
        return pipeline

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

optimize_config = {
    "is_sdxl": False,

    "width": 512,
    "height": 512,
}


# Helper latency-only dataloader that creates random tensors with no label
class RandomDataLoader:
    def __init__(self, create_inputs_func, batchsize, torch_dtype):
        self.create_input_func = create_inputs_func
        self.batchsize = batchsize
        self.torch_dtype = torch_dtype

    def __getitem__(self, idx):
        label = None
        return self.create_input_func(self.batchsize, self.torch_dtype), label

# -----------------------------------------------------------------------------
# TEXT ENCODER
# -----------------------------------------------------------------------------


def text_encoder_inputs(batchsize, torch_dtype):
    input_ids = torch.zeros((batchsize, 77), dtype=torch_dtype)
    return {
        "input_ids": input_ids,
        "output_hidden_states": True,
    } if optimize_config["is_sdxl"] else input_ids


def text_encoder_load(model_name):
    model = CLIPTextModel.from_pretrained(os.path.abspath(shared.opts.olive_temp_dir), subfolder="text_encoder")
    return model


def text_encoder_conversion_inputs(model):
    return text_encoder_inputs(1, torch.int32)


def text_encoder_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(text_encoder_inputs, batchsize, torch.int32)


# -----------------------------------------------------------------------------
# TEXT ENCODER 2
# -----------------------------------------------------------------------------


def text_encoder_2_inputs(batchsize, torch_dtype):
    return {
        "input_ids": torch.zeros((batchsize, 77), dtype=torch_dtype),
        "output_hidden_states": True,
    }


def text_encoder_2_load(model_name):
    model = CLIPTextModelWithProjection.from_pretrained(os.path.abspath(shared.opts.olive_temp_dir), subfolder="text_encoder_2")
    return model


def text_encoder_2_conversion_inputs(model):
    return text_encoder_2_inputs(1, torch.int64)


def text_encoder_2_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(text_encoder_2_inputs, batchsize, torch.int64)


# -----------------------------------------------------------------------------
# UNET
# -----------------------------------------------------------------------------


def unet_inputs(batchsize, torch_dtype, is_conversion_inputs=False):
    # TODO (pavignol): All the multiplications by 2 here are bacause the XL base has 2 text encoders
    # For refiner, it should be multiplied by 1 (single text encoder)
    height = optimize_config["height"]
    width = optimize_config["width"]

    if optimize_config["is_sdxl"]:
        inputs = {
            "sample": torch.rand((2 * batchsize, 4, height // 8, width // 8), dtype=torch_dtype),
            "timestep": torch.rand((1,), dtype=torch_dtype),
            "encoder_hidden_states": torch.rand((2 * batchsize, 77, height * 2), dtype=torch_dtype),
        }

        if is_conversion_inputs:
            inputs["additional_inputs"] = {
                "added_cond_kwargs": {
                    "text_embeds": torch.rand((2 * batchsize, height + 256), dtype=torch_dtype),
                    "time_ids": torch.rand((2 * batchsize, 6), dtype=torch_dtype),
                }
            }
        else:
            inputs["text_embeds"] = torch.rand((2 * batchsize, height + 256), dtype=torch_dtype)
            inputs["time_ids"] = torch.rand((2 * batchsize, 6), dtype=torch_dtype)
    else:
        inputs = {
            "sample": torch.rand((batchsize, 4, height // 8, width // 8), dtype=torch_dtype),
            "timestep": torch.rand((batchsize,), dtype=torch_dtype),
            "encoder_hidden_states": torch.rand((batchsize, 77, height + 256), dtype=torch_dtype),
            "return_dict": False,
        }

    return inputs


def unet_load(model_name):
    model = diffusers.UNet2DConditionModel.from_pretrained(os.path.abspath(shared.opts.olive_temp_dir), subfolder="unet")
    return model


def unet_conversion_inputs(model):
    return tuple(unet_inputs(1, torch.float32, True).values())


def unet_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(unet_inputs, batchsize, torch.float16)


# -----------------------------------------------------------------------------
# VAE ENCODER
# -----------------------------------------------------------------------------


def vae_encoder_inputs(batchsize, torch_dtype):
    return {
        "sample": torch.rand((batchsize, 3, optimize_config["height"], optimize_config["width"]), dtype=torch_dtype),
        "return_dict": False,
    }


def vae_encoder_load(model_name):
    source = os.path.join(os.path.abspath(shared.opts.olive_temp_dir), "vae")
    if not os.path.isdir(source):
        source += "_encoder"
    model = diffusers.AutoencoderKL.from_pretrained(source)
    model.forward = lambda sample, return_dict: model.encode(sample, return_dict)[0].sample()
    return model


def vae_encoder_conversion_inputs(model):
    return tuple(vae_encoder_inputs(1, torch.float32).values())


def vae_encoder_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(vae_encoder_inputs, batchsize, torch.float16)


# -----------------------------------------------------------------------------
# VAE DECODER
# -----------------------------------------------------------------------------


def vae_decoder_inputs(batchsize, torch_dtype):
    return {
        "latent_sample": torch.rand((batchsize, 4, optimize_config["height"] // 8, optimize_config["width"] // 8), dtype=torch_dtype),
        "return_dict": False,
    }


def vae_decoder_load(model_name):
    source = os.path.join(os.path.abspath(shared.opts.olive_temp_dir), "vae")
    if not os.path.isdir(source):
        source += "_decoder"
    model = diffusers.AutoencoderKL.from_pretrained(source)
    model.forward = model.decode
    return model


def vae_decoder_conversion_inputs(model):
    return tuple(vae_decoder_inputs(1, torch.float32).values())


def vae_decoder_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(vae_decoder_inputs, batchsize, torch.float16)
