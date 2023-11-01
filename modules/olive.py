import os
import sys
import json
import torch
import shutil
import diffusers
from transformers.models.clip.modeling_clip import CLIPTextModel, CLIPTextModelWithProjection
from installer import log
from modules import shared
from modules.paths import sd_configs_path
from modules.sd_models import CheckpointInfo
from modules.onnx import ExecutionProvider, OnnxStableDiffusionPipeline

is_available = "olive" in sys.modules # Olive is not available if it is not loaded at startup.

def enable_olive_onchange():
    if shared.opts.onnx_enable_olive:
        if "olive" in sys.modules:
            log.info("You already have Olive installed. No additional installation is required.")
            return
        from installer import install
        install('olive-ai', 'Olive')
        log.info("Olive is installed. Please restart ui completely to load Olive.")
    else:
        from installer import pip
        global is_available
        if "olive" in sys.modules:
            del sys.modules["olive"]
        is_available = False
        if shared.opts.diffusers_pipeline == 'ONNX Stable Diffusion with Olive':
            shared.opts.diffusers_pipeline = 'ONNX Stable Diffusion'
        pip('uninstall olive-ai --yes --quiet', ignore=True, quiet=True)

submodels = ("text_encoder", "unet", "vae_encoder", "vae_decoder",)

EP_TO_NAME = {
    ExecutionProvider.CPU: "cpu",
    ExecutionProvider.DirectML: "gpu-dml",
    ExecutionProvider.CUDA: "gpu-?", # TODO
    ExecutionProvider.ROCm: "gpu-rocm",
    ExecutionProvider.OpenVINO: "?", # TODO
}

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

        if shared.opts.onnx_execution_provider == ExecutionProvider.ROCm:
            from olive.hardware.accelerator import AcceleratorLookup
            AcceleratorLookup.EXECUTION_PROVIDERS["gpu"].append(ExecutionProvider.ROCm)

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
                olive_config["passes"]["optimize"]["config"]["float16"] = shared.opts.onnx_olive_float16
                if (submodel == "unet" or "vae" in submodel) and (shared.opts.onnx_execution_provider == ExecutionProvider.CUDA or shared.opts.onnx_execution_provider == ExecutionProvider.ROCm):
                    olive_config["passes"]["optimize"]["config"]["optimization_options"]["group_norm_channels_last"] = True
                olive_config["engine"]["execution_providers"] = [shared.opts.onnx_execution_provider]

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
        except Exception as e:
            log.error(f"Failed to optimize model '{self.original_filename}'.")
            log.error(e) # for test.
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
