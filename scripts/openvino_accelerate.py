# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: AGPL-3.0

import cv2
import os
import torch
import time
import hashlib
import functools
import gradio as gr
import numpy as np

import modules
import modules.paths as paths
import modules.scripts as scripts

from modules import images, devices, extra_networks, masking, shared, sd_models_config
from modules.processing import (
    StableDiffusionProcessing, Processed, apply_overlay, apply_color_correction,
    get_fixed_seed, create_infotext, setup_color_correction,
    process_images
)
from modules.sd_models import CheckpointInfo, get_checkpoint_state_dict
from modules.shared import opts, state
from modules.ui_common import create_refresh_button
from modules.timer import Timer

from PIL import Image, ImageOps
from types import MappingProxyType
from typing import Optional

from openvino.frontend import FrontEndManager
from openvino.frontend.pytorch.fx_decoder import TorchFXPythonDecoder
from openvino.frontend.pytorch.torchdynamo import backend # noqa: F401
from openvino.frontend.pytorch.torchdynamo.partition import Partitioner
from openvino.runtime import Core, Type, PartialShape, serialize

from torch._dynamo.backends.common import fake_tensor_unsupported
from torch._dynamo.backends.registry import register_backend
from torch._inductor.compile_fx import compile_fx
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx import GraphModule
from torch.utils._pytree import tree_flatten

from hashlib import sha256

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    AutoencoderKL,
)

class ModelState:
    def __init__(self):
        self.recompile = 1
        self.device = "CPU"
        self.height = 512
        self.width = 512
        self.batch_size = 1
        self.mode = 0
        self.partition_id = 0
        self.model_hash = ""
        self.control_models = []
        self.lora_model = "None"
        self.custom_vae = "None"
        self.is_sdxl = False

model_state = ModelState()

DEFAULT_OPENVINO_PYTHON_CONFIG = MappingProxyType(
    {
        "use_python_fusion_cache": True,
        "allow_single_op_fusion": True,
    },
)

compiled_cache = {}
max_openvino_partitions = 0
partitioned_modules = {}

@register_backend
@fake_tensor_unsupported
def openvino_fx(subgraph, example_inputs):
    try:
        executor_parameters = None
        inputs_reversed = False
        if os.getenv("OPENVINO_TORCH_MODEL_CACHING") is not None:
            # Create a hash to be used for caching
            model_hash_str = sha256(subgraph.code.encode('utf-8')).hexdigest()
            if (len(model_state.control_models) > 0 and model_state.partition_id == 0): #scn_model != "None" and model_state.partition_id == 0):
                for cn_model in model_state.control_models:
                    model_hash_str = model_hash_str + "_" + cn_model

            if (model_state.lora_model != "None"):
                model_hash_str = model_hash_str + model_state.lora_model

            executor_parameters = {"model_hash_str": model_hash_str}

            # Check if the model was fully supported and already cached
            example_inputs.reverse()
            inputs_reversed = True
            maybe_fs_cached_name = cached_model_name(model_hash_str + "_fs", get_device(), example_inputs, cache_root_path())

            if os.path.isfile(maybe_fs_cached_name + ".xml") and os.path.isfile(maybe_fs_cached_name + ".bin"):
                if (len(model_state.control_models) > 0 and model_state.control_models[0] in maybe_fs_cached_name):
                    example_inputs_reordered = []
                    if (os.path.isfile(maybe_fs_cached_name + ".txt")):
                        f = open(maybe_fs_cached_name + ".txt", "r")
                        reordered_idx = []
                        for input_data in example_inputs:
                            shape = f.readline()
                            if (str(input_data.size()) != shape):
                                for idx1, input_data1 in enumerate(example_inputs):
                                    if (str(input_data1.size()).strip() == str(shape).strip()):
                                        if idx1 not in reordered_idx:
                                            reordered_idx.append(idx1)
                                            example_inputs_reordered.append(example_inputs[idx1])
                                            break
                        example_inputs = example_inputs_reordered

                    # Model is fully supported and already cached. Run the cached OV model directly.
                    compiled_model = openvino_compile_cached_model(maybe_fs_cached_name, *example_inputs)

                    def _call(*args):
                        if (len(model_state.control_models) > 0 and model_state.control_models[0] in maybe_fs_cached_name): #if (model_state.cn_model != "None" and model_state.cn_model in maybe_fs_cached_name):
                            args_reordered = []
                            if (os.path.isfile(maybe_fs_cached_name + ".txt")):
                                f = open(maybe_fs_cached_name + ".txt", "r")
                                reordered_idx = []
                                for input_data in args:
                                    shape = f.readline()
                                    if (str(input_data.size()) != shape):
                                        for idx1, input_data1 in enumerate(args):
                                            if (str(input_data1.size()).strip() == str(shape).strip()):
                                                if idx1 not in reordered_idx:
                                                    reordered_idx.append(idx1)
                                                    args_reordered.append(args[idx1])
                                                    break
                            args = args_reordered

                        res = execute_cached(compiled_model, *args)
                        model_state.partition_id = model_state.partition_id + 1
                        return res
                    return _call

                if (len(model_state.control_models) == 0):
                    compiled_model = openvino_compile_cached_model(maybe_fs_cached_name, *example_inputs)
                    def _call(*args):
                        res = execute_cached(compiled_model, *args)
                        model_state.partition_id = model_state.partition_id + 1
                        return res
                    return _call

        if inputs_reversed:
            example_inputs.reverse()
        model = make_fx(subgraph)(*example_inputs)
        for node in model.graph.nodes:
            if node.target == torch.ops.aten.mul_.Tensor:
                node.target = torch.ops.aten.mul.Tensor
        with torch.no_grad():
            model.eval()
        partitioner = Partitioner()
        compiled_model = partitioner.make_partitions(model)

        if executor_parameters is not None and 'model_hash_str' in executor_parameters:
            # Check if the model is fully supported.
            fully_supported = partitioner.check_fully_supported(compiled_model)
            if fully_supported:
                executor_parameters["model_hash_str"] += "_fs"

        def _call(*args):
            res = execute(compiled_model, *args, executor="openvino",
                          executor_parameters=executor_parameters, file_name=maybe_fs_cached_name)
            return res
        return _call
    except Exception as e:
        print(e)
        return compile_fx(subgraph, example_inputs)

def check_fully_supported(self, graph_module: GraphModule) -> bool:
    num_fused = 0
    for node in graph_module.graph.nodes:
        if node.op == "call_module" and "fused_" in node.name:
            num_fused += 1
        elif node.op != "placeholder" and node.op != "output":
            return False
    if num_fused == 1:
        return True
    return False

Partitioner.check_fully_supported = functools.partial(check_fully_supported, Partitioner)

def execute(
    gm: GraphModule,
    *args,
    executor: str = "openvino",
    executor_parameters: Optional[dict] = None,
    file_name = ""
):
    if executor == "openvino":
        return openvino_execute_partitioned(gm, *args, executor_parameters=executor_parameters, file_name=file_name)
    elif executor == "strictly_openvino":
        return openvino_execute(gm, *args, executor_parameters=executor_parameters, file_name=file_name)

    msg = "Received unexpected value for 'executor': {0}. Allowed values are: openvino, strictly_openvino.".format(executor)
    raise ValueError(msg)


class OpenVINOGraphModule(torch.nn.Module):
    def __init__(self, gm, partition_id, use_python_fusion_cache, model_hash_str: str = None, file_name=""):
        super().__init__()
        self.gm = gm
        self.partition_id = partition_id
        self.executor_parameters = {"use_python_fusion_cache": use_python_fusion_cache,
                                    "model_hash_str": model_hash_str}
        self.file_name = file_name
        self.perm_fallback = False

    def __call__(self, *args):
        #if self.perm_fallback:
        #    return self.gm(*args)

        #try:
        result = openvino_execute(self.gm, *args, executor_parameters=self.executor_parameters, partition_id=self.partition_id, file_name=self.file_name)
        #except Exception:
        #    self.perm_fallback = True
        #    return self.gm(*args)

        return result


def partition_graph(gm: GraphModule, use_python_fusion_cache: bool, model_hash_str: str = None, file_name=""):
    global max_openvino_partitions
    for node in gm.graph.nodes:
        if node.op == "call_module" and "fused_" in node.name:
            openvino_submodule = getattr(gm, node.name)
            gm.delete_submodule(node.target)
            gm.add_submodule(
                node.target,
                OpenVINOGraphModule(openvino_submodule, model_state.partition_id, use_python_fusion_cache,
                        model_hash_str=model_hash_str, file_name=file_name),
            )
            model_state.partition_id = model_state.partition_id + 1

    return gm


def openvino_execute(gm: GraphModule, *args, executor_parameters=None, partition_id, file_name=""):
    executor_parameters = executor_parameters or DEFAULT_OPENVINO_PYTHON_CONFIG

    use_cache = executor_parameters.get(
        "use_python_fusion_cache",
        DEFAULT_OPENVINO_PYTHON_CONFIG["use_python_fusion_cache"],
    )
    global compiled_cache

    model_hash_str = executor_parameters.get("model_hash_str", None)
    if model_hash_str is not None:
        model_hash_str = model_hash_str + str(partition_id)

    if use_cache and (partition_id in compiled_cache):
        compiled = compiled_cache[partition_id]
    else:
        if (len(model_state.control_models)> 0 and file_name is not None
                and os.path.isfile(file_name + ".xml") and os.path.isfile(file_name + ".bin") and model_state.control_models[0] in file_name):
            compiled = openvino_compile_cached_model(file_name, *args)
        else:
            compiled = openvino_compile(gm, *args, model_hash_str=model_hash_str, file_name=file_name)
        compiled_cache[partition_id] = compiled

    flat_args, _ = tree_flatten(args)
    ov_inputs = [a.detach().cpu().numpy() for a in flat_args]

    res = compiled(ov_inputs)

    results1 = [torch.from_numpy(res[out]) for out in compiled.outputs]
    if len(results1) == 1:
        return results1[0]
    return results1

def openvino_execute_partitioned(gm: GraphModule, *args, executor_parameters=None, file_name=""):
    executor_parameters = executor_parameters or DEFAULT_OPENVINO_PYTHON_CONFIG

    global partitioned_modules

    use_python_fusion_cache = executor_parameters.get(
        "use_python_fusion_cache",
        DEFAULT_OPENVINO_PYTHON_CONFIG["use_python_fusion_cache"],
    )
    model_hash_str = executor_parameters.get("model_hash_str", None)

    signature = str(id(gm))
    for idx, input_data in enumerate(args):
        if isinstance(input_data, torch.Tensor):
            signature = signature + "_" + str(idx) + ":" + str(input_data.type())[6:] + ":" + str(input_data.size())[11:-1].replace(" ", "")
        else:
            signature = signature + "_" + str(idx) + ":" + type(input_data).__name__ + ":val(" + str(input_data) + ")"

    if signature not in partitioned_modules:
        partitioned_modules[signature] = partition_graph(gm, use_python_fusion_cache=use_python_fusion_cache,
                                                         model_hash_str=model_hash_str, file_name=file_name)

    return partitioned_modules[signature](*args)

def execute_cached(compiled_model, *args):
    flat_args, _ = tree_flatten(args)
    ov_inputs = [a.detach().cpu().numpy() for a in flat_args]

    if (len(model_state.control_models) == 0):
        ov_inputs.reverse()

    res = compiled_model(ov_inputs)
    result = [torch.from_numpy(res[out]) for out in compiled_model.outputs]
    return result

def cached_model_name(model_hash_str, device, args, cache_root, reversed = False):
    if model_hash_str is None:
        return None

    model_cache_dir = cache_root + "/model/"

    try:
        os.makedirs(model_cache_dir, exist_ok=True)
        file_name = model_cache_dir + model_hash_str + "_" + device
    except OSError as error:
        print("Cache directory ", cache_root, " cannot be created. Model caching is disabled. Error: ", error)
        return None

    inputs_str = ""
    for input_data in args:
        if reversed:
            inputs_str = "_" + str(input_data.type()) + str(input_data.size())[11:-1].replace(" ", "") + inputs_str
        else:
            inputs_str += "_" + str(input_data.type()) + str(input_data.size())[11:-1].replace(" ", "")
    inputs_str = sha256(inputs_str.encode('utf-8')).hexdigest()
    file_name += inputs_str

    return file_name

def cache_root_path():
    cache_root = "./cache/"
    if os.getenv("OPENVINO_TORCH_CACHE_DIR") is not None:
        cache_root = os.getenv("OPENVINO_TORCH_CACHE_DIR")
    return cache_root

def get_device():
    device = "CPU"
    core = Core()
    if os.getenv("OPENVINO_TORCH_BACKEND_DEVICE") is not None:
        device = os.getenv("OPENVINO_TORCH_BACKEND_DEVICE")
        assert device in core.available_devices, "Specified device " + device + " is not in the list of OpenVINO Available Devices"

    return device

def openvino_compile_cached_model(cached_model_path, *example_inputs):
    core = Core()
    om = core.read_model(cached_model_path + ".xml")

    dtype_mapping = {
        torch.float32: Type.f32,
        torch.float64: Type.f64,
        torch.float16: Type.f16,
        torch.int64: Type.i64,
        torch.int32: Type.i32,
        torch.uint8: Type.u8,
        torch.int8: Type.i8,
        torch.bool: Type.boolean
    }

    for idx, input_data in enumerate(example_inputs):
        om.inputs[idx].get_node().set_element_type(dtype_mapping[input_data.dtype])
        om.inputs[idx].get_node().set_partial_shape(PartialShape(list(input_data.shape)))
    om.validate_nodes_and_infer_types()

    core.set_property({'CACHE_DIR': cache_root_path() + '/blob'})

    compiled_model = core.compile_model(om, get_device())

    return compiled_model

def openvino_compile(gm: GraphModule, *args, model_hash_str: str = None, file_name=""):
    core = Core()

    device = get_device()
    cache_root = cache_root_path()

    if (file_name is not None and os.path.isfile(file_name + ".xml") and os.path.isfile(file_name + ".bin")):
        om = core.read_model(file_name + ".xml")
    else:
        fe_manager = FrontEndManager()
        fe = fe_manager.load_by_framework("pytorch")

        input_shapes = []
        input_types = []
        for input_data in args:
            input_types.append(input_data.type())
            input_shapes.append(input_data.size())

        decoder = TorchFXPythonDecoder(gm, gm, input_shapes=input_shapes, input_types=input_types)

        im = fe.load(decoder)

        om = fe.convert(im)

        if (file_name is not None):
            serialize(om, file_name + ".xml", file_name + ".bin")
            if (len(model_state.control_models) > 0):
                f = open(file_name + ".txt", "w")
                for input_data in args:
                    f.write(str(input_data.size()))
                    f.write("\n")
                f.close()

    dtype_mapping = {
        torch.float32: Type.f32,
        torch.float64: Type.f64,
        torch.float16: Type.f16,
        torch.int64: Type.i64,
        torch.int32: Type.i32,
        torch.uint8: Type.u8,
        torch.int8: Type.i8,
        torch.bool: Type.boolean
    }

    for idx, input_data in enumerate(args):
        om.inputs[idx].get_node().set_element_type(dtype_mapping[input_data.dtype])
        om.inputs[idx].get_node().set_partial_shape(PartialShape(list(input_data.shape)))
    om.validate_nodes_and_infer_types()

    if model_hash_str is not None:
        core.set_property({'CACHE_DIR': cache_root + '/blob'})

    compiled = core.compile_model(om, device)
    return compiled

def openvino_clear_caches():
    global partitioned_modules
    global compiled_cache

    compiled_cache.clear()
    partitioned_modules.clear()

def sd_diffusers_model(self):
    import modules.sd_models
    return modules.sd_models.model_data.get_sd_model()

def cond_stage_key(self):
    return None

shared.sd_diffusers_model = sd_diffusers_model
#refiner model
shared.sd_refiner_model = None

def set_scheduler(sd_model, sampler_name):
    if (sampler_name == "Euler a"):
        sd_model.scheduler = EulerAncestralDiscreteScheduler.from_config(sd_model.scheduler.config)
    elif (sampler_name == "Euler"):
        sd_model.scheduler = EulerDiscreteScheduler.from_config(sd_model.scheduler.config)
    elif (sampler_name == "LMS"):
        sd_model.scheduler = LMSDiscreteScheduler.from_config(sd_model.scheduler.config)
    elif (sampler_name == "Heun"):
        sd_model.scheduler = HeunDiscreteScheduler.from_config(sd_model.scheduler.config)
    elif (sampler_name == "DPM++ 2M"):
        sd_model.scheduler = DPMSolverMultistepScheduler.from_config(sd_model.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=False)
    elif (sampler_name == "LMS Karras"):
        sd_model.scheduler = LMSDiscreteScheduler.from_config(sd_model.scheduler.config, use_karras_sigmas=True)
    elif (sampler_name == "DPM++ 2M Karras"):
        sd_model.scheduler = DPMSolverMultistepScheduler.from_config(sd_model.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True)
    elif (sampler_name == "DDIM"):
        sd_model.scheduler = DDIMScheduler.from_config(sd_model.scheduler.config)
    elif (sampler_name == "PLMS"):
        sd_model.scheduler = PNDMScheduler.from_config(sd_model.scheduler.config)
    else:
        sd_model.scheduler = EulerAncestralDiscreteScheduler.from_config(sd_model.scheduler.config)

    return sd_model.scheduler

def get_diffusers_sd_model(model_config, vae_config, sampler_name, enable_caching, openvino_device, mode, is_xl_ckpt, refiner_checkpoint_name, refiner_steps):
    if (model_state.recompile == 1):
        model_state.partition_id = 0
        os.environ["INFERENCE_PRECISION_HINT"] = "None"
        torch._dynamo.reset()
        openvino_clear_caches()
        curr_dir_path = os.getcwd()
        checkpoint_name = shared.opts.sd_model_checkpoint.split(" ")[0]
        checkpoint_path = os.path.join(curr_dir_path, 'models', 'Stable-diffusion', checkpoint_name)
        checkpoint_info = CheckpointInfo(checkpoint_path)
        timer = Timer()
        state_dict = get_checkpoint_state_dict(checkpoint_info, timer)
        checkpoint_config = sd_models_config.find_checkpoint_config(state_dict, checkpoint_info)
        print("OpenVINO Script:  created model from config : " + checkpoint_config)
        local_config_file = checkpoint_config
        if model_config != "None":
            local_config_file = os.path.join(curr_dir_path, 'configs', model_config)

        if(is_xl_ckpt):
            sd_model = StableDiffusionXLPipeline.from_single_file(checkpoint_path, local_config_file=local_config_file, load_safety_checker=False, use_safetensors=True)
            if (mode == 1):
                sd_model = StableDiffusionXLImg2ImgPipeline.from_single_file(checkpoint_path, local_config_file=checkpoint_config, load_safety_checker=False, use_safetensors=True)
            elif (mode == 2):
                sd_model = StableDiffusionXLInpaintPipeline.from_single_file(checkpoint_path, local_config_file=checkpoint_config, load_safety_checker=False, use_safetensors=True)
        else:
            sd_model = StableDiffusionPipeline.from_single_file(checkpoint_path, local_config_file=checkpoint_config, load_safety_checker=False, torch_dtype=torch.float32)
            if (mode == 1):
                sd_model = StableDiffusionImg2ImgPipeline(**sd_model.components)
            elif (mode == 2):
                sd_model = StableDiffusionInpaintPipeline(**sd_model.components)
            elif (mode == 3):
                if (len(model_state.control_models) > 1):
                    controlnet = []
                    for cn_model in model_state.control_models:
                        controlnet.append(ControlNetModel.from_pretrained("lllyasviel/" + cn_model))
                else:
                    controlnet = ControlNetModel.from_pretrained("lllyasviel/" + model_state.control_models[0])
                sd_model = StableDiffusionControlNetPipeline(**sd_model.components, controlnet=controlnet)
                sd_model.controlnet = torch.compile(sd_model.controlnet, backend="openvino_fx")


        if ('lora' in modules.extra_networks.extra_network_registry):
            import lora
            if lora.loaded_loras:
                lora_model = lora.loaded_loras[0]
                sd_model.load_lora_weights(os.path.join(os.getcwd(), "models", "Lora"), weight_name=lora_model.name + ".safetensors")

        checkpoint_info = CheckpointInfo(checkpoint_path)
        os.environ["INFERENCE_PRECISION_HINT"] = "None"
        sd_model.sd_checkpoint_info = checkpoint_info
        sd_model.sd_model_hash = checkpoint_info.calculate_shorthash()
        sd_model.safety_checker = None
        sd_model.cond_stage_key = functools.partial(cond_stage_key, shared.sd_model)
        sd_model.scheduler = set_scheduler(sd_model, sampler_name)
        ## UNET
        sd_model.unet = torch.compile(sd_model.unet,  backend="openvino_fx")
        ## VAE
        if vae_config == "Disable-VAE-Acceleration":
            sd_model.vae.decode = sd_model.vae.decode
        elif vae_config == "None":
            #os.environ["INFERENCE_PRECISION_HINT"] = "f32"
            sd_model.vae.decode = torch.compile(sd_model.vae.decode, backend="openvino_fx")
        else:
            vae_path = os.path.join(curr_dir_path, 'models', 'VAE', vae_config)
            print("OpenVINO Script:  loading vae from : " + vae_path)
            sd_model.vae = AutoencoderKL.from_single_file(vae_path, local_files_only=True)
            #os.environ["INFERENCE_PRECISION_HINT"] = "f32"
            sd_model.vae = torch.compile(sd_model.vae,  backend="openvino_fx")
            print("VAE INFER PRECISION:" + os.getenv("INFERENCE_PRECISION_HINT"))
        shared.sd_diffusers_model = sd_model
        del sd_model
    return shared.sd_diffusers_model

##get refiner model

def get_diffusers_sd_refiner_model(model_config, vae_config, sampler_name, enable_caching, openvino_device, mode, is_xl_ckpt, refiner_checkpoint_name, refiner_steps):
    if (model_state.recompile == 1):
        os.environ["INFERENCE_PRECISION_HINT"] = "None"
        curr_dir_path = os.getcwd()
        if refiner_checkpoint_name != "None":
            refiner_checkpoint_path= os.path.join(curr_dir_path, 'models', 'Stable-diffusion', refiner_checkpoint_name)
            refiner_checkpoint_info = CheckpointInfo(refiner_checkpoint_path)
            refiner_model = StableDiffusionXLImg2ImgPipeline.from_single_file(refiner_checkpoint_path, load_safety_checker=False, use_safetensors=True)
            print("OpenVINO Script: refiner model loaded from" + refiner_checkpoint_path)
            refiner_model.sd_checkpoint_info = refiner_checkpoint_info
            refiner_model.sd_model_hash = refiner_checkpoint_info.calculate_shorthash()
            ## UNET
            refiner_model.unet = torch.compile(refiner_model.unet,  backend="openvino_fx")
            print("OpenVINO Script: refiner model compiled")
        shared.sd_refiner_model = refiner_model
        del refiner_model
    return shared.sd_refiner_model


def init_new(self, all_prompts, all_seeds, all_subseeds):
    crop_region = None

    image_mask = self.image_mask

    if image_mask is not None:
        image_mask = image_mask.convert('L')

        if self.inpainting_mask_invert:
            image_mask = ImageOps.invert(image_mask)

        if self.mask_blur_x > 0:
            np_mask = np.array(image_mask)
            kernel_size = 2 * int(4 * self.mask_blur_x + 0.5) + 1
            np_mask = cv2.GaussianBlur(np_mask, (kernel_size, 1), self.mask_blur_x)
            image_mask = Image.fromarray(np_mask)

        if self.mask_blur_y > 0:
            np_mask = np.array(image_mask)
            kernel_size = 2 * int(4 * self.mask_blur_y + 0.5) + 1
            np_mask = cv2.GaussianBlur(np_mask, (1, kernel_size), self.mask_blur_y)
            image_mask = Image.fromarray(np_mask)

        if self.inpaint_full_res:
            self.mask_for_overlay = image_mask
            mask = image_mask.convert('L')
            crop_region = masking.get_crop_region(np.array(mask), self.inpaint_full_res_padding)
            crop_region = masking.expand_crop_region(crop_region, self.width, self.height, mask.width, mask.height)
            x1, y1, x2, y2 = crop_region

            mask = mask.crop(crop_region)
            image_mask = images.resize_image(2, mask, self.width, self.height)
            self.paste_to = (x1, y1, x2-x1, y2-y1)
        else:
            image_mask = images.resize_image(self.resize_mode, image_mask, self.width, self.height)
            np_mask = np.array(image_mask)
            np_mask = np.clip((np_mask.astype(np.float32)) * 2, 0, 255).astype(np.uint8)
            self.mask_for_overlay = Image.fromarray(np_mask)

        self.overlay_images = []

    latent_mask = self.latent_mask if self.latent_mask is not None else image_mask

    add_color_corrections = opts.img2img_color_correction and self.color_corrections is None
    if add_color_corrections:
        self.color_corrections = []
    imgs = []
    for img in self.init_images:
        # Save init image
        if opts.save_init_img:
            self.init_img_hash = hashlib.md5(img.tobytes()).hexdigest()
            images.save_image(img, path=opts.outdir_init_images, basename=None, forced_filename=self.init_img_hash, save_to_dirs=False)

        image = images.flatten(img, opts.img2img_background_color)

        if crop_region is None and self.resize_mode != 3:
            image = images.resize_image(self.resize_mode, image, self.width, self.height)

        if image_mask is not None:
            image_masked = Image.new('RGBa', (image.width, image.height))
            image_masked.paste(image.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(self.mask_for_overlay.convert('L')))
            self.mask = image_mask
            self.overlay_images.append(image_masked.convert('RGBA'))

        # crop_region is not None if we are doing inpaint full res
        if crop_region is not None:
            image = image.crop(crop_region)
            image = images.resize_image(2, image, self.width, self.height)

        self.init_images = image
        if image_mask is not None:
            if self.inpainting_fill != 1:
                image = masking.fill(image, latent_mask)

        if add_color_corrections:
            self.color_corrections.append(setup_color_correction(image))

        image = np.array(image).astype(np.float32) / 255.0
        image = np.moveaxis(image, 2, 0)

        imgs.append(image)

    if len(imgs) == 1:
        if self.overlay_images is not None:
            self.overlay_images = self.overlay_images * self.batch_size

        if self.color_corrections is not None and len(self.color_corrections) == 1:
            self.color_corrections = self.color_corrections * self.batch_size

    elif len(imgs) <= self.batch_size:
        self.batch_size = len(imgs)
    else:
        raise RuntimeError(f"bad number of images passed: {len(imgs)}; expecting {self.batch_size} or less")

def process_images_openvino(p: StableDiffusionProcessing, model_config, vae_config, sampler_name, enable_caching, openvino_device, mode, is_xl_ckpt, refiner_checkpoint_name, refiner_steps) -> Processed:
    """this is the main loop that both txt2img and img2img use; it calls func_init once inside all the scopes and func_sample once per batch"""
    if (mode == 0 and p.enable_hr):
        return process_images(p)

    if type(p.prompt) == list:
        assert(len(p.prompt) > 0)
    else:
        assert p.prompt is not None

    devices.torch_gc()

    seed = get_fixed_seed(p.seed)
    subseed = get_fixed_seed(p.subseed)

    comments = {}
    custom_inputs = {}

    p.setup_prompts()

    if type(seed) == list:
        p.all_seeds = seed
    else:
        p.all_seeds = [int(seed) + (x if p.subseed_strength == 0 else 0) for x in range(len(p.all_prompts))]

    if type(subseed) == list:
        p.all_subseeds = subseed
    else:
        p.all_subseeds = [int(subseed) + x for x in range(len(p.all_prompts))]

    def infotext(iteration=0, position_in_batch=0):
        return create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, comments, iteration, position_in_batch)

    if p.scripts is not None:
        p.scripts.process(p)

    cn_model="None"
    control_models = []
    control_images = []

    for key in p.extra_generation_params.keys():
        if key.startswith('ControlNet'):
            control_images_cn = []
            cn_params = p.extra_generation_params[key]
            cn_param_elements = [part.strip() for part in cn_params.split(', ')]
            for element in cn_param_elements:
                if (element.split(':')[0] == "Model"):
                    cn_model = (element.split(':')[1]).split(' ')[1]

            if (cn_model != "None"):
                control_models.append(cn_model)
                control_res = Processed(
                    p,
                    images_list=control_images_cn,
                )
                p.scripts.postprocess(p, control_res)
                mode = 3
                for cn_image in control_images_cn:
                    control_images.append(cn_image)

    model_state.control_models = control_models

    infotexts = []
    output_images = []

    with torch.no_grad():
        with devices.autocast():
            p.init(p.all_prompts, p.all_seeds, p.all_subseeds)

        if state.job_count == -1:
            state.job_count = p.n_iter

        extra_network_data = None
        for n in range(p.n_iter):
            p.iteration = n

            if state.skipped:
                state.skipped = False

            if state.interrupted:
                break

            p.prompts = p.all_prompts[n * p.batch_size:(n + 1) * p.batch_size]
            p.negative_prompts = p.all_negative_prompts[n * p.batch_size:(n + 1) * p.batch_size]
            p.seeds = p.all_seeds[n * p.batch_size:(n + 1) * p.batch_size]
            p.subseeds = p.all_subseeds[n * p.batch_size:(n + 1) * p.batch_size]

            if p.scripts is not None:
                p.scripts.before_process_batch(p, batch_number=n, prompts=p.prompts, seeds=p.seeds, subseeds=p.subseeds)

            if len(p.prompts) == 0:
                break

            extra_network_data = p.parse_extra_network_prompts()

            if not p.disable_extra_networks:
                with devices.autocast():
                    extra_networks.activate(p, p.extra_network_data)

            lora_model_name = "None"
            if ('lora' in modules.extra_networks.extra_network_registry):
                import lora
                if lora.loaded_loras:
                    lora_model = lora.loaded_loras[0]
                    lora_model_name = lora_model.name
                    custom_inputs.update(cross_attention_kwargs={"scale" : lora_model.te_multiplier})

            if (model_state.height != p.height or model_state.width != p.width or model_state.batch_size != p.batch_size or model_state.lora_model != lora_model_name
                    or model_state.mode != mode or model_state.model_hash != shared.sd_model.sd_model_hash or model_state.cn_model != cn_model):
                model_state.recompile = 1
                model_state.height = p.height
                model_state.width = p.width
                model_state.batch_size = p.batch_size
                model_state.mode = mode
                model_state.cn_model = cn_model
                model_state.model_hash = shared.sd_model.sd_model_hash
                model_state.lora_model = lora_model_name

            shared.sd_diffusers_model = get_diffusers_sd_model(model_config, vae_config, sampler_name, enable_caching, openvino_device, mode, is_xl_ckpt, refiner_checkpoint_name, refiner_steps)
            shared.sd_diffusers_model.scheduler = set_scheduler(shared.sd_diffusers_model, sampler_name)

            if refiner_checkpoint_name != "None":
                shared.sd_refiner_model = get_diffusers_sd_refiner_model(model_config, vae_config, sampler_name, enable_caching, openvino_device, mode, is_xl_ckpt, refiner_checkpoint_name, refiner_steps)
                shared.sd_refiner_model.scheduler = set_scheduler(shared.sd_refiner_model, sampler_name)
                print("refiner used: " + refiner_checkpoint_name)


            if p.scripts is not None:
                p.scripts.process_batch(p, batch_number=n, prompts=p.prompts, seeds=p.seeds, subseeds=p.subseeds)

            # params.txt should be saved after scripts.process_batch, since the
            # infotext could be modified by that callback
            # Example: a wildcard processed by process_batch sets an extra model
            # strength, which is saved as "Model Strength: 1.0" in the infotext
            if n == 0:
                with open(os.path.join(paths.data_path, "params.txt"), "w", encoding="utf8") as file:
                    file.write(create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, comments=[], position_in_batch=0 % p.batch_size, iteration=0 // p.batch_size))

            if p.n_iter > 1:
                shared.state.job = f"Batch {n+1} out of {p.n_iter}"

            generator = [torch.Generator(device="cpu").manual_seed(s) for s in p.seeds]

            time_stamps = []

            def callback(iter, t, latents):
                time_stamps.append(time.time()) # noqa: B023

            time_stamps.append(time.time())

            if (mode == 0):
                custom_inputs.update({
                    'width': p.width,
                    'height': p.height,
                })
            elif (mode == 1):
                custom_inputs.update({
                    'image': p.init_images,
                    'strength':p.denoising_strength,
                })
            elif (mode == 2):
                custom_inputs.update({
                    'image': p.init_images,
                    'strength':p.denoising_strength,
                    'mask_image': p.mask,
                })
            elif (mode == 3):
                 custom_inputs.update({
                    'image': control_images,
                    'width': p.width,
                    'height': p.height,
                })

            if refiner_checkpoint_name != "None":
                base_output_type = "latent"
            else:
                base_output_type = "np"
            output = shared.sd_diffusers_model(
                    prompt=p.prompts,
                    negative_prompt=p.negative_prompts,
                    num_inference_steps=p.steps,
                    guidance_scale=p.cfg_scale,
                    generator=generator,
                    output_type=base_output_type,
                    callback = callback,
                    callback_steps = 1,
                    **custom_inputs
            )

            if refiner_checkpoint_name != "None":
                refiner_output = shared.sd_refiner_model(
                        prompt=p.prompts,
                        negative_prompt=p.negative_prompts,
                        num_inference_steps=refiner_steps,
                        image=output.images[0][None, :],
                        output_type="np"
                )
                print("refiner steps " + str(refiner_steps))


            model_state.recompile = 0

            warmup_duration = time_stamps[1] - time_stamps[0]
            generation_rate = (p.steps - 1) / (time_stamps[-1] - time_stamps[1])

            if refiner_checkpoint_name != "None":
                x_samples_ddim = refiner_output.images
            else:
                x_samples_ddim = output.images

            for i, x_sample in enumerate(x_samples_ddim):
                p.batch_index = i

                x_sample = (255. * x_sample).astype(np.uint8)

                if p.restore_faces:
                    if opts.save and not p.do_not_save_samples and opts.save_images_before_face_restoration:
                        images.save_image(Image.fromarray(x_sample), p.outpath_samples, "", p.seeds[i], p.prompts[i], opts.samples_format, info=infotext(n, i), p=p, suffix="-before-face-restoration")

                    devices.torch_gc()

                    x_sample = modules.face_restoration.restore_faces(x_sample)
                    devices.torch_gc()

                image = Image.fromarray(x_sample)

                if p.scripts is not None:
                    pp = scripts.PostprocessImageArgs(image)
                    p.scripts.postprocess_image(p, pp)
                    image = pp.image

                if p.color_corrections is not None and i < len(p.color_corrections):
                    if opts.save and not p.do_not_save_samples and opts.save_images_before_color_correction:
                        image_without_cc = apply_overlay(image, p.paste_to, i, p.overlay_images)
                        images.save_image(image_without_cc, p.outpath_samples, "", p.seeds[i], p.prompts[i], opts.samples_format, info=infotext(n, i), p=p, suffix="-before-color-correction")
                    image = apply_color_correction(p.color_corrections[i], image)

                image = apply_overlay(image, p.paste_to, i, p.overlay_images)

                if opts.samples_save and not p.do_not_save_samples:
                    images.save_image(image, p.outpath_samples, "", p.seeds[i], p.prompts[i], opts.samples_format, info=infotext(n, i), p=p)

                text = infotext(n, i)
                infotexts.append(text)
                if opts.enable_pnginfo:
                    image.info["parameters"] = text
                output_images.append(image)
                for cn_image in control_images:
                    output_images.append(cn_image)

                if hasattr(p, 'mask_for_overlay') and p.mask_for_overlay and any([opts.save_mask, opts.save_mask_composite, opts.return_mask, opts.return_mask_composite]):
                    image_mask = p.mask_for_overlay.convert('RGB')
                    image_mask_composite = Image.composite(image.convert('RGBA').convert('RGBa'), Image.new('RGBa', image.size), images.resize_image(2, p.mask_for_overlay, image.width, image.height).convert('L')).convert('RGBA')

                    if opts.save_mask:
                        images.save_image(image_mask, p.outpath_samples, "", p.seeds[i], p.prompts[i], opts.samples_format, info=infotext(n, i), p=p, suffix="-mask")

                    if opts.save_mask_composite:
                        images.save_image(image_mask_composite, p.outpath_samples, "", p.seeds[i], p.prompts[i], opts.samples_format, info=infotext(n, i), p=p, suffix="-mask-composite")

                    if opts.return_mask:
                        output_images.append(image_mask)

                    if opts.return_mask_composite:
                        output_images.append(image_mask_composite)

            del x_samples_ddim

            devices.torch_gc()

            state.nextjob()

        p.color_corrections = None

        index_of_first_image = 0
        unwanted_grid_because_of_img_count = len(output_images) < 2 and opts.grid_only_if_multiple
        if (opts.return_grid or opts.grid_save) and not p.do_not_save_grid and not unwanted_grid_because_of_img_count:
            grid = images.image_grid(output_images, p.batch_size)

            if opts.return_grid:
                text = infotext()
                infotexts.insert(0, text)
                if opts.enable_pnginfo:
                    grid.info["parameters"] = text
                output_images.insert(0, grid)
                index_of_first_image = 1

            if opts.grid_save:
                images.save_image(grid, p.outpath_grids, "grid", p.all_seeds[0], p.all_prompts[0], opts.grid_format, info=infotext(), short_filename=not opts.grid_extended_filename, p=p, grid=True)

    if not p.disable_extra_networks and extra_network_data:
        extra_networks.deactivate(p, p.extra_network_data)

    devices.torch_gc()

    res = Processed(
        p,
        images_list=output_images,
        seed=p.all_seeds[0],
        info=infotext(),
        comments="".join(f"{comment}\n" for comment in comments),
        subseed=p.all_subseeds[0],
        index_of_first_image=index_of_first_image,
        infotexts=infotexts,
    )

    res.info = res.info + ", Warm up time: " + str(round(warmup_duration, 2)) + " secs "

    if (generation_rate >= 1.0):
        res.info = res.info + ", Performance: " + str(round(generation_rate, 2)) + " it/s "
    else:
        res.info = res.info + ", Performance: " + str(round(1/generation_rate, 2)) + " s/it "


    if p.scripts is not None:
        p.scripts.postprocess(p, res)

    return res

class Script(scripts.Script):
    def title(self):
        return "Accelerate with OpenVINO"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        core = Core()

        def get_config_list():
            config_dir_list = os.listdir(os.path.join(os.getcwd(), 'configs'))
            config_list = []
            config_list.append("None")
            for file in config_dir_list:
                if file.endswith('.yaml'):
                    config_list.append(file)
            return config_list
        def get_vae_list():
            vae_dir_list = os.listdir(os.path.join(os.getcwd(), 'models', 'VAE'))
            vae_list = []
            vae_list.append("None")
            vae_list.append("Disable-VAE-Acceleration")
            for file in vae_dir_list:
                if file.endswith('.safetensors') or file.endswith('.ckpt') or file.endswith('.pt'):
                    vae_list.append(file)
            return vae_list
        def get_refiner_list():
            refiner_dir_list = os.listdir(os.path.join(os.getcwd(), 'models', 'Stable-diffusion'))
            refiner_list = []
            refiner_list.append("None")
            for file in refiner_dir_list:
                if file.endswith('.safetensors') or file.endswith('.ckpt') or file.endswith('.pt'):
                    refiner_list.append(file)
            return refiner_list


        with gr.Group():
            with gr.Row():
                with gr.Row():
                    model_config = gr.Dropdown(label="Select a local config for the model from the configs directory of the webui root", choices=get_config_list(), value="None", visible=True)
                    create_refresh_button(model_config, get_config_list, lambda: {"choices": get_config_list()},"refresh_model_config")
                with gr.Row():
                    vae_config = gr.Dropdown(label="Custom VAE", choices=get_vae_list(), value="None", visible=True)
                    create_refresh_button(vae_config, get_vae_list, lambda: {"choices": get_vae_list()},"refresh_vae_directory")
        openvino_device = gr.Dropdown(label="Select a device", choices=list(core.available_devices), value=model_state.device)
        is_xl_ckpt= gr.Checkbox(label="Loaded checkpoint is a SDXL checkpoint", value=False)
        with gr.Row():
                refiner_checkpoint_name = gr.Dropdown(label="Model", choices=get_refiner_list(), value="None")
                refiner_steps = gr.Slider(minimum=0, maximum=100, step=4, label='Refiner steps:', value=20)
        override_sampler = gr.Checkbox(label="Override the sampling selection from the main UI (Recommended as only below sampling methods have been validated for OpenVINO)", value=True)
        sampler_name = gr.Radio(label="Select a sampling method", choices=["Euler a", "Euler", "LMS", "Heun", "DPM++ 2M", "LMS Karras", "DPM++ 2M Karras", "DDIM", "PLMS"], value="Euler a")
        enable_caching = gr.Checkbox(label="Cache the compiled models on disk for faster model load in subsequent launches (Recommended)", value=True, elem_id=self.elem_id("enable_caching"))
        warmup_status = gr.Textbox(label="Device", interactive=False, visible=False)
        vae_status = gr.Textbox(label="VAE", interactive=False, visible=False)
        gr.Markdown(
        """
        ###
        ### Note:
        - First inference involves compilation of the model for best performance.
        Since compilation happens only on the first run, the first inference (or warm up inference) will be slower than subsequent inferences.
        - For accurate performance measurements, it is recommended to exclude this slower first inference, as it doesn't reflect normal running time.
        - Model is recompiled when resolution, batchsize, device, or samplers like DPM++ or Karras are changed.
        After recompiling, later inferences will reuse the newly compiled model and achieve faster running times.
        So it's normal for the first inference after a settings change to be slower, while subsequent inferences use the optimized compiled model and run faster.
        """)
        def device_change(choice):
            if (model_state.device == choice):
                return gr.update(value="Device selected is " + choice, visible=True)
            else:
                model_state.device = choice
                model_state.recompile = 1
                return gr.update(value="Device changed to " + choice + ". Model will be re-compiled", visible=True)
        openvino_device.change(device_change, openvino_device, warmup_status)
        def vae_change(choice):
            if (model_state.custom_vae == choice):
                return gr.update(value="Custom_VAE selected is " + choice, visible=True)
            else:
                model_state.custom_vae = choice
                model_state.recompile = 1
                return gr.update(value="Custom VAE changed to " + choice + ". Model will be re-compiled", visible=True)
        vae_config.change(vae_change, vae_config, vae_status)
        return [model_config, vae_config, openvino_device, override_sampler, sampler_name, enable_caching, is_xl_ckpt, refiner_checkpoint_name, refiner_steps]

    def run(self, p, model_config, vae_config, openvino_device, override_sampler, sampler_name, enable_caching, is_xl_ckpt, refiner_checkpoint_name, refiner_steps):
        model_state.partition_id = 0
        os.environ["OPENVINO_TORCH_BACKEND_DEVICE"] = str(openvino_device)

        if enable_caching:
            os.environ["OPENVINO_TORCH_MODEL_CACHING"] = "1"

        if override_sampler:
            p.sampler_name = sampler_name
        else:
            supported_samplers = ["Euler a", "Euler", "LMS", "Heun", "DPM++ 2M", "LMS Karras", "DPM++ 2M Karras", "DDIM", "PLMS"]
            if (p.sampler_name not in supported_samplers):
                p.sampler_name = "Euler a"

        # mode can be 0, 1, 2 corresponding to txt2img, img2img, inpaint respectively
        mode = 0
        if self.is_txt2img:
            mode = 0
            processed = process_images_openvino(p, model_config, vae_config, p.sampler_name, enable_caching, openvino_device, mode, is_xl_ckpt, refiner_checkpoint_name, refiner_steps)
        else:
            if p.image_mask is None:
                mode = 1
            else:
                mode = 2
            p.init = functools.partial(init_new, p)
            processed = process_images_openvino(p, model_config, vae_config, p.sampler_name, enable_caching, openvino_device, mode, is_xl_ckpt, refiner_checkpoint_name, refiner_steps)
        return processed


