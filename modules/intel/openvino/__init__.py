import os
import torch
from openvino.frontend.pytorch.torchdynamo.execute import execute, partitioned_modules, compiled_cache
from openvino.frontend.pytorch.torchdynamo.partition import Partitioner
from openvino.runtime import Core, Type, PartialShape
from torch._dynamo.backends.common import fake_tensor_unsupported
from torch._dynamo.backends.registry import register_backend
from torch.fx.experimental.proxy_tensor import make_fx
from torch._inductor.compile_fx import compile_fx
from torch.utils._pytree import tree_flatten
from hashlib import sha256
import functools
from modules import shared, devices

def openvino_clear_caches():
    global partitioned_modules
    global compiled_cache

    compiled_cache.clear()
    partitioned_modules.clear()

def get_device():
    core = Core()
    if os.getenv("OPENVINO_TORCH_BACKEND_DEVICE") is not None:
        device = os.getenv("OPENVINO_TORCH_BACKEND_DEVICE")
    elif any(openvino_cpu in cpu_module.lower() for cpu_module in shared.cmd_opts.use_cpu for openvino_cpu in ["openvino", "all"]):
        device = "CPU"
    elif shared.cmd_opts.device_id is not None:
        device = f"GPU.{shared.cmd_opts.device_id}"
    elif "GPU" in core.available_devices:
        device = "GPU"
    elif "GPU.1" in core.available_devices:
        device = "GPU.1"
    elif "GPU.0" in core.available_devices:
        device = "GPU.0"
    else:
        device = "CPU"
        shared.log.warning("OpenVINO: No compatible GPU detected!")
    os.environ.setdefault('OPENVINO_TORCH_BACKEND_DEVICE', device)
    shared.log.debug(f"OpenVINO Device: {device}")
    if shared.opts.cuda_compile_errors and device not in core.available_devices:
        shared.log.error(f"OpenVINO: Specified device {device} is not in the list of OpenVINO Available Devices")
    assert device in core.available_devices, f"OpenVINO: Specified device {device} is not in the list of OpenVINO Available Devices"

    return device

def cache_root_path():
    cache_root = "./cache/"
    if os.getenv("OPENVINO_TORCH_CACHE_DIR") is not None:
        cache_root = os.getenv("OPENVINO_TORCH_CACHE_DIR")
    return cache_root

def cached_model_name(model_hash_str, device, args, cache_root, reversed = False):
    if model_hash_str is None:
        return None

    model_cache_dir = cache_root + "/model/"

    try:
        os.makedirs(model_cache_dir, exist_ok=True)
        file_name = model_cache_dir + model_hash_str + "_" + device
    except OSError as error:
        shared.log.error(f"Cache directory {cache_root} cannot be created. Model caching is disabled. Error: {error}")
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

def execute_cached(compiled_model, *args):
    model_state = shared.compiled_model_state
    flat_args, _ = tree_flatten(args)
    ov_inputs = [a.detach().cpu().numpy() for a in flat_args]

    if (model_state.cn_model == "None"):
        ov_inputs.reverse()

    res = compiled_model(ov_inputs)
    result = [torch.from_numpy(res[out]) for out in compiled_model.outputs]
    return result

def check_fully_supported(self, graph_module):
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

@register_backend
@fake_tensor_unsupported
def openvino_fx(subgraph, example_inputs):
    model_state = shared.compiled_model_state
    executor_parameters = None
    inputs_reversed = False
    if os.getenv("OPENVINO_TORCH_MODEL_CACHING") != "0":
        os.environ.setdefault('OPENVINO_TORCH_MODEL_CACHING', "1")
        # Create a hash to be used for caching
        model_hash_str = sha256(subgraph.code.encode('utf-8')).hexdigest()
        if (model_state.cn_model != "None" and model_state.partition_id == 0):
            model_hash_str = model_hash_str + model_state.cn_model

        if (model_state.lora_model != "None"):
            model_hash_str = model_hash_str + model_state.lora_model

        executor_parameters = {"model_hash_str": model_hash_str}
        # Check if the model was fully supported and already cached
        example_inputs.reverse()
        inputs_reversed = True
        maybe_fs_cached_name = cached_model_name(model_hash_str + "_fs", get_device(), example_inputs, cache_root_path())

        if os.path.isfile(maybe_fs_cached_name + ".xml") and os.path.isfile(maybe_fs_cached_name + ".bin"):
            if (model_state.cn_model != "None" and model_state.cn_model in maybe_fs_cached_name):
                example_inputs_reordered = []
                if (os.path.isfile(maybe_fs_cached_name + ".txt")):
                    f = open(maybe_fs_cached_name + ".txt", "r")
                    for input_data in example_inputs:
                        shape = f.readline()
                        if (str(input_data.size()) != shape):
                            for idx1, input_data1 in enumerate(example_inputs):
                                if (str(input_data1.size()).strip() == str(shape).strip()):
                                    example_inputs_reordered.append(example_inputs[idx1])
                    example_inputs = example_inputs_reordered

                # Model is fully supported and already cached. Run the cached OV model directly.
                compiled_model = openvino_compile_cached_model(maybe_fs_cached_name, *example_inputs)

                def _call(*args):
                    if (model_state.cn_model != "None" and model_state.cn_model in maybe_fs_cached_name):
                        args_reordered = []
                        if (os.path.isfile(maybe_fs_cached_name + ".txt")):
                            f = open(maybe_fs_cached_name + ".txt", "r")
                            for input_data in args:
                                shape = f.readline()
                                if (str(input_data.size()) != shape):
                                    for idx1, input_data1 in enumerate(args):
                                        if (str(input_data1.size()).strip() == str(shape).strip()):
                                            args_reordered.append(args[idx1])
                        args = args_reordered

                    res = execute_cached(compiled_model, *args)
                    model_state.partition_id = model_state.partition_id + 1
                    return res
                return _call
    else:
        maybe_fs_cached_name = None

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
                        executor_parameters=executor_parameters) #, file_name=maybe_fs_cached_name)
        return res
    return _call
