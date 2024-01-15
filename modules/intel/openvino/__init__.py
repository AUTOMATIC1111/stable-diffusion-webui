import os
import sys
import torch
import nncf

from openvino.frontend import FrontEndManager
from openvino.frontend.pytorch.fx_decoder import TorchFXPythonDecoder
from openvino.frontend.pytorch.torchdynamo.partition import Partitioner
from openvino.runtime import Core, Type, PartialShape, serialize

from torch._dynamo.backends.common import fake_tensor_unsupported
from torch._dynamo.backends.registry import register_backend
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx import GraphModule
from torch.utils._pytree import tree_flatten

from types import MappingProxyType
from hashlib import sha256
import functools

from modules import shared, devices, sd_models

NNCFNodeName = str
def get_node_by_name(self, name: NNCFNodeName) ->  nncf.common.graph.NNCFNode:
    node_ids = self._node_name_to_node_id_map.get(name, None)
    if node_ids is None:
        raise RuntimeError("Could not find a node {} in NNCFGraph!".format(name))

    node_key = f"{node_ids[0]} {name}"
    return self._nodes[node_key]
nncf.common.graph.NNCFGraph.get_node_by_name = get_node_by_name

def BUILD_MAP_UNPACK(self, inst):
        items = self.popn(inst.argval)
        # ensure everything is a dict
        items = [BuiltinVariable(dict).call_function(self, [x], {}) for x in items] # noqa: F821
        result = dict()
        for x in items:
            assert isinstance(x, ConstDictVariable) # noqa: F821
        result.update(x.items)
        self.push(
            ConstDictVariable( # noqa: F821
                result,
                dict,
                mutable_local=MutableLocal(), # noqa: F821
                **VariableTracker.propagate(items), # noqa: F821
            )
        )
tmp_torch = sys.modules["torch"]
tmp_torch.BUILD_MAP_UNPACK_WITH_CALL = BUILD_MAP_UNPACK
max_openvino_partitions = 0

DEFAULT_OPENVINO_PYTHON_CONFIG = MappingProxyType(
    {
        "use_python_fusion_cache": True,
        "allow_single_op_fusion": True,
    },
)

class OpenVINOGraphModule(torch.nn.Module):
    def __init__(self, gm, partition_id, use_python_fusion_cache, model_hash_str: str = None, file_name=""):
        super().__init__()
        self.gm = gm
        self.partition_id = partition_id
        self.executor_parameters = {"use_python_fusion_cache": use_python_fusion_cache,
                                    "model_hash_str": model_hash_str}
        self.file_name = file_name

    def __call__(self, *args):
        result = openvino_execute(self.gm, *args, executor_parameters=self.executor_parameters, partition_id=self.partition_id, file_name=self.file_name)
        return result

def get_device_list():
    core = Core()
    return core.available_devices

def get_device():
    if hasattr(shared, "opts") and len(shared.opts.openvino_devices) == 1:
        return shared.opts.openvino_devices[0]

    core = Core()
    if hasattr(shared, "opts") and len(shared.opts.openvino_devices) > 1:
        device = ""
        available_devices = shared.opts.openvino_devices.copy()
        available_devices.remove("CPU")
        for hetero_device in available_devices:
            device = f"{device},{hetero_device}"
        if "CPU" in shared.opts.openvino_devices:
            device = f"{device},CPU"
        device = f"HETERO:{device[1:]}"
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
        device = core.available_devices[-1]
        shared.log.warning(f"OpenVINO: No compatible GPU detected! Using {device}")
    return device

def get_openvino_device():
    core = Core()
    try:
        return core.get_property(get_device(), "FULL_DEVICE_NAME")
    except Exception:
        return f"OpenVINO {get_device()}"

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
        if isinstance(input_data, torch.SymInt):
            if reversed:
                inputs_str = "_" + "torch.SymInt" + inputs_str
            else:
                inputs_str += "_" + "torch.SymInt1"
        else:
            if reversed:
                inputs_str = "_" + str(input_data.type()) + str(input_data.size())[11:-1].replace(" ", "") + inputs_str
            else:
                inputs_str += "_" + str(input_data.type()) + str(input_data.size())[11:-1].replace(" ", "")
    inputs_str = sha256(inputs_str.encode('utf-8')).hexdigest()
    file_name += inputs_str

    return file_name

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
    gm,
    *args,
    executor = "openvino",
    executor_parameters = None,
    file_name = ""
):
    if executor == "openvino":
        return openvino_execute_partitioned(gm, *args, executor_parameters=executor_parameters, file_name=file_name)
    elif executor == "strictly_openvino":
        return openvino_execute(gm, *args, executor_parameters=executor_parameters, file_name=file_name)

    msg = "Received unexpected value for 'executor': {0}. Allowed values are: openvino, strictly_openvino.".format(executor)
    raise ValueError(msg)

def execute_cached(compiled_model, *args):
    flat_args, _ = tree_flatten(args)
    ov_inputs = [a.detach().cpu().numpy() for a in flat_args]

    if (shared.compiled_model_state.cn_model == []):
        ov_inputs.reverse()

    res = compiled_model(ov_inputs)
    result = [torch.from_numpy(res[out]) for out in compiled_model.outputs]
    return result

def openvino_compile(gm: GraphModule, *args, model_hash_str: str = None, file_name=""):
    core = Core()

    device = get_device()
    cache_root = shared.opts.openvino_cache_path
    global dont_use_4bit_nncf
    global dont_use_nncf

    if file_name is not None and os.path.isfile(file_name + ".xml") and os.path.isfile(file_name + ".bin"):
        om = core.read_model(file_name + ".xml")
    else:
        fe_manager = FrontEndManager()
        fe = fe_manager.load_by_framework("pytorch")

        input_shapes = []
        input_types = []
        for input_data in args:
            if isinstance(input_data, torch.SymInt):
                input_types.append(torch.SymInt)
                input_shapes.append(1)
            else:
                input_types.append(input_data.type())
                input_shapes.append(input_data.size())

        decoder = TorchFXPythonDecoder(gm, gm, input_shapes=input_shapes, input_types=input_types)

        im = fe.load(decoder)

        om = fe.convert(im)

        if (file_name is not None):
            serialize(om, file_name + ".xml", file_name + ".bin")
            if (shared.compiled_model_state.cn_model != []):
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
    if shared.opts.nncf_compress_weights and not dont_use_nncf:
        if dont_use_4bit_nncf or shared.opts.nncf_compress_weights_mode == "INT8":
            om = nncf.compress_weights(om)
        else:
            om = nncf.compress_weights(om, mode=getattr(nncf.CompressWeightsMode, shared.opts.nncf_compress_weights_mode), group_size=8, ratio=shared.opts.nncf_compress_weights_raito)

    if model_hash_str is not None:
        core.set_property({'CACHE_DIR': cache_root + '/blob'})
    dont_use_nncf = False
    dont_use_4bit_nncf = False

    compiled_model = core.compile_model(om, device)
    return compiled_model

def openvino_compile_cached_model(cached_model_path, *example_inputs):
    core = Core()
    om = core.read_model(cached_model_path + ".xml")

    global dont_use_4bit_nncf
    global dont_use_nncf

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
    if shared.opts.nncf_compress_weights and not dont_use_nncf:
        if dont_use_4bit_nncf or shared.opts.nncf_compress_weights_mode == "INT8":
            om = nncf.compress_weights(om)
        else:
            om = nncf.compress_weights(om, mode=getattr(nncf.CompressWeightsMode, shared.opts.nncf_compress_weights_mode), group_size=8, ratio=shared.opts.nncf_compress_weights_raito)

    core.set_property({'CACHE_DIR': shared.opts.openvino_cache_path + '/blob'})
    dont_use_nncf = False
    dont_use_4bit_nncf = False

    compiled_model = core.compile_model(om, get_device())
    return compiled_model

def openvino_execute(gm: GraphModule, *args, executor_parameters=None, partition_id, file_name=""):
    executor_parameters = executor_parameters or DEFAULT_OPENVINO_PYTHON_CONFIG

    use_cache = executor_parameters.get(
        "use_python_fusion_cache",
        DEFAULT_OPENVINO_PYTHON_CONFIG["use_python_fusion_cache"],
    )

    model_hash_str = executor_parameters.get("model_hash_str", None)
    if model_hash_str is not None:
        model_hash_str = model_hash_str + str(partition_id)

    if use_cache and (partition_id in shared.compiled_model_state.compiled_cache):
        compiled = shared.compiled_model_state.compiled_cache[partition_id]
    else:
        if (shared.compiled_model_state.cn_model != [] and file_name is not None
                and os.path.isfile(file_name + ".xml") and os.path.isfile(file_name + ".bin")):
            compiled = openvino_compile_cached_model(file_name, *args)
        else:
            compiled = openvino_compile(gm, *args, model_hash_str=model_hash_str, file_name=file_name)
        shared.compiled_model_state.compiled_cache[partition_id] = compiled

    flat_args, _ = tree_flatten(args)
    ov_inputs = [a.detach().cpu().numpy() for a in flat_args]

    res = compiled(ov_inputs)

    results1 = [torch.from_numpy(res[out]) for out in compiled.outputs]
    if len(results1) == 1:
        return results1[0]
    return results1

def openvino_execute_partitioned(gm: GraphModule, *args, executor_parameters=None, file_name=""):
    executor_parameters = executor_parameters or DEFAULT_OPENVINO_PYTHON_CONFIG

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

    if signature not in shared.compiled_model_state.partitioned_modules:
        shared.compiled_model_state.partitioned_modules[signature] = partition_graph(gm, use_python_fusion_cache=use_python_fusion_cache,
                                                         model_hash_str=model_hash_str, file_name=file_name)

    return shared.compiled_model_state.partitioned_modules[signature](*args)

def partition_graph(gm: GraphModule, use_python_fusion_cache: bool, model_hash_str: str = None, file_name=""):
    global max_openvino_partitions
    for node in gm.graph.nodes:
        if node.op == "call_module" and "fused_" in node.name:
            openvino_submodule = getattr(gm, node.name)
            gm.delete_submodule(node.target)
            gm.add_submodule(
                node.target,
                OpenVINOGraphModule(openvino_submodule, shared.compiled_model_state.partition_id, use_python_fusion_cache,
                        model_hash_str=model_hash_str, file_name=file_name),
            )
            shared.compiled_model_state.partition_id = shared.compiled_model_state.partition_id + 1

    return gm

def generate_subgraph_str(tensor):
    if hasattr(tensor, "weight"):
        shared.compiled_model_state.model_hash_str = shared.compiled_model_state.model_hash_str + sha256(str(tensor.weight).encode('utf-8')).hexdigest()
    return tensor

def get_subgraph_type(tensor):
    global subgraph_type
    subgraph_type.append(type(tensor))
    return tensor

@register_backend
@fake_tensor_unsupported
def openvino_fx(subgraph, example_inputs):
    global dont_use_4bit_nncf
    global dont_use_nncf
    global subgraph_type

    dont_use_4bit_nncf = False
    dont_use_nncf = False
    dont_use_faketensors = False
    executor_parameters = None
    inputs_reversed = False
    maybe_fs_cached_name = None

    subgraph_type = []
    subgraph.apply(get_subgraph_type)

    # SD 1.5 / SDXL VAE
    if (subgraph_type[0] is torch.nn.modules.conv.Conv2d and
        subgraph_type[1] is torch.nn.modules.conv.Conv2d and
        subgraph_type[2] is torch.nn.modules.normalization.GroupNorm and
        subgraph_type[3] is torch.nn.modules.activation.SiLU):

        dont_use_4bit_nncf = True
        dont_use_nncf = bool("VAE" not in shared.opts.nncf_compress_weights)

    # SD 1.5 / SDXL Text Encoder
    elif (subgraph_type[0] is torch.nn.modules.sparse.Embedding and
        subgraph_type[1] is torch.nn.modules.sparse.Embedding and
        subgraph_type[2] is torch.nn.modules.normalization.LayerNorm and
        subgraph_type[3] is torch.nn.modules.linear.Linear):

        dont_use_faketensors = True
        dont_use_nncf = bool("Text Encoder" not in shared.opts.nncf_compress_weights)

    if not shared.opts.openvino_disable_model_caching:
        os.environ.setdefault('OPENVINO_TORCH_MODEL_CACHING', "1")

        # Create a hash to be used for caching
        subgraph.apply(generate_subgraph_str)
        shared.compiled_model_state.model_hash_str = shared.compiled_model_state.model_hash_str + sha256(subgraph.code.encode('utf-8')).hexdigest()
        model_hash_str = sha256(shared.compiled_model_state.model_hash_str.encode('utf-8')).hexdigest()
        shared.compiled_model_state.model_hash_str = ""

        if (shared.compiled_model_state.cn_model != [] and shared.compiled_model_state.partition_id == 0):
            shared.compiled_model_state.shared.compiled_model_state.model_hash_str = model_hash_str + str(shared.compiled_model_state.cn_model)

        if (shared.compiled_model_state.lora_model != []):
            shared.compiled_model_state.model_hash_str = shared.compiled_model_state.model_hash_str + str(shared.compiled_model_state.lora_model)

        executor_parameters = {"model_hash_str": shared.compiled_model_state.model_hash_str}
        # Check if the model was fully supported and already cached
        example_inputs.reverse()
        inputs_reversed = True
        maybe_fs_cached_name = cached_model_name(shared.compiled_model_state.model_hash_str + "_fs", get_device(), example_inputs, shared.opts.openvino_cache_path)

        if os.path.isfile(maybe_fs_cached_name + ".xml") and os.path.isfile(maybe_fs_cached_name + ".bin"):
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

            if dont_use_faketensors:
                pass
            else:
                # Delete unused subgraphs
                subgraph = subgraph.apply(sd_models.convert_to_faketensors)
                devices.torch_gc(force=True)

            # Model is fully supported and already cached. Run the cached OV model directly.
            compiled_model = openvino_compile_cached_model(maybe_fs_cached_name, *example_inputs)

            def _call(*args):
                if (shared.compiled_model_state.cn_model != [] and str(shared.compiled_model_state.cn_model) in maybe_fs_cached_name):
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
                shared.compiled_model_state.partition_id = shared.compiled_model_state.partition_id + 1
                return res
            return _call
    else:
        os.environ.setdefault('OPENVINO_TORCH_MODEL_CACHING', "0")
        maybe_fs_cached_name = None

    if inputs_reversed:
        example_inputs.reverse()
    model = make_fx(subgraph)(*example_inputs)
    for node in model.graph.nodes:
        if node.target == torch.ops.aten.mul_.Tensor:
            node.target = torch.ops.aten.mul.Tensor
    with devices.inference_context():
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
